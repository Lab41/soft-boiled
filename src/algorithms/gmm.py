from sklearn import mixture
import numpy as np
import urlparse
# local includes
from src.utils.geo import bb_center, GeoCoord, haversine
import statsmodels.sandbox.distributions.extras as ext
import math
import gzip
import csv
import itertools
from collections import namedtuple, defaultdict

GMMLocEstimate = namedtuple('LocEstimate', ['geo_coord', 'prob'])

def get_location_from_tweet(row):
    """
    Extract location from a tweet object. If geo.coordinates not present use center of place.bounding_box.

    Args:
        row (Row): A spark sql row containing a tweet

    Retruns:
        GeoCoord: The location in the tweet
    """
    # Get true location
    if row.geo and row.geo.type == 'Point':
        ll = row.geo.coordinates
        location = GeoCoord(lat=ll[0], lon=ll[1])
    elif row.place and row.place.bounding_box and row.place.bounding_box.type =='Polygon' \
            and row.place.place_type in ['city','poi','neighborhood']:
        ll = bb_center(row.place.bounding_box.coordinates)
        location = GeoCoord(lat=ll[0], lon=ll[1])
    else:
        location = None

    return location


def tokenize_tweet(inputRow, fields):
    """
    A simple tokenizer that takes a tweet as input and, splitting on whitespace, and returns words in the tweet

    Args:
      inputRow (Row): A spark sql row containing a tweet
      fields (list): A list of field names which directs tokenize on which fields to use as source data

    Returns:
        tokens (list): A list of words appearing in the tweet
    """
    # Allow us to select which fields get pulled for model
    text = []
    if 'text' in fields:
        text.append(inputRow.text.strip())
    if 'user.location' in fields:
        try:
            text.append(inputRow.location.strip())
        except:
            text.append(inputRow.user.location.strip())
    text = ' '.join(text)

    if 'text' in fields:
        # Clean up URLs in tweet
        updates_to_make = []
        if inputRow.entities and inputRow.entities.urls:
            for url_row in inputRow.entities.urls:
                updates_to_make.append((url_row.url, urlparse.urlparse(url_row.expanded_url).netloc.replace('.', '_')))
        if inputRow.extended_entities and inputRow.extended_entities.media:
            for media_row in inputRow.extended_entities.media:
                updates_to_make.append((media_row.url, urlparse.urlparse(media_row.expanded_url).netloc.replace('.', '_')))
        for (original, new_string) in updates_to_make:
            text = text.replace(original, new_string)

    # Convert to lowercase and get remove @mentions
    tokens = []
    for item in text.lower().split():
        if not item.startswith('@'):
            tokens.append(item)
    return tokens


def get_errors(model, points):
    """
    Computes the median error for a GMM model and a set of training points

    Args:
        model (mixture.GMM): A GMM model for a word
        points (list): A list of (lat, lon) tuples

    Returns:
        median (float): The median distance to the training points from the most likely point
    """
    (best_lat, best_lon) = model.means_[np.argmax(model.weights_)]
    best_point = GeoCoord(lat=best_lat, lon=best_lon)
    errors = []
    for (lat, lon) in points:
        point = GeoCoord(lat, lon)
        error = haversine(best_point, point)
        errors.append(error)
    median = np.median(errors)
    return median


def fit_gmm_to_locations(geo_coords, max_num_components):
    """
    Searches within bounts to fit a GMM with the optimal number of components

    Args:
        geo_coords (list): A list of GeoCoord points to fit a GMM distribution to
        max_num_components (int): The maximum number of components that the GMM model can have

    Returns:
        gmm_estimate (tuple): Tuple containing the best mixture.GMM and the error of that model on the training data
    """
    # GMM Code expects numpy arrays not named tuples
    data_array = []
    for geo_coord in geo_coords:
        data_array.append([geo_coord.lat, geo_coord.lon])

    min_components = 1
    max_components = min(len(data_array)-1, max_num_components)
    models = []
    min_bic_seen = 10000000
    best_model = None
    for i in range(min_components, max_components+1):
        model = mixture.GMM(n_components=i, covariance_type='full', min_covar=0.001).fit(data_array)
        models.append(model)
        bic = model.bic(np.array(data_array))
        if bic < min_bic_seen:
            min_bic_seen = bic
            best_model = model

    median_error = get_errors(best_model, data_array)
    return (best_model, median_error)


def combine_gmms(gmms):
    """
    Takes an array of gaussian mixture models and produces a GMM that is the weighted sum of the models

    Args:
        gmms (list): A list of (mixture.GMM, median_error_on_training) models

    Returns:
        new_gmm (mixture.GMM): A single GMM model that is the weighted sum of the input gmm models
    """
    n_components = sum([g[0].n_components for g in gmms])
    covariance_type = gmms[0][0].covariance_type
    new_gmm = mixture.GMM(n_components=n_components, covariance_type=covariance_type)
    new_gmm.means_ = np.concatenate([g[0].means_ for g in gmms])
    new_gmm.covars_ = np.concatenate([g[0].covars_ for g in gmms])
    weights = np.concatenate([g[0].weights_ * ((1/max(g[1],1))**4) for g in gmms])
    new_gmm.weights_ = weights / np.sum(weights) # Normalize weights
    new_gmm.converged_ = True
    return new_gmm


def get_most_likely_point(tokens, model_bcast, radius=None):
    '''
    Create the combined GMM and find the most likely point. This function is called in a flatMap so return a list with
    0 or 1 item

    Args:
        tokens (list): list of words in tweet
        model_bcast (pyspark.Broadcast): A broadcast version of a dictionary of GMM model for the entire vocabulary
        radius (float): Distance from most likely point at which we should estimate containment probability (if not None)

    Returns:
        loc_estimate (list): A list with 0 or 1 GMMLocEstimates
    '''
    model = model_bcast.value
    models = []
    for token in tokens:
        if token in model:
            models.append(model[token])

    if len(models) > 1:
        combined_gmm = combine_gmms(models)
        (best_lat, best_lon) = combined_gmm.means_[np.argmax(combined_gmm.weights_)]
        if radius:
            prob = predict_probability_radius(combined_gmm, radius, (best_lat, best_lon))
            return [GMMLocEstimate(GeoCoord(lat=best_lat, lon=best_lon), prob)]
        else:
            return [GMMLocEstimate(GeoCoord(lat=best_lat, lon=best_lon), None)]
    elif len(models) == 1:
        (best_lat, best_lon) = models[0][0].means_[np.argmax(models[0][0].weights_)]
        if radius:
            prob = predict_probability_radius(models[0][0], radius, (best_lat, best_lon))
            return [GMMLocEstimate(GeoCoord(lat=best_lat, lon=best_lon), prob)]
        else:
            return [GMMLocEstimate(GeoCoord(lat=best_lat, lon=best_lon), None)]
    else:
        return []


def predict_probability_area(model, upper_bound, lower_bound):
    """
    Predict the probability that the true location is within a specified bounding box given a GMM model

    Args:
        model (mixture.GMM): GMM model to use
        upper_bound (list): [upper lat, right lon] of bounding box
        lower_bound (list): [lower_lat, left_lon] of bounding box

    Returns:
        total_prob (float): Probability from 0 to 1 of true location being in bounding box
    """
    total_prob = 0
    for i in range(0, len(model.weights_)):
        val = ext.mvnormcdf(upper_bound, model.means_[i], model.covars_[i], lower_bound, maxpts=2000)
        # below is necessary as a very rare occurance causes some guassians to have a result of nan
        #(likely exeedingly low probability)
        if math.isnan(val):
            pass
        else:
            weighted_val = val * model.weights_[i]
            total_prob += weighted_val
    return total_prob


def predict_probability_radius(gmm_model, radius, center_point):
    """
    Attempt to estimate the probability that the true location is within some radius of a given center point.
    Estimate is based on estimating probability in corners of bounding box and subtracting from total probability mass

    Args:
        gmm_model (mixture.GMM): GMM model to use
        radius (float): Radius from center point to include in estimate
        center_point (tuple): (lat, lon) center point

    Return:
        total_prob (float): Probability from 0 to 1 of true location being in the specified radius
    """
    total_prob = 0
    # determine the upper and lower bounds based on a km radius
    center_lat = center_point[0]
    center_lon = center_point[1]
    lat_dist = radius/111.32
    upper_lat = center_lat + lat_dist
    lower_lat = center_lat - lat_dist
    lon_dist = radius/111.32 * math.cos(math.radians(center_lat))
    right_lon = center_lon + lon_dist
    left_lon = center_lon - lon_dist

    upper_bound = [upper_lat, right_lon]
    lower_bound = [lower_lat, left_lon]
    initial_prob = predict_probability_area(gmm_model, upper_bound, lower_bound)

    #remove the corner probabilities to better estimate the area
    #determine the approximate probability distribution at the corners vs the center
    #for a completely homogenous distribution the corners are approximately 20% of the area of the square
    #as we do not have a homegenous distribution this is a better approximation
    ur_prob = np.exp(gmm_model.score(upper_bound))[0]
    ll_prob = np.exp(gmm_model.score(lower_bound))[0]
    ul_prob = np.exp(gmm_model.score([upper_lat, left_lon]))[0]
    lr_prob = np.exp(gmm_model.score([lower_lat, right_lon]))[0]
    center_prob = np.exp(gmm_model.score(center_point))[0]
    dist_adjustment = np.mean([ur_prob, ll_prob,ul_prob,lr_prob])/center_prob

    total_prob = initial_prob - (.2*dist_adjustment)*initial_prob
    if total_prob<0.0:
        total_prob =0.0
    elif total_prob>1.0:
        total_prob=1.0
    return total_prob


def predict_user_gmm(sc, tweets_to_predict, fields, model, radius=None, predict_lower_bound=0):
    """
    Takes a set of tweets and for each user in those tweets it predicts a location
    Also returned are the probability of that prediction location being w/n 100 km of the true point

    Args:
        sc (pyspark.SparkContext): Spark Context to use for execution
        tweets_to_predict (RDD): RDD of twitter Row objects
        fields (list): List of field names to extract and then use for GMM prediction
        model (dict): Dictionary of {word:(mixture.GMM, error)}
        radius (float): Distance from most likely point at which we should estimate containment probability (if not None)
        predict_lower_bound (float): Probability hreshold below which we should filter tweets

    Returns:
        loc_est_by_user (RDD): An RDD of (id_str, GMMLocEstimate)
    """

    model_bcast = sc.broadcast(model)

    tweets_by_user = tweets_to_predict.rdd.filter(lambda row: row.user!=None)\
                        .keyBy(lambda row: row.user.id_str).groupByKey()

    loc_est_by_user = tweets_by_user\
        .mapValues(lambda tweets: list(itertools.chain(*[tokenize_tweet(tweet, fields) for tweet in tweets])))\
        .flatMapValues(lambda tokens: get_most_likely_point(tokens, model_bcast, radius=radius))\
        .filter(lambda (id_str, est_loc): est_loc.prob >= predict_lower_bound or radius is None)

    return loc_est_by_user



def train_gmm(sqlCtx, table_name, fields, min_occurrences=10, max_num_components=12, where_clause=''):
    """
    Train a set of GMMs for a given set of training data

    Args:
        sqlCtx (pyspark.sql.SQLContext): Spark SQL Context to use for sql queries
        table_name (str): Table name to query for test data
        fields (list): List of field names to extract and then use for GMM prediction
        min_occurrences (int): Number of times a word must appear to be incldued in the model
        max_num_components (int): The maximum number of components that the GMM model can have
        where_clause (str): A where clause that can be applied to the query

    Returns:
        model (dict): Dictionary of {word:(mixture.GMM, error)}
    """

    tweets_w_geo = sqlCtx.sql('select geo, place, entities,  extended_entities, %s from %s where (geo.coordinates is not null \
                                    or (place is not null and place.bounding_box.type="Polygon")) %s'
                                   % (','.join(fields), table_name, where_clause))

    model = tweets_w_geo.rdd.keyBy(lambda row: get_location_from_tweet(row))\
                            .filter(lambda (location, row): location is not None)\
                            .flatMapValues(lambda row: tokenize_tweet(row, fields))\
                            .map(lambda (location, word): (word, location))\
                            .groupByKey()\
                            .filter(lambda (word, locations): len(list(locations)) >= min_occurrences)\
                            .mapValues(lambda locations: fit_gmm_to_locations(locations, max_num_components))\
                            .collectAsMap()

    # TODO: Add filter of infrequent words before move to the local context
    # Clean out words that occur less than a threshold number of times
    words_to_delete = []
    for word in model:
        (gmm, error) = model[word]
        if gmm is None:
            words_to_delete.append(word)

    for word in words_to_delete:
        del model[word]
    return model


def run_gmm_test(sc, sqlCtx, table_name, fields, model, where_clause=''):
    """
    Test a pretrained model on a table of test data

    Args:
        sc (pyspark.SparkContext): Spark Context to use for execution
        sqlCtx (pyspark.sql.SQLContext): Spark SQL Context to use for sql queries
        table_name (str): Table name to query for test data
        fields (list): List of field names to extract and then use for GMM prediction
        model (dict): Dictionary of {word:(mixture.GMM, error)}
        where_clause (str): A where clause that can be applied to the query

    Returns:
        final_result (dict): A description of the performance of the GMM Algorithm
    """
    tweets_w_geo = sqlCtx.sql('select geo, entities,  extended_entities, %s from %s where geo.coordinates is not null %s'
                                   % (','.join(fields), table_name, where_clause))

    # for each tweet calculate most likely position
    model_bcast = sc.broadcast(model)

    errors_rdd = tweets_w_geo.rdd.keyBy(lambda row: get_location_from_tweet(row))\
                                .flatMapValues(lambda row: get_most_likely_point(tokenize_tweet(row, fields), model_bcast))\
                                .map(lambda (true_geo_coord, est_loc): haversine(true_geo_coord, est_loc.geo_coord))

    errors = np.array(errors_rdd.collect())
    num_vals = tweets_w_geo.count()
    errors = errors[np.isnan(errors) == False]

    median_error = np.median(errors)
    mean_error = np.mean(errors)
    print('Median Error', median_error)
    print('Mean Error: ', mean_error)

    # calculate coverage
    try:
        coverage = len(errors)/float(num_vals)
    except ZeroDivisionError:
        coverage = np.nan

    # gather errors
    final_results = {'median': median_error, 'mean': mean_error, 'coverage': coverage,
                     'num_locs': len(errors), 'fields': fields}
    return final_results



def load_model(input_fname):
    """
    Load a pre-trained model

    Args:
        input_fname (str): Local file path to read GMM model from

    Returns:
        model (dict): A dictionary of the form {word: (mixture.GMM, error)}
    """
    model = {}
    if input_fname.endswith('.gz'):
        input_file = gzip.open(input_fname, 'rb')
    else:
        input_file = open(input_fname, 'rb')
    csv_reader = csv.reader(input_file)

    for line in csv_reader:
        word = line[0].decode('utf-8')
        error = float(line[1])
        n_components = int(line[2])
        covariance_type = line[3]
        if covariance_type != 'full':
            raise NotImplementedError('Only full covariance matricies supported')

        HEADER = 4
        NUM_ITEMS_PER_COMPONENT= 7
        means = []
        covars = []
        weights = []
        for i in range(n_components):
            lat = float(line[i*NUM_ITEMS_PER_COMPONENT + HEADER + 0])
            lon = float(line[i*NUM_ITEMS_PER_COMPONENT + HEADER + 1])
            mean = [lat, lon]
            means.append(mean)
            weight = float(line[i*NUM_ITEMS_PER_COMPONENT + HEADER + 2])
            weights.append(weight)

            vals = map(float, line[i*NUM_ITEMS_PER_COMPONENT + HEADER + 3: i*NUM_ITEMS_PER_COMPONENT + HEADER + 7])
            covar = np.array([vals[:2], vals[2:4]])
            covars.append(covar)

        new_gmm = mixture.GMM(n_components=n_components, covariance_type=covariance_type)
        new_gmm.covars_ = np.array(covars)
        new_gmm.means_ = np.array(means)
        new_gmm.weights_ = np.array(weights)
        new_gmm.converged_ = True
        model[word] = (new_gmm, error)

    return model


def save_model(model, output_fname):
    """
    Save the current model for future use

    Args:
        model (dict): A dictionary of the form {word: (mixture.GMM, error)}
        output_fname (str): Local file path to store GMM model
    """
    if output_fname.endswith('.gz'):
        output_file = gzip.open(output_fname, 'w')
    else:
        output_file = open(output_fname, 'w')
    csv_writer = csv.writer(output_file)
    LAT = 0
    LON = 1
    for word in model:
        (gmm, error) = model[word]
        row = [word.encode('utf-8'), error, gmm.n_components, gmm.covariance_type]
        for mean, weight, covar in zip(gmm.means_, gmm.weights_, gmm.covars_):
            row.extend([mean[LAT], mean[LON], weight, covar[0][0], covar[0][1], covar[1][0], covar[1][1]])
        csv_writer.writerow(row)
    output_file.close()

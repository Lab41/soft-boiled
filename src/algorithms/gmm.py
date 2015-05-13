import pickle
from sklearn import mixture
import numpy as np
import re
import urlparse
# Spark Includes
from pyspark.sql import SQLContext
# local includes
from algorithm import Algorithm
from geo import haversine

RE_REMOVE_CHARS = re.compile(r'[^a-zA-Z#_\']')
def tokenize(inputRow):
    """Initial stand in attempt at tokenizing strings
    Params:
      inputRow: a pyspark row
    Output:
     tokens: List of (word, location) tuples. Location field is the same for each tuple in a row
    """
    text = inputRow.text.strip()
    if inputRow.geo and inputRow.geo.type == 'Point':
        location = inputRow.geo.coordinates
    else:
        location = None
    updates_to_make = []
    if inputRow.entities and inputRow.entities.urls:
        for url_row in inputRow.entities.urls:
            updates_to_make.append((url_row.url, urlparse.urlparse(url_row.expanded_url).netloc.replace('.', '_')))
    if inputRow.extended_entities and inputRow.extended_entities.media:
        for media_row in inputRow.extended_entities.media:
            updates_to_make.append((media_row.url, urlparse.urlparse(media_row.expanded_url).netloc.replace('.', '_')))
    for (original, new_string) in updates_to_make:
        #print(original, new_string)
        text = text.replace(original, new_string)
    text = RE_REMOVE_CHARS.sub(' ', text)
    tokens = []
    for item in text.lower().split():
        if not item.startswith('@'):
            tokens.append(item)
    return (location, tokens)

def tokenize_w_location(inputRow):
    (location, tokens) = tokenize(inputRow)
    output_tokens = []
    for token in tokens:
        output_tokens.append((token, location))
    return output_tokens

class GMM(Algorithm):
    def __init__(self, context, options, saved_model_fname=None):
        self.options = options
        self.model = None
        if saved_model_fname:
            self.load(saved_model_fname)
        self.sc = context
        self.sqlCtx = SQLContext(context)

    @staticmethod
    def transform_tweets(tweet):
        """ tokenizes and splits tweet"""
        if tweet and tweet.text and tweet.coordinates:
            URL = re.compile('http[s]*://[^ ]+')
            AT_MENTION = re.compile(r'@[a-zA-Z0-9_]+')
            RE_TWEET = re.compile(r',RT ')
            REPLACE_CHARS = re.compile(r'[.!?@()]')
            inputString = tweet.text
            workingString = AT_MENTION.sub('AT_MENTION', inputString)
            workingString = URL.sub('URL', workingString)
            workingString = RE_TWEET.sub(',', workingString)
            workingString = REPLACE_CHARS.sub('', workingString)
            workingString = workingString.replace('\\n', ' ')
            words = workingString.lower().split()

            results = []
            for word in words:
                results.append((word, tweet.coordinates))
            return results
    @staticmethod
    def get_errors(model, points):
        (best_lat, best_lon) = model.means_[np.argmax(model.weights_)]
        errors = []
        for point in points:
            (lat, lon) = point
            error = haversine(best_lon, best_lat, lon, lat)
            errors.append(error)
        return np.median(errors)

    @staticmethod
    def fit_gmm(data_array):
        data_array = list(data_array)
        if len(data_array) < 10:
            return (None, None)
        min_components = 1
        max_components = min(len(data_array)-1, 12)
        models = []
        bics = []
        min_bic_seen = 10000000
        best_i = -1
        best_model = None
        for i in range(min_components, max_components+1):
            #print('Trying %d'%i)
            model = mixture.GMM(n_components=i, covariance_type='full', min_covar=0.001).fit(data_array)
            models.append(model)
            bic = model.bic(np.array(data_array))
            bics.append(bic)
            if bic < min_bic_seen:
                min_bic_seen = bic
                best_model = model
                best_i = i

        # TODO: Add median error and return that
        median_error = GMM.get_errors(best_model, data_array)
        return (best_model, median_error)

    def train(self, data_path):
        if 'parquet' in data_path:
            all_tweets = self.sqlCtx.parquetFile(data_path)
        else:
            all_tweets = self.sqlCtx.jsonFile(data_path)
        all_tweets.registerTempTable('tweets')
        tweets_w_geo = self.sqlCtx.sql('select geo, entities,  extended_entities, text from tweets where created_at is not null and geo.coordinates is not null')
        word_ocurrences = tweets_w_geo.flatMap(tokenize_w_location)

        # In this line we group occurrences of words, fit a gmm to each word and bring it back to the local context
        self.model = word_ocurrences.groupByKey().mapValues(GMM.fit_gmm).collectAsMap()

        # TODO: Add filter of infrequent words before move to the local context
        # Clean out words that occur less than a threshold number of times
        words_to_delete = []
        for word in self.model:
            (gmm, error) = self.model[word]
            if gmm is None:
                words_to_delete.append(word)

        for word in words_to_delete:
            del self.model[word]

    @staticmethod
    def combine_gmms(gmms):
        n_components = sum([g[0].n_components for g in gmms])
        covariance_type = gmms[0][0].covariance_type
        new_gmm = mixture.GMM(n_components=n_components, covariance_type=covariance_type)
        new_gmm.means_ = np.concatenate([g[0].means_ for g in gmms])
        new_gmm.covars_ = np.concatenate([g[0].covars_ for g in gmms])
        new_gmm.weights_ = np.concatenate([g[0].weights_ * ((1/g[1])**4) for g in gmms])
        new_gmm.converged_ = True
        return new_gmm

    @staticmethod
    def compute_error(input_val, model=None):
        (location, tokens) = input_val
        true_lat, true_lon = location
        models = []
        for token in tokens:
            if token in model:
                models.append(model[token])
        if len(models) > 1:
            combined_gmm = GMM.combine_gmms(models)
            (best_lat, best_lon) = combined_gmm.means_[np.argmax(combined_gmm.weights_)]
        elif len(models) == 1:
            (best_lat, best_lon) = models[0][0].means_[np.argmax(models[0][0].weights_)]
        else:
            return np.nan
        distance = haversine(best_lon, best_lat, true_lon, true_lat)
        return distance

    def test(self, data_path):
        if 'parquet' in data_path:
            all_tweets = self.sqlCtx.parquetFile(data_path)
        else:
            all_tweets = self.sqlCtx.jsonFile(data_path)
        all_tweets.registerTempTable('tweets')
        tweets_w_geo = self.sqlCtx.sql('select geo, entities,  extended_entities, text from tweets where created_at is not null and geo.coordinates is not null')

        # for each tweet calculate most likely position
        model = self.model
        # TODO: Explore using sc.broadcast instead of lexical closure to make this work
        def compute_error_w_model(model):
            return (lambda x: GMM.compute_error(x, model=model))

        error_function_w_closure = compute_error_w_model(model)
        errors = np.array(tweets_w_geo.map(tokenize).map(error_function_w_closure).collect())
        errors = errors[np.isnan(errors) == False]

        median_error = np.median(errors)
        mean_error = np.mean(errors)
        print('Median Error', median_error)
        print('Mean Error: ', mean_error)
        # gather errors
        final_results = {'median': median_error, 'mean': mean_error}
        return final_results

    def load(self, input_fname):
        self.model = pickle.load(open(input_fname, 'rb'))


    def save(self, output_fname):
        pickle.dump(self.model, open(output_fname, 'wb'))

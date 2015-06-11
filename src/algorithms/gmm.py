import pickle
from sklearn import mixture
import numpy as np
import re
import urlparse
# local includes
from src.algorithms.algorithm import Algorithm
from src.utils.geo import haversine
from src.utils.schema import get_twitter_schema
import statsmodels.sandbox.distributions.extras as ext
import math

class GMM(Algorithm):
    def __init__(self, context, sqlCtx, options, saved_model_fname=None):
        self.options = options
        if 'fields' not in options:
            self.options['fields'] = set(['text', 'user.location'])

        if 'where_clause' in options:
            if not self.options['where_clause'].strip().startswith('and'):
                self.options['where_clause'] = 'and ' + self.options['where_clause'].strip()
        else:
            self.options['where_clause'] = ''

        if 'temp_table_name' not in options:
            self.options['temp_table_name'] = 'tweets'

        self.model = None
        self.hasRegisteredTable = False

        if saved_model_fname:
            self.load(saved_model_fname)
        self.sc = context
        self.sqlCtx = sqlCtx

    @staticmethod
    def tokenize(inputRow, fields=set(['text'])):
        """Initial stand in attempt at tokenizing strings
        Params:
          inputRow: a pyspark row
        Output:
        (location, tokens): a tuple of the location of the tweet and a list of tokens in the tweet
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
        # Get true location
        if inputRow.geo and inputRow.geo.type == 'Point':
            location = inputRow.geo.coordinates
        else:
            location = None

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
                #print(original, new_string)
                text = text.replace(original, new_string)

        # Convert to lowercase and get remove @mentions
        tokens = []
        for item in text.lower().split():
            if not item.startswith('@'):
                tokens.append(item)
        return (location, tokens)

    @staticmethod
    def tokenize_w_location(inputRow, fields=set(['text'])):
        """ Takes the result of tokenize and turns it into a list of (word, location) tuples to be aggregated"""
        (location, tokens) = GMM.tokenize(inputRow, fields)
        output_tokens = []
        for token in tokens:
            output_tokens.append((token, location))
        return output_tokens

    @staticmethod
    def get_errors(model, points):
        """Computes the median error for a GMM and a set of training points"""
        (best_lat, best_lon) = model.means_[np.argmax(model.weights_)]
        errors = []
        for point in points:
            (lat, lon) = point
            error = haversine(best_lon, best_lat, lon, lat)
            errors.append(error)
        return np.median(errors)

    @staticmethod
    def fit_gmm(data_array):
        """ Searches within bounts to fit a GMM with the optimal number of components"""
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
            model = mixture.GMM(n_components=i, covariance_type='full', min_covar=0.001).fit(data_array)
            models.append(model)
            bic = model.bic(np.array(data_array))
            bics.append(bic)
            if bic < min_bic_seen:
                min_bic_seen = bic
                best_model = model
                best_i = i

        median_error = GMM.get_errors(best_model, data_array)
        return (best_model, median_error)

    @staticmethod
    def combine_gmms(gmms):
        """ Takes an array of gaussian mixture models and produces a GMM that is the weighted sum of the models"""
        n_components = sum([g[0].n_components for g in gmms])
        covariance_type = gmms[0][0].covariance_type
        new_gmm = mixture.GMM(n_components=n_components, covariance_type=covariance_type)
        new_gmm.means_ = np.concatenate([g[0].means_ for g in gmms])
        new_gmm.covars_ = np.concatenate([g[0].covars_ for g in gmms])
        weights = np.concatenate([g[0].weights_ * ((1/g[1])**4) for g in gmms])
        new_gmm.weights_ = weights / np.sum(weights) # Normalize weights
        new_gmm.converged_ = True
        return new_gmm

    @staticmethod
    def compute_error_using_model(input_val, model=None):
        """ Given a model that maps tokens -> GMMs this will compute the most likely point and return the distance
            from the most likely point to the true location"""
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

    @staticmethod
    def prob_mass(model, upper_bound, lower_bound):
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

    def load(self, data_path):
        options = self.sc.broadcast(self.options)
        # TODO: Make the following parameters: table name, # locations required
        if 'parquet' in data_path or 'use_parquet' in self.options and self.options['use_parquet']:
            all_tweets = self.sqlCtx.parquetFile(data_path)
        elif 'use_zip' in self.options and self.options['use_zip']:
            rdd_vals_only = self.sc.newAPIHadoopFile(data_path, 'com.cotdp.hadoop.ZipFileInputFormat',
                                      'org.apache.hadoop.io.LongWritable',
                                      'org.apache.hadoop.io.Text').map(lambda (a,b): b)
            if 'json_path' in self.options:
                schema = get_twitter_schema(self.options['json_path'])
                all_tweets = self.sqlCtx.jsonRDD(rdd_vals_only, schema=schema)
            else:
                all_tweets = self.sqlCtx.jsonRDD(rdd_vals_only)
        else:
            if 'json_path' in self.options:
                schema = get_twitter_schema(self.options['json_path'])
                all_tweets = self.sqlCtx.jsonFile(data_path, schema)
            else:
                all_tweets = self.sqlCtx.jsonFile(data_path)

        self.all_tweets = all_tweets
        return all_tweets

    def train(self, all_tweets):
        """ Train a set of GMMs for a given set of training data"""
        if not self.hasRegisteredTable:
            all_tweets.registerTempTable(self.options['temp_table_name'])
            self.hasRegisteredTable = True

        tweets_w_geo = self.sqlCtx.sql('select geo, entities,  extended_entities, %s from %s where geo.coordinates is not null %s'
                                       % (','.join(list(self.options['fields'])), self.options['temp_table_name'],
                                          self.options['where_clause']))

        def tokenize_with_defaults(fields):
            return (lambda x: GMM.tokenize_w_location(x, fields=fields))
        word_ocurrences = tweets_w_geo.flatMap(tokenize_with_defaults(self.options['fields']))

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

    def test(self, all_tweets):
        """ Test a pretrained model on a set of test data"""
        if not self.hasRegisteredTable:
            all_tweets.registerTempTable(self.options['temp_table_name'])
            self.hasRegisteredTable = True

        tweets_w_geo = self.sqlCtx.sql('select geo, entities,  extended_entities, %s from %s where geo.coordinates is not null %s'
                                       % (','.join(list(self.options['fields'])), self.options['temp_table_name'],
                                          self.options['where_clause']))

        # for each tweet calculate most likely position
        model = self.model
        # TODO: Explore using sc.broadcast instead of lexical closure to make this work
        def compute_error_w_model(model):
            return (lambda x: GMM.compute_error_using_model(x, model=model))

        def tokenize_with_defaults(fields):
            return (lambda x: GMM.tokenize(x, fields=fields))

        error_function_w_closure = compute_error_w_model(model)
        errors = np.array(tweets_w_geo.map(tokenize_with_defaults(self.options['fields'])).map(error_function_w_closure).collect())
        num_vals = len(errors)
        errors = errors[np.isnan(errors) == False]

        median_error = np.median(errors)
        mean_error = np.mean(errors)
        print('Median Error', median_error)
        print('Mean Error: ', mean_error)
        # gather errors
        final_results = {'median': median_error, 'mean': mean_error, 'coverage': len(errors)/float(num_vals),
                         'num_locs': len(errors), 'options': self.options}
        return final_results

    def load(self, input_fname):
        """Load a pre-trained model"""
        self.model = pickle.load(open(input_fname, 'rb'))

    def save(self, output_fname):
        """Save the current model for future use"""
        pickle.dump(self.model, open(output_fname, 'wb'))

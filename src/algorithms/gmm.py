import pickle
from sklearn import mixture
import numpy as np
import re
import urlparse
# Spark Includes
from pyspark.sql import SQLContext
# local includes
from algorithm import Algorithm

RE_REMOVE_CHARS = re.compile(r'[^a-zA-Z#\']')
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
            tokens.append((item, location))
    return tokens

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
    def fit_gmm(data_array):
        data_array = list(data_array)
        if len(data_array) < 10:
            return None
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
        return best_model

    def train(self, data_path):
        if 'parquet' in data_path:
            all_tweets = self.sqlCtx.parquetFile(data_path)
        else:
            all_tweets = self.sqlCtx.jsonFile(data_path)
        all_tweets.registerTempTable('tweets')
        tweets_w_geo = self.sqlCtx.sql('select geo, entities,  extended_entities, text from tweets where created_at is not null and geo.coordinates is not null')
        word_ocurrences = tweets_w_geo.flatMap(tokenize)
        self.model = word_ocurrences.groupByKey().mapValues(GMM.fit_gmm).collectAsMap()


        # Collect as Map in local context
    def test(self, data_path):
        pass

    def load(self, input_fname):
        self.model = pickle.load(open(fname, 'rb'))


    def save(self, output_fname):
        pickle.dump(self.model, open(fname, 'wb'))

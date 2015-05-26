import collections
import time
import numpy as np
import pickle
# local includes
from src.algorithms.algorithm import Algorithm
from src.utils.geo import haversine, median_point
from src.utils.schema import get_twitter_schema

class SLP(Algorithm):
    def __init__(self, context, sqlCtx, options, saved_model_fname=None):
        self.options = options
        if 'num_iters' not in options:
            self.options['num_iters'] = 5

        if 'hold_out' not in options:
            self.options['hold_out'] = set(['9'])
        self.model = None
        if saved_model_fname:
            self.load(saved_model_fname)
        self.sc = context
        self.sqlCtx = sqlCtx

        #self.broadcast_options = self.sc.broadcast(options)

    @staticmethod
    def get_at_mentions(inputRow):
        output = []
        if inputRow.user_mentions and len(inputRow.user_mentions) > 0:
            for user_mention in inputRow.user_mentions:
                src = inputRow.id_str
                dst = user_mention.id_str
                if src and dst:
                    if src < dst:
                        canoncial = src + '_' + dst
                    else:
                        canoncial = dst + '_' + src
                    output.append((canoncial, (src, dst)))
        return output

    @staticmethod
    def get_frequency(input_vals):
        output_dict = collections.defaultdict(int)
        for val in input_vals:
            output_dict[val] += 1
        return output_dict

    @staticmethod
    def make_bidirectional(inputDict):
        bidirectional_dict = {}
        for src in inputDict:
            for dst in inputDict[src]:
                if dst in inputDict and src in inputDict[dst]:
                    if src not in bidirectional_dict:
                        bidirectional_dict[src] = collections.defaultdict(int)
                    bidirectional_dict[src][dst] += 1
                    if dst not in bidirectional_dict:
                        bidirectional_dict[dst] = collections.defaultdict(int)
                    bidirectional_dict[dst][src] += 1
        return bidirectional_dict

    def train(self, data_path):
        options = self.sc.broadcast(self.options)
        # TODO: Make the following parameters: table name, # locations required
        if 'parquet' in data_path:
            all_tweets = self.sqlCtx.parquetFile(data_path)
        else:
            if 'json_path' in self.options:
                schema = get_twitter_schema(self.options['json_path'])
                all_tweets = self.sqlCtx.jsonFile(data_path, schema)
            else:
                all_tweets = self.sqlCtx.jsonFile(data_path)
        all_tweets.registerTempTable('tweets')

        print 'Building edge list'
        # Build full_edge_list
        # Build Bi-directional graph
        # the first flatMap turns src, [dsts]) -> [(cannonical order, (src, dst),...]
        # Group by key turns that into [(canoncial order, [(src,dst), (src, dst)..), ...
        # Filter removes any edges that aren't bidirectional
        # The 2nd flatMap turns [(canoncial order, [(src,dst), (src, dst)..), ...] -> [(src1, dst1), (src1, dst2)]
        # coalesce then reduces the number of parittions in the edge list
        full_edge_list = self.sqlCtx.sql('select user.id_str, entities.user_mentions from tweets where entities.user_mentions is not null')\
            .flatMap(SLP.get_at_mentions).groupByKey()\
            .filter(lambda (a,b): len(set(b)) > 1)\
            .flatMap(lambda (a,b): list(b)).coalesce(300)
        full_edge_list.cache()

        print 'Finding known user locations'
        # Find Known user locations
        # First map turns Row(id_str, coordinates) -> (id_str, coordinates)
        # Group by key turns (id_str, coordinates -> (id_str, [coordinates1,coordinates2,..])
        # Filter removes enteries without at least 3 locations
        # Calculate the median point of the locations (id_str, [coordinates1,..]) -> (id_str, median_location)
        # coalesce then reduces the number of partitions
        original_user_locations = self.sqlCtx.sql('select user.id_str, geo.coordinates from tweets where geo.coordinates is not null')\
            .map(lambda a: (a.id_str, a.coordinates))\
            .groupByKey().filter(lambda (a,b):len(b) > 3).mapValues(median_point).mapValues(lambda (a,b): a).coalesce(300)

        print 'Filtering out user locations that end in:', ','.join(list(self.options['hold_out']))
        # Filter users that will be in the test set

        filter_function = lambda (a,b): a[-1] not in options.value['hold_out']
        original_user_locations = original_user_locations.filter(filter_function)
        original_user_locations.cache()
        # Propagate locations
        updated_locations = original_user_locations

        print 'Building a filtered edge list'
        # Build a filtered edge list so we don't ever try to approximate the known user locations
        filtered_edge_list = full_edge_list.keyBy(lambda (a, b): b).leftOuterJoin(updated_locations)\
                .map(lambda (a, b): b).filter(lambda (a,b): b is None).map(lambda (a,b): a)
        filtered_edge_list.cache()

        # Keep track so number of original to control number of partitions through iterations
        num_partitions = updated_locations.getNumPartitions()

        print 'Begining iterations'
        # Perform iterations
        for i in range(self.options['num_iters']):
            start_time = time.time()
            adj_list_w_locations = filtered_edge_list.join(updated_locations).map(lambda (a,b): (b[0], b[1])).groupByKey()
            new_locations = adj_list_w_locations.map(lambda (a,b): (a, median_point(b))).filter(lambda (a,b): b is not None)\
                .filter(lambda (a,b): b[1] < 50).mapValues(lambda (a,b): a)
            updated_locations = new_locations.union(original_user_locations).coalesce(num_partitions)
            print 'Completed iteration: ', i,' in ', time.time()-start_time

        self.updated_locations = updated_locations
        self.updated_locations_local = updated_locations.collect()

    def test(self, data_path):
        # Push config to all nodes
        options = self.sc.broadcast(self.options)
        if 'parquet' in data_path:
            all_tweets = self.sqlCtx.parquetFile(data_path)
        else:
            if 'json_path' in self.options:
                schema = get_twitter_schema(self.options['json_path'])
                all_tweets = self.sqlCtx.jsonFile(data_path, schema)
            else:
                all_tweets = self.sqlCtx.jsonFile(data_path)
        all_tweets.registerTempTable('tweets')

        # Find Known user locations
        # First map turns Row(id_str, coordinates) -> (id_str, coordinates)
        # Group by key turns (id_str, coordinates -> (id_str, [coordinates1,coordinates2,..])
        # Filter removes enteries without at least 3 locations
        # Calculate the median point of the locations (id_str, [coordinates1,..]) -> (id_str, median_location)
        # coalesce then reduces the number of partitions
        original_user_locations = self.sqlCtx.sql('select user.id_str, geo.coordinates from tweets where geo.coordinates is not null')\
            .map(lambda a: (a.id_str, a.coordinates))\
            .groupByKey().filter(lambda (a,b):len(b) > 3).mapValues(median_point).mapValues(lambda (a,b): a).coalesce(300)

        # Filter users that might have been in training set
        filter_function = lambda (a,b): a[-1] in options.value['hold_out']
        original_user_locations = original_user_locations.filter(filter_function)
        number_locations = original_user_locations.count()

        found_locations = original_user_locations.join(self.updated_locations)
        found_locations_local = found_locations.collect()
        print 'Number of Found Locations: ', len(found_locations_local)
        errors = []
        for (id_str, ll_tuple) in found_locations_local:
            (ll_1,ll_2) = ll_tuple
            errors.append(haversine(ll_1[1], ll_1[0], ll_2[1], ll_2[0]))

        median_error = np.median(errors)
        mean_error = np.mean(errors)
        print('Median Error', median_error)
        print('Mean Error: ', mean_error)
        # gather errors
        final_results = {'median': median_error, 'mean': mean_error, 'coverage': len(errors)/float(number_locations),
                         'data_path':data_path, 'options': self.options}
        return final_results


    def load(self, input_fname):
        """Load a pre-trained model"""
        self.updated_locations_local = pickle.load(open(input_fname, 'rb'))
        # pull into spark context
        self.updated_locations = self.sc.parallelize(self.updated_locations_local)


    def save(self, output_fname):
        """Save the current model for future use"""
        pickle.dump(self.updated_locations_local, open(output_fname, 'wb'))

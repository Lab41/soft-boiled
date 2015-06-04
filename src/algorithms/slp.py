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

        if 'num_points_req' not in options:
            self.options['num_points_req'] = 3

        if 'dispersion_threshold' not in options:
            self.options['dispersion_threshold'] = None # km

        self.model = None
        if saved_model_fname:
            self.load(saved_model_fname)
        self.sc = context
        self.sqlCtx = sqlCtx

        self.updated_locations = None
        self.original_user_locations = None
        self.filtered_edge_list = None
        self.updated_locations_local = None

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
    def filter_non_bidirectional(edges):
        """ Take in a list of edges and output number of bidirectional edges"""
        edges = list(edges)
        if len(edges) == 0:
            return []

        counts = [0, 0]
        for src, dst in edges:
            if src < dst:
                counts[0] += 1
            else:
                counts[1] += 1
        num_edges_to_output = min(counts)
        src, dst = edges[0]
        output_vals = []
        for i in range(num_edges_to_output):
            output_vals.append((src, dst))
            output_vals.append((dst, src))
        return output_vals

    def train(self, data_path):
        options = self.sc.broadcast(self.options)
        # TODO: Make the following parameters: table name, # locations required
        if 'parquet' in data_path or 'use_parquet' in self.options and self.options['use_parquet']:
            all_tweets = self.sqlCtx.parquetFile(data_path)
        else:
            if 'json_path' in self.options:
                schema = get_twitter_schema(self.options['json_path'])
                all_tweets = self.sqlCtx.jsonFile(data_path, schema)
            else:
                all_tweets = self.sqlCtx.jsonFile(data_path)
        all_tweets.registerTempTable('tweets')

        # Helper function exploits python closure to pass options to map tasks
        def median_point_w_options_generator(num_points_req, dispersion_threshold):
            return (lambda x: median_point(x, num_points_req=num_points_req, return_dispersion=False,
                                           dispersion_treshold=dispersion_threshold))

        print 'Building edge list'
        # Build full_edge_list
        # Build Bi-directional graph
        # the first flatMap turns src, [dsts]) -> [(cannonical order, (src, dst),...]
        # Group by key turns that into [(canoncial order, [(src,dst), (src, dst)..), ...
        # The 2nd flatMap turns filters out non-bidirectional and
        #    transforms to[(canoncial order, [(src,dst), (src, dst)..), ...] -> [(src1, dst1), (src1, dst2)]
        # coalesce then reduces the number of parittions in the edge list
        full_edge_list = self.sqlCtx.sql('select user.id_str, entities.user_mentions from tweets where size(entities.user_mentions) > 0')\
            .flatMap(SLP.get_at_mentions).groupByKey()\
            .flatMap(lambda (a,b): SLP.filter_non_bidirectional(b)).coalesce(300)
        full_edge_list.cache()

        print 'Finding known user locations'
        # Find Known user locations
        # First map turns Row(id_str, coordinates) -> (id_str, coordinates)
        # Group by key turns (id_str, coordinates -> (id_str, [coordinates1,coordinates2,..])
        # Calculate the median point of the locations (id_str, [coordinates1,..]) -> (id_str, median_location)
        # coalesce then reduces the number of partitions
        median_point_w_options = median_point_w_options_generator(self.options['num_points_req'],self.options['dispersion_threshold'])
        original_user_locations = self.sqlCtx.sql('select user.id_str, geo.coordinates from tweets where geo.coordinates is not null')\
            .map(lambda a: (a.id_str, a.coordinates))\
            .groupByKey().flatMapValues(lambda input_locations:
                                            median_point_w_options(input_locations)).coalesce(300)

        # Save a reference to all locations if we are going to test immediately afterwards
        self.all_user_locations = original_user_locations
        print 'Filtering out user locations that end in:', ','.join(list(self.options['hold_out']))
        filter_function = lambda (a,b): a[-1] not in options.value['hold_out']
        original_user_locations = original_user_locations.filter(filter_function)
        original_user_locations.cache()
        # Propagate locations
        updated_locations = original_user_locations

        print 'Building a filtered edge list'
        # Build a filtered edge list so we don't ever try to approximate the known user locations
        filtered_edge_list = full_edge_list.keyBy(lambda (a, b): b).leftOuterJoin(updated_locations)\
                .flatMap(lambda (a,b): [b[0]] if b is not None else [])
        filtered_edge_list.cache()

        self.updated_locations = updated_locations
        self.original_user_locations = original_user_locations
        self.filtered_edge_list = filtered_edge_list

        print 'Begining iterations'
        # Perform iterations
        start_time = time.time()
        for i in range(self.options['num_iters']):
            if i + 1 == self.options['num_iters']:
                self.do_iteration(True)
            else:
                self.do_iteration(False)

        print 'Completed training', time.time() - start_time

    def do_iteration(self, pull_to_local_ctx=False):
        # Use closure to encode options for median value
        def median_point_w_options_generator(num_points_req, dispersion_threshold):
            return (lambda x: median_point(x, num_points_req=num_points_req, return_dispersion=False,
                                           dispersion_treshold=dispersion_threshold))

        # Keep track so number of original to control number of partitions through iterations
        num_partitions = self.updated_locations.getNumPartitions()

        start_time = time.time()
        # Create edge list (src -> dst) where we attach known locations to "src"  and then group by dst
        # end result is is [(dst, [all known locations of neighbors])]
        adj_list_w_locations = self.filtered_edge_list.join(self.updated_locations).map(lambda (a,b): (b[0], b[1])).groupByKey()

        # For each "dst" calculate median point of known neighbors
        median_point_w_options = median_point_w_options_generator(self.options['num_points_req'],self.options['dispersion_threshold'])
        new_locations = adj_list_w_locations.flatMapValues(lambda input_locations:median_point_w_options(input_locations))

        # Join back in original locations to estimated locations to get all locations
        self.updated_locations = new_locations.union(self.original_user_locations).coalesce(num_partitions)

        # If we want to count for this iteration how many known locations there are do that here
        if 'suppress_count' in self.options and self.options['suppress_count']:
            new_count = -1
        else:
            new_count = self.updated_locations.count()
        print 'Completed iteration in:', time.time()-start_time, new_count
        
        if pull_to_local_ctx:
            start_time = time.time()
            self.updated_locations_local = self.updated_locations.collect()
            print 'Pulled to local context', time.time()-start_time

    def test(self, data_path, skip_load=False):
        # Push config to all nodes
        options = self.sc.broadcast(self.options)
        if skip_load and  self.all_user_locations is not None:
            # If we've just trained then there is no need to go back to original data
            original_user_locations = self.all_user_locations
        else:
            if 'parquet' in data_path or 'use_parquet' in self.options and self.options['use_parquet']:
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
            def median_point_w_options_generator(num_points_req, dispersion_threshold):
                return (lambda x: median_point(x, num_points_req=num_points_req, return_dispersion=False, dispersion_treshold=dispersion_threshold))

            f = median_point_w_options_generator(self.options['num_points_req'],self.options['dispersion_threshold'])
            original_user_locations = self.sqlCtx.sql('select user.id_str, geo.coordinates from tweets where geo.coordinates is not null')\
                .map(lambda a: (a.id_str, a.coordinates))\
                .groupByKey().flatMapValues(lambda input_locations:f(input_locations)).coalesce(300)

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
                         'num_locs': number_locations, 'data_path':data_path, 'options': self.options}
        return final_results


    def load(self, input_fname):
        """Load a pre-trained model"""
        self.updated_locations_local = pickle.load(open(input_fname, 'rb'))
        # pull into spark context
        self.updated_locations = self.sc.parallelize(self.updated_locations_local)


    def save(self, output_fname):
        """Save the current model for future use"""
        pickle.dump(self.updated_locations_local, open(output_fname, 'wb'))

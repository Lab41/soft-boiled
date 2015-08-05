import collections
import time
import numpy as np
import pickle
import pandas as pd
# local includes
from src.algorithms.algorithm import Algorithm
from src.utils.geo import haversine, median_point
from src.utils.schema import get_twitter_schema
import gzip
import csv

class SLP(Algorithm):
    def __init__(self, context, sqlCtx, options, saved_model_fname=None):
        self.options = options
        if 'num_iters' not in options:
            self.options['num_iters'] = 5

        if 'hold_out' not in options:
            self.options['hold_out'] = set(['9'])

        if 'num_located_neighbors_req' not in options:
            self.options['num_located_neighbors_req'] = 3

        if 'num_points_req_for_known' not in options:
            self.options['num_points_req_for_known'] = self.options['num_located_neighbors_req']

        if 'dispersion_threshold' not in options:
            self.options['dispersion_threshold'] = None # km

        if 'home_radius_for_known' not in options:
            self.options['home_radius_for_known'] = None

        if 'temp_table_name' not in options:
            self.options['temp_table_name'] = 'tweets'

        self.model = None
        if saved_model_fname:
            self.load(saved_model_fname)
        self.sc = context
        self.sqlCtx = sqlCtx

        self.updated_locations = None
        self.original_user_locations = None
        self.filtered_edge_list = None
        self.updated_locations_local = None

        self.iterations_completed = 0
        self.predictions_curve = None

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

    def load_data(self, data_path):
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

    def train(self, all_tweets, predictions_curve=None):
        options = self.sc.broadcast(self.options)
        all_tweets.registerTempTable(self.options['temp_table_name'])

        # Helper function exploits python closure to pass options to map tasks
        def median_point_w_options_generator(num_points_req_for_known, home_radius_for_known):
            return (lambda x: median_point(x, num_points_req=num_points_req_for_known, return_dispersion=True,
                                           dispersion_treshold=home_radius_for_known))

        print 'Building edge list'
        # Build full_edge_list
        # Build Bi-directional graph
        # the first flatMap turns src, [dsts]) -> [(cannonical order, (src, dst),...]
        # Group by key turns that into [(canoncial order, [(src,dst), (src, dst)..), ...
        # The 2nd flatMap turns filters out non-bidirectional and
        #    transforms to[(canoncial order, [(src,dst), (src, dst)..), ...] -> [(src1, dst1), (src1, dst2)]
        # coalesce then reduces the number of parittions in the edge list
        full_edge_list = self.sqlCtx.sql('select user.id_str, entities.user_mentions from %s where size(entities.user_mentions) > 0'%\
            self.options['temp_table_name'])\
            .flatMap(SLP.get_at_mentions).groupByKey()\
            .flatMap(lambda (a,b): SLP.filter_non_bidirectional(b)).coalesce(300)
        full_edge_list.cache()
        self.full_edge_list = full_edge_list

        print 'Finding known user locations'
        # Find Known user locations
        # First map turns Row(id_str, coordinates) -> (id_str, coordinates)
        # Group by key turns (id_str, coordinates -> (id_str, [coordinates1,coordinates2,..])
        # Calculate the median point of the locations (id_str, [coordinates1,..]) -> (id_str, median_location)
        # coalesce then reduces the number of partitions
        median_point_w_options = median_point_w_options_generator(self.options['num_points_req_for_known'],\
                                                                  self.options['home_radius_for_known'])
        original_user_locations = self.sqlCtx.sql('select user.id_str, geo.coordinates from %s where geo.coordinates is not null'%\
            self.options['temp_table_name'])\
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

        if predictions_curve is None:
            print 'Building the error estimation curve'
            # For the users in the full edge list, determine all neighbors median point of the neighbors
            # Define a new median points generator which now returns the neighbor dispersion and standard dev of the dispersion
            def median_point_w_options_generator(num_located_neighbors_req, dispersion_threshold):
                return (lambda x: median_point(x, num_points_req=num_located_neighbors_req, return_dispersion=True,
                                               dispersion_treshold=dispersion_threshold, use_usr_ids=True))
            median_point_w_options = median_point_w_options_generator(self.options['num_points_req_for_known'],\
                                                                      self.options['home_radius_for_known'])

            user_location_only = original_user_locations.map(lambda (a,b): (a, b[0]))
            adj_list_w_locations = full_edge_list.join(user_location_only).map(lambda (a,b): (b[0], (b[1],a))).groupByKey()
            neighbor_locations = adj_list_w_locations.flatMapValues(lambda input_locations:median_point_w_options(input_locations))
            network_info = user_location_only.join(neighbor_locations)

            std_mults = network_info.map\
                (lambda (id_str,(lat0,(lat1, disp, mean_dis, std_dev))) : (haversine(lat0[1], lat0[0], lat1[1], lat1[0]) - disp)/std_dev)

            std_mults_loc = std_mults.collect()
            sorted_vals = np.sort(std_mults_loc)
            yvals=np.arange(len(sorted_vals))/float(len(sorted_vals))
            self.predictions_curve = pd.DataFrame(np.column_stack((sorted_vals, yvals)), columns=["std_range", "pct_within_med"])
        else:
            self.predictions_curve = predictions_curve

        print 'Building a filtered edge list'
        # Build a filtered edge list so we don't ever try to approximate the known user locations
        filtered_edge_list = full_edge_list.keyBy(lambda (a, b): b).leftOuterJoin(updated_locations)\
                .flatMap(lambda (a,b): [b[0]] if b[1] is None else [])
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
        def median_point_w_options_generator(num_located_neighbors_req, dispersion_threshold):
            return (lambda x: median_point(x, num_points_req=num_located_neighbors_req, return_dispersion=True,
                                           dispersion_treshold=dispersion_threshold, use_usr_ids=True))

        # Keep track so number of original to control number of partitions through iterations
        num_partitions = self.updated_locations.getNumPartitions()

        start_time = time.time()
        # Create edge list (src -> dst) where we attach known locations to "src"  and then group by dst
        # end result is is [(dst, [all known locations of neighbors])]
        location_only = self.updated_locations.map(lambda (a,b): (a, b[0]))
        adj_list_w_locations = self.filtered_edge_list.join(location_only).map(lambda (a,b): (b[0], (b[1],a))).groupByKey()

        # For each "dst" calculate median point of known neighbors
        median_point_w_options = median_point_w_options_generator(self.options['num_located_neighbors_req'],self.options['dispersion_threshold'])
        new_locations = adj_list_w_locations.flatMapValues(lambda input_locations:median_point_w_options(input_locations))

        # Join back in original locations to estimated locations to get all locations
        self.updated_locations = new_locations.union(self.original_user_locations).coalesce(num_partitions)

        # If we want to count for this iteration how many known locations there are do that here
        if 'suppress_count' in self.options and self.options['suppress_count']:
            new_count = -1
        else:
            new_count = self.updated_locations.count()
        print 'Completed iteration in:', time.time()-start_time, new_count
        self.iterations_completed += 1

        if pull_to_local_ctx:
            start_time = time.time()
            self.updated_locations_local = self.updated_locations.collect()
            print 'Pulled to local context', time.time()-start_time

    def test(self, all_tweets, skip_load=False):
        # Push config to all nodes
        options = self.sc.broadcast(self.options)
        all_tweets.registerTempTable(self.options['temp_table_name'])

        if skip_load and  self.all_user_locations is not None:
            # If we've just trained then there is no need to go back to original data
            original_user_locations = self.all_user_locations
        else:

            # Find Known user locations
            # First map turns Row(id_str, coordinates) -> (id_str, coordinates)
            # Group by key turns (id_str, coordinates -> (id_str, [coordinates1,coordinates2,..])
            # Filter removes enteries without at least 3 locations
            # Calculate the median point of the locations (id_str, [coordinates1,..]) -> (id_str, median_location)
            # coalesce then reduces the number of partitions
            def median_point_w_options_generator(num_points_req, dispersion_threshold):
                return (lambda x: median_point(x, num_points_req=num_points_req, return_dispersion=False, dispersion_treshold=dispersion_threshold))

            f = median_point_w_options_generator(self.options['num_points_req_for_known'],self.options['dispersion_threshold'])
            original_user_locations = self.sqlCtx.sql('select user.id_str, geo.coordinates from %s where geo.coordinates is not null'%\
                self.options['temp_table_name'])\
                .map(lambda a: (a.id_str, a.coordinates))\
                .groupByKey().flatMapValues(lambda input_locations:f(input_locations)).coalesce(300)

        # Filter users that might have been in training set
        filter_function = lambda (a,b): a[-1] in options.value['hold_out']
        original_user_locations = original_user_locations.filter(filter_function)
        number_locations = original_user_locations.count()

        found_locations = original_user_locations.join(self.updated_locations.map(lambda (a,b): (a, b[0])))
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
                         'num_locs': number_locations,
                         'iterations_completed': self.iterations_completed, 'options': self.options}

        return final_results

    def predict_probability_radius(self, dist, median_dist, std_dev):
        try:
            dist_diff = dist-median_dist
            if std_dev>0:
                stdev_mult = dist_diff/std_dev
            else:
                stdev_mult=0
            rounded_stdev = np.around(stdev_mult, decimals=3)
            predict_pct_median=0
            max_std = np.max(self.predictions_curve["std_range"])
            min_std = np.min(self.predictions_curve["std_range"])
            if (rounded_stdev< max_std and rounded_stdev>min_std) :
                predict_med = self.predictions_curve.ix[(self.predictions_curve.std_range-rounded_stdev).abs().argsort()[:1]]
                predict_pct_median = predict_med.iloc[0]['pct_within_med']
            elif rounded_stdev< max_std:
                predict_pct_median = 1

        except:
            predict_pct_median = None

        prob = predict_pct_median
        return prob


    def predict_probability_area(self, upper_bound, lower_bound, center, med_error, std_dev):
        #For now this function will return the minimum and maximum probability using the circle prediction algorithm
        (lat, lon) = center
        (max_lat, max_lon) = upper_bound
        (min_lat, min_lon) = upper_bound
        top_dist = haversine(lon, lat, lon, max_lat)
        bottom_dist = haversine(lon, lat, lon, min_lat)
        r_dist = haversine(lon, lat, max_lon, lat)
        l_dist = haversine(lon, lat, min_lon, lat)
        min_dist = min([top_dist, bottom_dist, r_dist, l_dist])
        max_dist = max([top_dist, bottom_dist, r_dist, l_dist])
        min_prob = SLP.predict_probability_radius(min_dist, med_error, std_dev)
        max_prob = SLP.predict_probability_radius(max_dist, med_error, std_dev)
        return (min_prob, max_prob)

    def load_model(self, input_fname):
        """Load a pre-trained model"""
        if input_fname.endswith('.gz'):
            input_file = gzip.open(input_fname, 'rb')
        else:
            input_file = open(input_fname, 'rb')
        csv_reader = csv.reader(input_file)
        self.updated_locations_local = []
        self.original_user_locations_local = []
        for usr_id, lat, lon, median, mean, std_dev in csv_reader:
            latlon = [float(lat), float(lon)]
            if len(median) > 0:
                # Estimated user location
                self.updated_locations_local.append((usr_id, latlon))
            else:
                self.original_user_locations_local.append((usr_id, latlon))

        self.updated_locations = self.sc.parallelize(self.updated_locations_local)
        self.original_user_locations = self.sc.parallelize(self.original_user_locations_local)


    def save_model(self, output_fname):
        """Save the current model for future use"""
        if output_fname.endswith('.gz'):
            output_file = gzip.open(output_fname, 'w')
        else:
            output_file = open(output_fname, 'w')
        csv_writer = csv.writer(output_file)

        for user_location in self.updated_locations_local:
            usr_id = user_location[0]
            results_object = user_location[1]
            lat = results_object[0][0]
            lon = results_object[0][1]
            dispersion = results_object[1]
            mean = results_object[2]
            std_dev = results_object[3]

            csv_writer.writerow([usr_id, lat, lon, dispersion, mean, std_dev])

        output_file.close()

import csv
import matplotlib.pyplot as plt
import numpy as np
from src.algorithms.slp import haversine, median

class EstimatorCurve:
    '''
    The EstimatorCurve class is used to assess the confidence of a predicted location for SLP.

    Attributes:
        w_stdev  (numpy arr): A two dimensional numpy array representing the estimator curve. The x
            axis is the standard deviations and y axis is the probability. The curve is a CDF. This
            curve is generated from known locations where at least two neighbors are at different
            locations.
        wo_stdev (numpy_arr): A two dimensional numpy array representing the estimator curve

    '''

    def __init__(self, w_stdev, wo_stdev):
        self.w_stdev = w_stdev
        self.wo_stdev = wo_stdev

    def predict_probability_area(self, upper_bound, lower_bound, estimated_loc):
        '''
        Given a prediction and a bounding box this will return a confidence range
        for that prediction

        Args:
            upper_bound (geoCoord): bounding box top right geoCoord
            lower_bound (geoCoord): bounding box bottom left geoCoord
            estimated_loc (LocEstimate): geoCoord of the estimated location

        Returns:
            Probability Tuple(Tuple(float,float)): A probability range tuple (min probability, max probability)
         '''

        geo = estimated_loc.geo_coord

        top_dist = haversine(geo, geoCoord(upper_bound.lat, geo.lon))
        bottom_dist = haversine(geo, geoCoord(lower_bound.lat, geo.lon))

        r_dist = haversine(geo, geoCoord(geo.lat, upper_bound.lon))
        l_dist = haversine(geo, geoCoord(geo.lat, lower_bound.lon))
        min_dist = min([top_dist, bottom_dist, r_dist, l_dist])
        max_dist = max([top_dist, bottom_dist, r_dist, l_dist])

        #min_prob = self.lookup( (min_dist- med_error)/std_dev)
        #max_prob = self.lookup( (max_dist - med_error)/ std_dev)

        return (self.lookup(min_dist/estimated_loc.dispersion_std_dev),\
                self.lookup(max_dist, estimated_loc.dispersion_std_dev))

    @staticmethod
    def load_from_rdds(locs_known, edges, desired_samples=1000):
        '''
        Creates an EstimatorCurve

        Args:
            locs_known (rdd of LocEstimate): RDD of locations that are known
            edges (rdd of (src_id (dest_id, weight)): RDD of edges in the network
            desired_samples (int): Limit the curve to just a sample of data

        Returns:
            EstimatorCurve:  A new EstimatorCurve representing the known input data
        '''

        # Filter edge list so we never attempt to estimate a "known" location
        known_edges = edges.keyBy(lambda (src_id, (dst_id, weight)): dst_id)\
            .leftOuterJoin(locs_known)\
            .flatMap(lambda (dst_id, (edge, loc_known)): [edge] if loc_known is not None else [] )


        medians =  known_edges.join(locs_known)\
            .map(lambda (src_id, ((dst_id, weight), src_loc)) : (dst_id, (src_loc, weight)))\
            .groupByKey()\
            .filter(lambda (src_id, neighbors) : len(neighbors) > 2)\
            .mapValues(lambda neighbors :\
                       median(haversine, [loc for loc,w in neighbors], [w for loc,w in neighbors]))\
            .join(locs_known)\
            .mapValues(lambda (found_loc, known_loc) :\
                (known_loc, found_loc, haversine(known_loc.geo_coord,  found_loc.geo_coord)))


        #some medians might have std_devs of zero
        close_locs = medians.filter(lambda (src_id, (found_loc, known_loc, dist)) : found_loc.dispersion_std_dev == 0)
        remaining_locs = medians.filter(lambda (src_id, (found_loc, known_loc, dist)) : found_loc.dispersion_std_dev != 0)

        #values_w_stdev = remaining_locs.map(lambda (src_id, (found_loc, known_loc, dist)) :\
        #    (src_id, (dist-found_loc.dispersion/found_loc.dispersion_std_dev)))\
        #    .values()

        values_w_stdev = remaining_locs.map(lambda (src_id, (found_loc, known_loc, dist)) :\
                (src_id, (dist/found_loc.dispersion_std_dev)))\
                .values()

        values_wo_stdev = close_locs.map(lambda (src_id, (found_loc, known_loc, dist)): (src_id, dist))\
                                                .values()

        return EstimatorCurve(EstimatorCurve.build_curve(values_w_stdev, desired_samples),\
            EstimatorCurve.build_curve(values_wo_stdev, desired_samples))

    @staticmethod
    def build_curve(vals, desired_samples):
        '''
        Static helper method for building the curve from a set of stdev stample

        Args:
            vals (rdd of floats): The rdd containing the standard deviation from the distance
                between the estimated location and the actual locationn
            desired_samples (int): For larger RDDs it is more efficient to take a sample for
                the collect

        Returns:
            curve (numpy.ndarray): two dimensional array representing the curve.

            Column 0 is the sorted stdevs and column 1 is the percentage for the CDF.
        '''

        cnt = vals.count()

        sample = vals;

        if(cnt > desired_samples):
            sample = vals.sample(False, desired_samples/float(cnt), 45)
            print("Before sample: ", cnt, " records")
            cnt = sample.count()

        print("Sample count: ", cnt)

        return np.column_stack((np.sort(sample.collect()), np.arange(cnt)/float(cnt)))


    def lookup(self, num_std_devs):
        '''
        lookups up closes stdev by subtracting from lookup table, taking absolute value
        and finding which is closest to zero by sorting and taking the first element

        Args:
            num_std_devs (float): the stdev to lookup

        Returns:
            CDF (float) : Percentage of actual locations found to be within the input stdev
        '''

        max_std = np.max(self.w_stdev[:, 0])
        min_std = np.min(self.w_stdev[:, 0])

        if (num_std_devs < max_std and num_std_devs > min_std) :
            arr = np.absolute(self.w_stdev-np.array([num_std_devs,0]))
            return arr[arr[:,0].argsort()][0][1]
        else if num_std_devs == 0:
            print("reference the stdev == 0 curve")
            return -1
        else:
            return 1


    def plot(self, w_stdev_lim=10, wo_stdev_lim=1000):
        '''
        Plots both the stdev curve and the distance curve for when the stdev
        is 0

        Args:
            w_stdev_lim(int) : x axis limit for the plot
            wo_stdev_lim(int) : x axis limit for the plot
        '''

        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(121)
        ax1.plot(self.w_stdev[:,0], self.w_stdev[:,1], label="stdev>0")
        ax1.set_xlim([-1,w_stdev_lim])
        ax1.set_xlabel("standard deviation")
        ax1.set_ylabel("percentage (CDF)")
        ax1.legend()
        ax2 = fig.add_subplot(122)
        ax2.plot(self.wo_stdev[:,0], self.wo_stdev[:,1], label="stddev==0")
        ax2.set_xlim([0,wo_stdev_lim])
        ax2.set_xlabel("distance")
        ax2.set_ylabel("percentage (CDF)")
        ax2.legend()
        plt.show()

    def save(self, name="estimator"):
        '''
        Saves the EstimatorCurve as a csv

        Args:
            name(string): A prefix name for the filename. Two CSVs will be created-- one for
                when the stdev is 0, and one for when it is greater than 0
        '''

        np.savetxt(open(name+"_curve.csv",'w'), self.w_stdev, delimiter=",")
        np.savetxt(open(name+"_curve_zero_stdev.csv", "w"), self.wo_stdev, delimiter=",")
        print("Saved estimator curve as \'%s.curve\'" % name)
        print("Saved estimator curve with 0 stdev as \'%s.curve_zero_stdev\'" % name)

    @staticmethod
    def load_from_file(name="estimator"):
        '''
        Loads an Estimator curve from csv files

        Args:
            name(string): prefix name for the two CSV files
        '''

        return EstimatorCurve(np.loadtxt(open(name+"_curve.csv"), delimiter=","),\
                np.loadtxt(open(name+"_curve_zero_stdev.csv"), delimiter=","))

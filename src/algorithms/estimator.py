import csv
import matplotlib.pyplot as plt
import numpy as np
from src.algorithms.slp2 import haversine, median

class EstimatorCurve:

    def __init__(self, w_stdev, wo_stdev):
        self.w_stdev = w_stdev
        self.wo_stdev = wo_stdev

    def predict_probability_area(self, upper_bound, lower_bound, estimated_loc):
        '''
        upper_bound: geoCoord
        lower_bound: geoCoord
        center: geoCoord
        std_dev: standard deviation of dispersions from predicted location
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
        lookups up closes stdev by subtracting from lookup table, taking absolution value
        and finding which is closest to zero by sorting and taking the first element
        '''

        max_std = np.max(self.w_stdev[0])
        min_std = np.min(self.w_stdev[0])

        if (num_std_devs < max_std and num_std_devs > min_std) :
            arr = np.absolute(self.w_stdev-np.array([num_std_devs,0]))
            return arr[arr[:,0].argsort()][0][1]
        else:
            return 1


    def plot(self, w_stdev_lim=10, wo_stdev_lim=1000):
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
        np.savetxt(open(name+"_curve.csv",'w'), self.w_stdev, delimiter=",")
        np.savetxt(open(name+"_curve_zero_stdev.csv", "w"), self.wo_stdev, delimiter=",")
        print("Saved estimator curve as \'%s.curve\'" % name)
        print("Saved estimator curve with 0 stdev as \'%s.curve_zero_stdev\'" % name)

    @staticmethod
    def load_from_file(name="estimator"):
        return EstimatorCurve(np.loadtxt(open(name+"_curve.csv"), delimiter=","),\
                np.loadtxt(open(name+"_curve_zero_stdev.csv"), delimiter=","))

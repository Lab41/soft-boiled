import numpy as np
import itertools
from collections import namedtuple, defaultdict
import math
from math import floor, ceil, radians, sin, cos, asin, sqrt, pi
from src.utils.geo import bb_center, GeoCoord, haversine

LocEstimate = namedtuple('LocEstimate', ['geo_coord', 'dispersion', 'dispersion_std_dev'])

def median(distance_func, vertices, weights=None):
    """
    given a python list of vertices, and a distance function, this will find the vertex that is most central
    relative to all other vertices. All of the vertices must have geocoords

    Args:
        distance_func (function) : A function to calculate the distance between two GeoCoord objects
        vertices (list) : List of GeoCoord objects

    Returns:
        LocEstimate: The median point
    """

    #get the distance between any two points
    distances = map(lambda (v0, v1) :distance_func(v0.geo_coord, v1.geo_coord), itertools.combinations (vertices, 2))

    #create a dictionary with keys representing the index of a location
    m = { a: list() for a in range(len(vertices)) }

    #add the distances from each point to the dict
    for (k0,k1),distance in zip(itertools.combinations(range(len(vertices)), 2), distances):
        #a distance can be None if one of the vertices does not have a geocoord
        if(weights is None):
            m[k0].append(distance)
            m[k1].append(distance)
        else:
            # Weight distances by weight of destination vertex
            # Ex: distance=3, weight =4 extends m[k0] with [3, 3, 3, 3]
            m[k0].extend([distance]*weights[k1])
            m[k1].extend([distance]*weights[k0])


    summed_values = map(sum, m.itervalues())

    idx = summed_values.index(min(summed_values))

    if weights is not None and weights[idx] > 1:
        # Handle self-weight (i.e. if my vertex has weight of 6 there are 5 additional self connections if
        # Starting from my location)
        m[idx].extend([0.0]*(weights[idx]-1))

    return LocEstimate(geo_coord=vertices[idx].geo_coord, dispersion=np.median(m[idx]), dispersion_std_dev=np.std(m[idx]))


def get_known_locs(sqlCtx, table_name, include_places=True,  min_locs=3, num_partitions=30, dispersion_threshold=50):
    '''
    Given a loaded twitter table, this will return all the twitter users with locations. A user's location is determined
    by the median location of all known tweets. A user must have at least min_locs locations in order for a location to be
    estimated


    Args:
        sqlCtx (Spark SQL Context) :  A Spark SQL context
        table_name (string): Table name that was registered when loading the data
        min_locs (int) : Minimum number tweets that have a location in order to infer a location for the user
        num_partitions (int) : Optimizer for specifying the number of partitions for the resulting
            RDD to use.
        dispersion_threhold (int) : A distance threhold on the dispersion of the estimated location for a user.
            We consider those estimated points with dispersion greater than the treshold unable to be
            predicted given how dispersed the tweet distances are from one another.

    Returns:
        locations (rdd of LocEstimate) : Found locations of users. This rdd is often used as the ground truth of locations
    '''

    geo_coords = sqlCtx.sql('select user.id_str, geo.coordinates from %s where geo.coordinates is not null' % table_name)\
        .map(lambda row: (row.id_str, row.coordinates))

    if(include_places):
        place_coords = sqlCtx.sql("select user.id_str, place.bounding_box.coordinates from %s "%table_name +
            "where geo.coordinates is null and size(place.bounding_box.coordinates) > 0 and place.place_type " +
            "in ('city', 'neighborhood', 'poi')").map(lambda row: (row.id_str, bb_center(row.coordinates)))
        geo_coords = geo_coords.union(place_coords)

    return geo_coords.groupByKey()\
        .filter(lambda (id_str,coord_list): len(coord_list) >= min_locs)\
            .map(lambda (id_str,coords): (id_str, median(haversine, [LocEstimate(GeoCoord(lat,lon), None, None)\
                for lat,lon in coords])))\
            .filter(lambda (id_str, loc): loc.dispersion < dispersion_threshold)\
            .coalesce(num_partitions).cache()


def get_edge_list(sqlCtx, table_name, num_partitions=300):
    '''

    Given a loaded twitter table, this will return the @mention network in the form (src_id, (dest_id, num_@mentions))

    Args:
        sqlCtx (Spark SQL Context) : A Spark SQL context
        table_name (string) : Table name that was registered when loading the data
        num_paritions (int) : Optimizer for specifying the number of paritions for the resulting RDD to use

    Returns:
        edges (rdd (src_id, (dest_id, weight))) : edges loaded from the table

    '''
    tmp_edges = sqlCtx.sql('select user.id_str, entities.user_mentions from %s where size(entities.user_mentions) > 0' % table_name)\
        .flatMap(lambda row : [((row.id_str, mentioned_user.id_str),1)\
                               for mentioned_user in row.user_mentions\
                               if mentioned_user.id_str is not None and row.id_str !=  mentioned_user.id_str])\
            .reduceByKey(lambda x,y:x+y)

    return tmp_edges.map(lambda ((src_id,dest_id),num_mentions): ((dest_id,src_id),num_mentions))\
        .join(tmp_edges)\
            .map(lambda ((src_id,dest_id), (count0, count1)): (src_id, (dest_id, min(count0,count1))))\
            .coalesce(num_partitions).cache()


def train_slp(locs_known, edge_list, num_iters, neighbor_threshold=3, dispersion_threshold=100):
    '''
    Core SLP algorithm

    Args:
        locs_known (rdd of LocEstimate objects) : Locations that are known for the SLP network
        edge_list (rdd of edges (src_id, (dest_id, weight))) : edges representing the at mention
            network
        num_iters (int) : number of iterations to run the algorithm
        neighbor_threshold (int) : The minimum number of neighbors required in order for SLP to
            try and predict a location of a node in the network
        dispersion_theshold (int) : The maximum median distance amoung a local at mention network
            in order to predict a node's location.

    Returns:
        locations (rdd of LocEstimate objects) : The locations found and known
    '''

    num_partitions = edge_list.getNumPartitions()

    # Filter edge list so we never attempt to estimate a "known" location
    filtered_edge_list = edge_list.keyBy(lambda (src_id, (dst_id, weight)): dst_id)\
                            .leftOuterJoin(locs_known)\
                            .flatMap(lambda (dst_id, (edge, loc_known)): [edge] if loc_known is None else [] )

    l = locs_known

    for i in range(num_iters):
        l = filtered_edge_list.join(l)\
            .map(lambda (src_id, ((dst_id, weight), known_vertex)) : (dst_id, (known_vertex, weight)))\
            .groupByKey()\
            .filter(lambda (src_id, neighbors) : neighbors.maxindex >= neighbor_threshold)\
            .map(lambda (src_id, neighbors) :\
                 (src_id, median(haversine, [v for v,w in neighbors],[w for v,w in neighbors])))\
            .filter(lambda (src_id, estimated_loc): estimated_loc.dispersion < dispersion_threshold)\
            .union(locs_known).coalesce(num_partitions)

    return l

def evaluate(locs_known, edges, holdout_func, slp_closure):
    '''
    This function is used to assess various stats regarding how well SLP is running.
    Given all locs that are known and all edges that are known, this funciton will first
    apply the holdout to the locs_known, allowing for a ground truth comparison to be used.
    Then, it applies the non-holdout set to the training function, which should yield the
    locations of the holdout for comparison.

        For example::

            holdout = lambda (src_id) : src_id[-1] == '6'
            trainer = lambda l, e : slp.train_slp(l, e, 3)
            results = evaluate(locs_known, edges, holdout, trainer)

    Args:
        locs_known (rdd of LocEstimate objects) : The complete list of locations

        edges (rdd of (src_id, (dest_id, weight)): all available edge information

        holdout_func (function) : function responsible for filtering a holdout data set. For example::

                lambda (src_id) : src_id[-1] == '6'

            can be used to get approximately 10% of the data since the src_id's are evenly distributed numeric values

        slp_closure (function closure): a closure over the slp train function. For example::

                lambda locs, edges :\n
                        slp.train_slp(locs, edges, 4, neighbor_threshold=4, dispersion_threshold=150)

            can be used for training with specific threshold parameters


    Returns:
        results (dict) : stats of the results from the SLP algorithm

            `median:` median difference of predicted versus actual

            `mean:` mean difference of predicted versus actual

            `coverage:` ratio of number of predicted locations to number of  original unknown locations

            `reserved_locs:` number of known locations used to train

            `total_locs:` number of known locations input into this function

            `found_locs:` number of predicted locations

            `holdout_ratio:` ratio of the holdout set to the entire set
        '''

    reserved_locs = locs_known.filter(lambda (src_id, loc): not holdout_func(src_id))
    num_locs = reserved_locs.count()
    total_locs = locs_known.count()

    print('Total Locations %s' % total_locs)

    results = slp_closure(reserved_locs, edges)

    errors = results\
        .filter(lambda (src_id, loc): holdout_func(src_id))\
        .join(locs_known)\
        .map(lambda (src_id, (vtx_found, vtx_actual)) :\
             (src_id, (haversine(vtx_found.geo_coord, vtx_actual.geo_coord), vtx_found)))

    errors_local = errors.map(lambda (src_id, (dist, est_loc)) : dist).collect()

    #because cannot easily calculate median in RDDs we will bring deltas local for stats calculations.
    #With larger datasets, we may need to do this in the cluster, but for now will leave.
    return (errors, {
        'median': np.median(errors_local),
        'mean': np.mean(errors_local),
        'coverage':len(errors_local)/float(total_locs - num_locs),
        'reserved_locs': num_locs,
        'total_locs':total_locs,
        'found_locs': len(errors_local),
        'holdout_ratio' : 1 - num_locs/float(total_locs)
    })

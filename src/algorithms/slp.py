import numpy as np
import itertools
from collections import namedtuple, defaultdict
from math import floor, ceil, radians, sin, cos, asin, sqrt, pi
import pandas as pd
from src.utils.geo import bb_center, GeoCoord, haversine

LocEstimate = namedtuple('LocEstimate', ['geo_coord', 'dispersion', 'dispersion_std_dev'])

def median(distance_func, vertices, weights=None):
    """
        given a python list of vertices, and a distance function, this will find the vertex that is most central
        relative to all other vertices. All of the vertices must have geocoords
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


def get_known_locs(sqlCtx, table_name, min_locs=3, num_paritions=30, dispersion_threshold=50):
    '''
        Given a loaded twitter table, this will return all the twitter users with locations. A user's location is determined
        by the median location of all known tweets. A user must have at least min_locs locations in order for a location to be
        estimated
        '''
    # Geo Coords


    geo_coords = sqlCtx.sql('select user.id_str, geo.coordinates from %s where geo.coordinates is not null' % table_name)\
        .map(lambda row: (row.id_str, row.coordinates))

    place_coords = sqlCtx.sql("select user.id_str, place.bounding_box.coordinates from %s "%table_name +
        "where geo.coordinates is null and size(place.bounding_box.coordinates) > 0 and place.place_type " +
         "in ('city', 'neighborhood', 'poi')").map(lambda row: (row.id_str, bb_center(row.coordinates)))


    return geo_coords.union(place_coords).groupByKey()\
        .filter(lambda (id_str,coord_list): len(coord_list) >= min_locs)\
            .map(lambda (id_str,coords): (id_str, median(haversine, [LocEstimate(GeoCoord(lat,lon), None, None)\
                                                                     for lat,lon in coords])))\
                                                                     .filter(lambda (id_str, loc): loc.dispersion < dispersion_threshold)\
                                                                     .coalesce(num_paritions).cache()


def get_edge_list(sqlCtx, table_name, num_paritions=300):
    '''
        Given a loaded twitter table, this will return the @mention network in the form (src_id, (dest_id, num_@mentions))
        '''
    tmp_edges = sqlCtx.sql('select user.id_str, entities.user_mentions from %s where size(entities.user_mentions) > 0' % table_name)\
        .flatMap(lambda row : [((row.id_str, mentioned_user.id_str),1)\
                               for mentioned_user in row.user_mentions\
                               if mentioned_user.id_str is not None and row.id_str !=  mentioned_user.id_str])\
            .reduceByKey(lambda x,y:x+y)

    return tmp_edges.map(lambda ((src_id,dest_id),num_mentions): ((dest_id,src_id),num_mentions))\
        .join(tmp_edges)\
            .map(lambda ((src_id,dest_id), (count0, count1)): (src_id, (dest_id, min(count0,count1))))\
            .coalesce(num_paritions).cache()

def run(sqlCtx, table_name, holdout_function=None):

    all_locs_known = get_known_locs(sqlCtx, table_name)
    if holdout_function:
        filtered_locs_known = all_locs_known.filter(lambda (id_str, loc_estimate): holdout_function(id_str))
    else:
        filtered_locs_known = all_locs_known
    edge_list = get_edge_list(sqlCtx, table_name)
    result = train(locs_known, edge_list)
    return result


def train_slp(locs_known, edge_list, num_iters, neighbor_threshold=3, dispersion_threshold=100):
    '''
        Inputs:
        locs_known => (src_id, vertex)
        edge_list  => (src_id, (dest_id, weight))
        line 0:  attach the locations to each of the sources in the edges... (src_id, ((dst_id, weight), src_vertex))
        line 1:  get the src and coord in value in prep for group by...      (dst_id, (Vertex, weight))
        line 2:  filter out those edges where a Vertex has no geoCoord...    (dst_id, (Vertex, weight)) #has geocoord
        line 3:  group by in prep for neighbor operations...                 (dst_id, [(Vertex, weight)..])
        line 4:  filter out nodes with fewer than 2 neighbors...             (dst_id, [(Vertex, weight)..]) # >2
        line 5:  add back in known locs so we only predict unknown...        (dst_id, ([(Vertex, weight)..], hasLoc))
        line 6:  only keep the nodes we are trying to predict...             (dst_id, ([(Vertex, weight)..], hasLoc))
        line 7:  apply the median to the neighbors...                        (dst_id, (median_vtx, neighbors))
        line 8:  given the median, filter out high dispersion....            (dst_id, (median_vtx, neighbors)) <disp
        line 9:  prepare for the union by adjusting format...                (dst_id, median_coord)
        line 8:  union to create the global location rdd...                  (dst_id, median_geoCoord)
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


#holdout_10pct = lambda (src_id) : src_id[-1] != '6'


def run_slp_test(original_locs_known, estimated_locs,  holdout_func):
    '''
        '''

    reserved_locs = original_locs_known.filter(lambda (src_id, loc): not holdout_func(src_id))
    num_locs = reserved_locs.count()


    errors = estimated_locs\
        .filter(lambda (src_id, loc): not holdout_func(src_id))\
        .join(reserved_locs)\
        .map(lambda (src_id, (vtx_found, vtx_actual)) :\
             haversine(vtx_found.geo_coord, vtx_actual.geo_coord))

    errors_local = errors.collect()

    #because cannot easily calculate median in RDDs we will bring deltas local for stats calculations.
    #With larger datasets, we may need to do this in the cluster, but for now will leave.
    return {
        'median': np.median(errors_local),
            'mean': np.mean(errors_local),
            'coverage':len(errors_local)/float(num_locs),
            'num_locs': num_locs
    }


def predict_country(tweets, bounding_boxes):
    '''
    Take a set of estimates of user locations and estimate the country that user is in

    tweets: RDD of (id_str, LocEstimate)
    bounding_boxes: list [(country_code, (min_lat, max_lat, min_lon, max_lon)),...]
    '''
    # Convert Bounding boxes to allow for more efficient lookups
    bb_lookup_lat = defaultdict(set)
    bb_lookup_lon = defaultdict(set)
    for i, (cc, (min_lat, max_lat, min_lon, max_lon)) in enumerate(bounding_boxes):
        for lon in range(int(math.floor(min_lon)), int(math.ceil(max_lon))):
            bb_lookup_lon[lon].add(i)
        for lat in range(int(math.floor(min_lat)), int(math.ceil(max_lat))):
            bb_lookup_lat[lat].add(i)

    # broadcast to make more efficient
    bb_lookup_lat_bcast = sc.broadcast(bb_lookup_lat)
    bb_lookup_lon_bcast = sc.broadcast(bb_lookup_lon)
    bounding_boxes_bcast = sc.broadcast(bounding_boxes)

    # Do country lookups and return an RDD that is (id_str, [country_codes])
    return tweets.mapValues(lambda loc_estimate: _predict_country_using_lookup(loc_estimate,\
                                                                              bb_lookup_lat_bcast,
                                                                              bb_lookup_lon_bcast,
                                                                              bounding_boxes_bcast))

def _predict_country_using_lookup(loc_estimate, lat_dict, lon_dict, bounding_boxes):
    '''
    Internal helper function that uses broadcast lookup tables to take a single location estimate and show
        what country bounding boxes include that point

    loc_estimate: LocEstimate
    lat_dict: broadcast dictionary {integer_lat:set([bounding_box_indexes containing this lat])}
    lon_dict: broadcast dictionary {integer_lon:set([bounding_box_indexes containing this lon])}
    bounding_boxes: broadcast list [(country_code, (min_lat, max_lat, min_lon, max_lon)),...]
    '''
    lat = loc_estimate.geo_coord.lat
    lon = loc_estimate.geo_coord.lon
    countries = set()
    potential_lats = lat_dict.value[math.floor(lat)]
    potential_lons = lon_dict.value[math.floor(lon)]
    intersection = potential_lats.intersection(potential_lons)
    if len(intersection) == 0:
        return []
        #raise ValueError('uh oh')
    else:
        for index in intersection:
            cc, (min_lat, max_lat, min_lon, max_lon) = bounding_boxes.value[index]
            if min_lon < lon and lon < max_lon and min_lat < lat and lat < max_lat:
                countries.add(cc)
    return list(countries)
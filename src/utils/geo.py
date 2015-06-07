from math import floor, radians, sin, cos, asin, sqrt, pi
import numpy as np
#from haversine import haversine

EARTH_RADIUS = 6367
def haversine(lon1, lat1, lon2, lat2):
    """
    From: http://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(min(1, sqrt(a))) # Added min to protect against roundoff errors for nearly antipodal locations
    return c * EARTH_RADIUS

def median_point(points, num_points_req=3, return_dispersion=True, dispersion_treshold=None, use_usr_ids=False):
    """ Return the median point and the dispersion"""
    local_cache = {}
    if use_usr_ids:
        # If we've passed in [(ll, usr_id), (ll, usr_id)] check that we have the right number of user ids instead
        # of just points
        actual_points = []
        user_ids = set()
        for ll, src in points:
            user_ids.add(src)
            actual_points.append(ll)
        if len(user_ids) < num_points_req:
            return []
        points = actual_points
    else:
        # Just make sure we have enough points
        if len(points) < num_points_req:
            return []
    points = list(points)
    min_distance = None
    current_errors = None
    min_index = None
    for i in range(len(points)):
        errors = []
        for j in range(len(points)):
            if i != j: # Don't compute self distance
                p1 = points[i]
                p2 = points[j]

                # Cache canonical ordering
                if p1 < p2:
                    point_tuple = (points[i][1], points[i][0], points[j][1], points[j][0])
                else:
                    point_tuple = (points[j][1], points[j][0], points[i][1], points[i][0])

                # Pull from cache or calculate
                if point_tuple in local_cache:
                    error = local_cache[point_tuple]
                elif p1 == p2:
                    error = 0
                else:
                    error = haversine(points[j][1], points[j][0], points[i][1], points[i][0])
                    local_cache[point_tuple] = error
                errors.append(error)
        distance_sum = sum(errors)
        if min_distance is None or distance_sum < min_distance:
            min_distance = distance_sum
            current_errors = errors
            min_index = i
    dispersion = np.median(current_errors)
    if dispersion_treshold:
        if dispersion > dispersion_treshold:
            return []

    if return_dispersion:
        return [(points[min_index], dispersion)]
    else:
        return [points[min_index]]

def median_point2(points):
    """ Return the median point and the dispersion"""
    points_list = []
    for (point, isTrue) in points:
        if isTrue:
            return (point, isTrue)
        else:
            points_list.append(point)
    points = points_list
    if len(points) < 3:
        return None
    points = list(points)
    min_distance = None
    current_errors = None
    min_index = None
    for i in range(len(points)):
        distance_sum = 0
        errors = []
        for j in range(len(points)):
            error = haversine(points[j][1], points[j][0], points[i][1], points[i][0])
            errors.append(error)
        distance_sum = sum(errors)
        if min_distance is None or distance_sum < min_distance:
            min_distance = distance_sum
            current_errors = errors
            min_index = i
    return (points[min_index], False)
   # return (points[min_index], np.median(current_errors))

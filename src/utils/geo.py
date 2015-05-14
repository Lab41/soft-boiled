from math import floor, radians, sin, cos, asin, sqrt

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
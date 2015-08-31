from collections import namedtuple, defaultdict
import math


GeoCoord = namedtuple('GeoCoord', ['lat', 'lon'])
# TODO: Move this geo geo
EARTH_RADIUS = 6367
def haversine(x, y):
    """
        From: http://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """

    #if either distance is None return None
    if x is None or y is None:
        raise Exception("Null coordinate")

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [x.lon, x.lat, y.lon, y.lat])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(min(1, math.sqrt(a))) # Added min to protect against roundoff errors for nearly antipodal locations
    return c * EARTH_RADIUS


def bb_center(row):
    """ Takes a twitter place bounding box (in lon, lat) and returns the centroid of the quadrilateral (in lat, lon)."""
    bb_coords = row[0]

    avg_lon = (bb_coords[0][0] + bb_coords[1][0] + bb_coords[2][0] + bb_coords[3][0])/4
    avg_lat = (bb_coords[0][1] + bb_coords[1][1] + bb_coords[2][1] + bb_coords[3][1])/4

    return [avg_lat,avg_lon]




def shapefile_2_df(filepath, region_name, ur_boundary, ll_boundary, proj='cyl', ellipse='WGS84'):
    '''
    Given a shape file, returns a Pandas dataframe representing the region
    http://biogeo.ucdavis.edu/data/world/countries_shp.zip downloaded in Aug 2015 is the shapefile used in testing

    Args:
        filepath (string) : filepath is filepath to shapefile minus the extension (i.e., no .shp)
        region_name (string) : depth of the shape file (ISO2, state, etc)
        ur_boundary (GeoCoord) : Upper Right coord of the bounding box
        ll_boundary (GeoCoord) : Lower Left coord of the bounding box
        proj (string) : proj is the projection of the shapefile from matplotlib basemap projection options
        ellipse (string) : ellipse is the ellipsoid from matplotlib basemap ellipse options

    Returns:
        a pandas dataframe of polygon vertices, polygon ID, area m, area km.
    '''

    import pandas as pd
    from mpl_toolkits.basemap import Basemap
    from shapely.geometry import Polygon


    m = Basemap(
        projection=proj,
        lon_0=0.,
        lat_0=0.,
        ellps = ellipse,
        llcrnrlon=ll_boundary.lon,
        llcrnrlat=ll_boundary.lat,
        urcrnrlon=ur_boundary.lon,
        urcrnrlat=ur_boundary.lat,
        lat_ts=0,
        resolution='i',
        suppress_ticks=True)

    world=m.readshapefile(filepath,'world')

    df_map = pd.DataFrame({
        'poly': [Polygon(xy) for xy in m.world],
        'region_name': [row[region_name] for row in m.world_info]})

    df_map['area_m'] = df_map['poly'].map(lambda x: x.area)
    df_map['area_km'] = df_map['area_m'] / 1000000

    return df_map, m

def find_best_region(geo_coord_rdd, shapefile_path, region_name, sc):
    '''
    Take a set of estimates of user locations and estimate the country that user is in
    Args:
        tweets (RDD  (id_str, LocEstimate))
        bounding_boxes (list [(country_code, (min_lat, max_lat, min_lon, max_lon)),...])
        polygon_df (pandas dataframe (polygon verticies, polygon ID string))
            ideally is returned from shapefile_2_df()
    Returns:
        Country Codes (list) : Predicted countries reperesented as their numeric codes
    '''

    df_map, m = shapefile_2_df(shapefile_path, region_name, GeoCoord(89.9, 180.0),\
                               GeoCoord(-89.9, -180.0), proj='cyl', ellipse='WGS84')

    bounding_boxes=zip(df_map['region_name'],df_map['poly'].map(lambda (row) :\
        (GeoCoord(row.bounds[1], row.bounds[0]), GeoCoord(row.bounds[3], row.bounds[2]))))

    # Convert Bounding boxes to allow for more efficient lookups
    bb_lookup_lat = defaultdict(set)
    bb_lookup_lon = defaultdict(set)

    # build the lookup table which keeps track of which countries exist between
    # unit longitudes and latitudes
    for i, (cc, (ll_geo, ur_geo)) in enumerate(bounding_boxes):
        for lon in range(int(math.floor(ll_geo.lon)), int(math.ceil(ur_geo.lon))):
            bb_lookup_lon[lon].add(i)
        for lat in range(int(math.floor(ll_geo.lat)), int(math.ceil(ur_geo.lat))):
            bb_lookup_lat[lat].add(i)

    bb_lkup_lat_bcast=sc.broadcast(bb_lookup_lat)
    bb_lkup_lon_bcast=sc.broadcast(bb_lookup_lon)
    bbox_bcast=sc.broadcast(bounding_boxes)


    # Do country lookups and return an RDD that is (id_str, [country_codes])
    polycode = geo_coord_rdd.mapValues(lambda geo_coord: _predict_country_using_lookup(geo_coord,\
        bb_lkup_lat_bcast,\
        bb_lkup_lon_bcast,\
        bbox_bcast))

    df_bcast = sc.broadcast(df_map)


    return polycode.mapValues(lambda (id_list, geo_coord):\
        country_finder(geo_coord, id_list, df_bcast.value, m))



def country_finder(geo_coord, bbox_hits, polygon_df, m):
    from shapely.geometry import Point
    from shapely.prepared import prep

    inpoint = Point(m(geo_coord.lon,geo_coord.lat))
    region=polygon_df[polygon_df['region_name'].isin(bbox_hits)]

    pgon_list = [idx  for idx, pgon in enumerate([prep(pgon) for pgon in region['poly']])\
                 if pgon.contains(inpoint)]

    return region.iloc[pgon_list]



def _predict_country_using_lookup(geo_coord, lat_dict, lon_dict, bounding_boxes):
    '''
    Internal helper function that uses broadcast lookup tables to take a single location estimate and show
        what country bounding boxes include that point
    Args:
        loc_estimate (LocEstimate) : Estimate location
        lat_dict (broadcast dictionary {integer_lat:set([bounding_box_indexes containing this lat])}) :
            Indexed lookup dictionary for finding countries that exist at the specified latitude
        lon_dict ((broadcast dictionary) {integer_lon:set([bounding_box_indexes containing this lon])})) :
            Index lookup dictionary for finding countries that exist at the speficied longitude
        bounding_boxes (broadcast list [(country_code, (min_lat, max_lat, min_lon, max_lon)),...]) :
            List of countries and their boudning boxes
    '''

    countries = set()
    potential_lats = lat_dict.value[math.floor(geo_coord.lat)]
    potential_lons = lon_dict.value[math.floor(geo_coord.lon)]
    intersection = potential_lats.intersection(potential_lons)
    if len(intersection) == 0:
        return []
        #raise ValueError('uh oh')
    else:
        for index in intersection:
            cc, (ll_geo, ur_geo) = bounding_boxes.value[index]
            if ll_geo.lon < geo_coord.lon and geo_coord.lon < ur_geo.lon\
                and ll_geo.lat < geo_coord.lat and geo_coord.lat < ur_geo.lat:
                    countries.add(cc)
    return list(countries), geo_coord
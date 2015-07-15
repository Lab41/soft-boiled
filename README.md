# Geo-Inferencing in Twitter [Soft-Boiled] 

## General
The usage examples below assume that you have created a zip file containing the top level directory of the repo called soft-boiled.zip

## Spatial Label Propagation [slp.py]:

### Usage:
```python
sc.addPyFile ('/path/to/zip/soft-boiled.zip') # Can be an hdfs path
from src.algorithms.slp import SLP

# Create dataframe from parquet data
tweets = sqlCtx.parquetFile('hdfs:///post_etl_datasets/twitter')

# Configure options object
options = {'dispersion_threshold':100, 'num_located_neighbors_req':3, 
                   'num_iters':5, 'hold_out':set(['9'])}
# Create algorithm with options
slp = SLP(sc, sqlCtx, options)
# Train 
slp.train(tweets)
# Test
slp.test(tweets)
# Save a dictionary of known and estimated user id_str and their corresponding location
slp.save('/my/local/path/filename.pkl') 
```

### Options:
##### Related to calculating the median point amongst a collection of points:
***dispersion_threshold***: This is the maximum median distance in km a point can be from the remaining points and still estimate a location

***num_points_req_for_known***:  Number of geotagged posts that a user must have to be included in ground truth. 

***home_radius_for_known***: Median distance in km that num_points_req_for_known must be under from each other.

***num_located_neighbors_req***: The min number of vertex neighbors with known locations needed before we will calculate a median location to assign an unlocatd user.

##### Related to the actual label propagation:
***num_iters***: This controls the number of iterations of label propagation performed


##### Related to reading in data [Applicable if slp.load is used]:
***use_parquet***: If true load the data as parquet files, if no use_[type] then code assumes JSON

***hold_out***:  A set of final digits of the id_str that will not be included in training and only used in testing [eg: set([‘9’]) ]

***json_path***: This is a path to a file describing the schema of the data. For the JSON input format this avoids an extra pass through the data to estimate the format and ensures a consistent format independent of the amount of data included


## Gaussian Mixture Model [gmm.py]
### Usage:
```python
sc.addPyFile ('/path/to/zip/soft-boiled.zip') # Can be an hdfs path
from src.algorithms.gmm import GMM

# Create dataframe from parquet data
tweets = sqlCtx.parquetFile('hdfs:///post_etl_datasets/twitter')

# Configure options object
options = {'fields':set(['user.location', 'text'])}
# Create algorithm with options
gmm = GMM(sc, sqlCtx, options)
# Train
gmm.train(tweets)
#Test
gmm.test(tweets)
# Save a dictionary of words and the GMM that corresponds to that word
save('/my/local/path/filename.pkl')
```
### Options
##### Related to GMM :
***fields***: A set of fields to use to train/test the GMM model. Currently only user.location and text are supported

##### Related to reading in data [Applicable if gmm.load is used]:
***use_parquet***: If true load the data as parquet files, if no use_[type] then code assumes JSON

***json_path***: This is a path to a file describing the schema of the data. For the JSON input format this avoids an extra pass through the data to estimate the format and ensures a consistent format independent of the amount of data included

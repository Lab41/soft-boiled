# Geo-Inferencing in Twitter [Soft-Boiled] 

## General
The usage examples below assume that you have created a zip file containing the top level directory of the repo called soft-boiled.zip

## Spatial Label Propagation [slp.py]:

### Usage:
```python
sc.addPyFile ('/path/to/zip/soft-boiled.zip') # Can be an hdfs path
from src.algorithms.slp import *

# Create dataframe from parquet data
tweets = sqlCtx.read.parquet('hdfs:///post_etl_datasets/twitter')
tweets.registerTempTable('my_tweets')

# Get Known Locations
locs_known = get_known_locs(sqlCtx, 'my_tweets', min_locs=3, dispersion_threshold=50, num_partitions=30)

# Filter locs_known to get training split
holdout_10pct = lambda (src_id): src_id[-1] != '9'
filtered_locs_known = locs_known.filter(lambda (id_str, loc_estimate): holdout_10pct(id_str))

# Get at mention network, bi-directional at mentions
edge_list = get_edge_list(sqlCtx, 'my_tweets')

# Run spaital label propagation with 5 iterations
estimated_locs = train_slp(filtered_locs_known, edge_list, 5, dispersion_threshold=100)

# Test results
test_results = run_slp_test(locs_known, estimated_locs, holdout_10pct)
print test_results
```

### Options:
##### Related to calculating the median point amongst a collection of points:
***dispersion_threshold***: This is the maximum median distance in km a point can be from the remaining points and still estimate a location

***min_locs***:  Number of geotagged posts that a user must have to be included in ground truth. 


##### Related to the actual label propagation:
***num_iters***: This controls the number of iterations of label propagation performed


## Gaussian Mixture Model [gmm.py]
### Usage:
```python
sc.addPyFile ('/path/to/zip/soft-boiled.zip') # Can be an hdfs path
from src.algorithms.gmm import *

# Create dataframe from parquet data
tweets = sqlCtx.read.parquet('hdfs:///post_etl_datasets/twitter')
tweets.registerTempTable('my_tweets')

# Train GMM model
gmm_model = train_gmm(sqlCtx, 'my_tweets', ['user.location', 'text'], min_occurrences=10, max_num_components=12)

# Test GMM model
test_results = run_gmm_test(sc, sqlCtx, 'my_tweets', ['user.location', 'text'], gmm_model)
print test_results

# Use GMM model to predict tweets
other_tweets = sqlCtx.read.parquet('hdfs:///post_etl_datasets/twitter')
estimated_locs = predict_user_gmm(sc, other_tweets, ['user.location'], gmm_model, radius=100, predict_lower_bound=0.2)

# Save model for future prediction use
save_model(gmm_model, '/local/path/to/model_file.csv.gz')

# Load a model, produces the same output as train
gmm_model = load_model('/local/path/to/model_file.csv.gz')
```
### Options
##### Related to GMM :
***fields***: A set of fields to use to train/test the GMM model. Currently only user.location and text are supported

***min_occurrences***: Number of times that a token must appear with a known location in the text to be estimated

***max_num_components***: Limit on the number of GMM components that can be used

#### Predict User Options

***radius***: Predict the probability that the user is within this distance of most likely point, used with predict_lower_bound

***predict_lower_bound***: Used with radius to filter user location estimates with probability lower than threshold

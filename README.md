# AI-Project
Content-based recommender system for the AI Project course

## Data
The test data that will be used for this recommender is a snapshot of the steam dataset from 2019, located [here](http://deepx.ucsd.edu/public/jmcauley/steam/). 
* `steam_games`: A collection of all games present on the steam platform
* `australian_user_items`: Steam users along with the games they own.
* `australian_user_reviews`: Reviews made by steam users.

This dataset provides us with items (games), users and user-item interactions in the form of their game library and user reviews of games. These reviews are either positive or negative. 

## Code organization
* `data` folder: where the necessary steam data files should be placed
    - `raw` folder: contains the data directly obtained from the online source
    - `v2` folder: contains the preprocessed data
    - `v3` folder: contains the preprocessed data with top 1% most popular items removed
* `deadline1` folder: contains the data which was used to present for the first deadline
* `evaluation` folder: contains all the quantitative and qualitative results for v2 and v3 data, since qualitative data was too large to upload, you are expected to generate it yourself from [here](./experiments_history.ipynb)
* `plots` folder: contains all the visuals for the presentation and more
* `src` folder:
    - `data` folder: contains all the code to preprocess the input files to be used.
    - `recommender.py`: contains classes which implements the content-based and popularity recommender
* `experiments.ipynb`: all the experiments ran to improve our recommender, hyper parameter optimisations
* `experiments_history.ipynb`: all the experiments of the optimised models on all the data and splits, stores the data to files to be visualised
* `qualitative_evaluation.ipynb`: additional qualitative evaluation of our recommender
* `visualisation.ipynb`: code to visualise all of the results

## Installation & running the code
First, make sure to install all necessary requirements by running `pip install -r requirements.txt`. All requirements are *hard* requirements, the code will not work without them.
To run the code, the processed dataset should be generated with [clean_steam_dataset.ipynb](./src/data/clean_steam_dataset.ipynb). **NOTE:** this might not be the latest version, the latest version should be used from https://github.com/m4urin/autorec_project. Alternatively you can download the cleaned data from the same source.

Once the cleaned data is present, two scripts need to be ran in order to prepare the data. First we need to unpack the data into its individual splits and generate the user_ids.
```
$ python convert_pkl_parquet.py interactions_splits.pkl
```
This will create all the required parquet files and user_ids, all this data should be moved to the data folder if it is not already generated there. After the data is unpacked, we preprocess the reviews with following script. For this the reviews json should be present in the data folder.
```
$ python preprocess_reviews.py
```
The generated reviews parquet file should be present in the data folder with all other files. 

## Base recommender
Information about the base recommender can be viewed in the [readme](./deadline1/README.md) from deadline 1.

## Improved recommender
The improvements are discussed in more detail in the [visualisation notebook](./visualisation.ipynb).

Additionally, qualitative results comparing the recommender with and without a weighting scheme are provided in the [qualitative evaluation notebook](./qualitative_evaluation.ipynb). Please note that `scrape_users.py` should be run in order to provide the required user info for the recommendations that are generated there.
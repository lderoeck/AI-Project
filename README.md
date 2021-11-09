# AI-Project
Recommender system for the AI Project course based on Content-based Recommender Systems 

## Data
Recommendations are done on a snapshot of the steam dataset from 2019, located [here](http://deepx.ucsd.edu/public/jmcauley/steam/). 
* steam_games: A collection of all games present on the steam platform
* australian_user_items: Steam users along with the games they own.
* australian_user_reviews: Reviews made by steam users.

## Basic Recommender
In the basic recommender we create a feature vector for each game, this vector consists out of the different tags a game is able to posses and is one-hot encoded. A user will be represented by a similar feature vector, composed of the median for their reviewed games. To recommend items to users we compute the distance from their feature vector to the feature vectors of the different games. 

We utilized different distance metrics and experimented with different tf-idf approaches.
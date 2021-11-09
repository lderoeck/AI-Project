# AI-Project
Recommender system for the AI Project course based on Content-based Recommender Systems 

## Content-based recommender
Content-based recommenders only rely on the data obtainable from the items themselves to be able to compute the similarity between items. This approach has several advantages over user-based recommenders, like circumventing the cold start problem. We do not need well-established user-item relations to be able to recommend items to users, we can recommend items from the moment a user interacts with it's first item. 

## Data
The test data that will be used for this recommender is a snapshot of the steam dataset from 2019, located [here](http://deepx.ucsd.edu/public/jmcauley/steam/). 
* steam_games: A collection of all games present on the steam platform
* australian_user_items: Steam users along with the games they own.
* australian_user_reviews: Reviews made by steam users.

This dataset provides us with items (games), users and user-item interactions in the form of their game library and user reviews of games. These reviews are either positive or negative. 

## Base Recommender
![base recommender visualization (source: Content-based recommender systems: State of the art and trends)](./ContentBasedRec.png)

We first implement a base recommender based on the content-based recommendation approach, we will later attempt to optimise this recommender for our dataset. 

### Content analyzer
The content analyzer analyses the items and creates easily processable item representations. In the steam dataset, games are already subdivided in different genres and tags. The feature vector for each game is thus a simple one-hot encoded vector for all these different genres and tags.

### Profile learner
The profile learner takes into account the user feedback and item feature vectors. For steam the user feedback is provided in the form of game reviews, these reviews can be positive or negative, indicating the user likes or dislikes a certain kind of game. To create a feature vector for the user, we compute the mean for all the game feature vectors for which the user has provided a review.

TODO: explain positive/negative impact vs assuming all reviews are relevant

### Filtering component
The filtering component decides, based on the user profile, which items to recommend to the user. To implement this for our steam dataset, we simply performed a nearest neighbours search on our user vector in the item space, using a combination of different distance metrics. 

#### Distance metrics:
- Euclidian distance: `sqrt(sum((x - y)^2))`
- Cosine distance: $1-\frac{x \cdot y}{||x||_2||y||_2}$
- Manhattan distance: `sum(|x - y|)`
- Chebyshev distance: `max(|x - y|)`

#### Tf-idf methods:
- No tf-idf
- default tf-idf: `tf(t, d) * [log [n/df(t)] + 1]`
- smoothed tf-idf: `tf(t, d) * [log [(1+n)/(1+df(t))] + 1]`
- sublinear tf-idf: `[1 + log(tf)] * [log [n/df(t)] + 1]`
- smoothed sublinear tf-idf: `[1 + log(tf)] * [log [(1+n)/(1+df(t))] + 1]`

Since we utilise a one-hot encoded feature vector, the sublinear tf-idf produces identical results to the normal tf-idf. 

TODO: more info

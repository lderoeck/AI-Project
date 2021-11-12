# AI-Project
Content-based recommender system for the AI Project course

## Code organization
* `data` folder: this is where the necessary steam data files should be placed
* `src` folder:
    - `recommender.py`: Contains a class which implements the content-based recommender
    - `evaluate_recommender.py`: Contains functions which help with evaluating the recommendations
* `data_analysis.ipynb`: Notebook containing different types of data analysis results which helps to identify properties of the underlying data. This helps us to better understand why the recommender behaves in a certain way for the given dataset.
* `notebook.ipynb`: The main notebook which implements the evaluation of the recommender system for several techniques, as well as the generation of human-readable data files for qualitative evaluation.
* `evaluation` folder: This is where the output files of the recommender evaluation go. We provided a zip file which contains the generated recommendations for a number of techniques.

## Installation & running the code
First, make sure to install all necessary requirements by running `pip install -r requirements.txt`. All requirements are *hard* requirements, the code will not work without them.
Next, place all of the necessary data files (australian_user_reviews.json, australian_user_items.json, ...) in the data folder.

Now, the evaluation results can be reproduced by running the code blocks in `notebook.ipynb`. Note that generating recommendations for all of the different techniques can take a lot of time (it took around 66 minutes on an AMD Ryzen 5900x). Therefore, we also provided an `evaluation.zip` file, which contains generated recommendations for most of the technique combinations (except for feedback weighting). Simply running the last cell of the notebook then provides human-readable data files that can be used for qualitative evaluation. Generating recommendations doesn't use too much RAM however, so a system with 16GB of RAM should suffice.

## Content-based recommender
Content-based recommenders rely only on the data obtainable from the items themselves to be able to compute the similarity between items. This approach has several advantages over user-based recommenders, like circumventing the cold start problem. We do not need well-established user-item relations to be able to recommend items to users, we can recommend items from the moment a user interacts with it's first item.

## Data
The test data that will be used for this recommender is a snapshot of the steam dataset from 2019, located [here](http://deepx.ucsd.edu/public/jmcauley/steam/). 
* `steam_games`: A collection of all games present on the steam platform
* `australian_user_items`: Steam users along with the games they own.
* `australian_user_reviews`: Reviews made by steam users.

This dataset provides us with items (games), users and user-item interactions in the form of their game library and user reviews of games. These reviews are either positive or negative. 

## Base recommender
![base recommender visualization (source: Content-based recommender systems: State of the art and trends)](./ContentBasedRec.png)

We first implement a base recommender based on the content-based recommendation approach, we will later attempt to optimise this recommender for our dataset. 

### Content analyser
The content analyser is responsible for analysing the items and creating easily processable item representations. In the steam dataset, games are already subdivided in different genres and tags. The feature vector for each game is thus a simple one-hot encoded vector for all these different genres and tags. The feature matrix that is generated for the library of games is by default encoded as a sparse matrix in order to reduce memory consumption.

We can weight features under the assumption that features that occur less frequently are more informative than those that occur frequently. For this, we can use the following tf-idf weighting schemes:

#### Tf-idf methods:
- No tf-idf
- default tf-idf: `tf(t, d) * [log [n/df(t)] + 1]`
- smoothed tf-idf: `tf(t, d) * [log [(1+n)/(1+df(t))] + 1]`
- sublinear tf-idf: `[1 + log(tf)] * [log [n/df(t)] + 1]`
- smoothed sublinear tf-idf: `[1 + log(tf)] * [log [(1+n)/(1+df(t))] + 1]`

Since we utilise a one-hot encoded feature vector, we expect the sublinear tf-idf to produce identical results to the normal tf-idf (the same holds for smoothed and sublinear smoothed tf-idf).

We expect tf-idf to improve performance, as it prevents bias towards tags that are very common across the steam games library.

### Profile learner
The profile learner takes into account the user feedback and item feature vectors. With steam, the user feedback is provided in the form of game reviews. These reviews can be positive or negative, indicating the user likes or dislikes a certain kind of game. To create the user feature vector, we compute the mean of all game feature vectors for which the user has provided a review.

For this we can either opt to interpret a review as a general indicator of interest, or take into account whether a review is positive or negative. We call the latter method 'feedback weighting', as we use a weight of -1 for negative reviews and a weight of +1 for positive reviews (where the weight is multiplied with the game's feature vector). With feedback weighting, we should be able to provide recommendations that are more in line with the user's positive experiences. However, on steam users can only provide reviews for games they own, which means that they are likely interested in a type of game that they reviewed, even if the review for that game is negative.

### Filtering component
The filtering component decides, based on the user profile, which items to recommend to the user. To implement this for our steam dataset, we simply performed a nearest neighbours search on our user vector in the item space. The nearest neighbour search can be performed with different distance metrics.

#### Distance metrics:
- Euclidean distance: `sqrt(sum((x - y)^2))`
- Cosine distance: `1 - (x · y) / (l2_norm(x)*l2_norm(y))`
- Manhattan distance: `sum(|x - y|)`

Euclidean distance is distance metric that is used often for real-valued vector space problems. This is the most basic and general distance metric. This means that distance metrics that are known to perform well for our specific problem may perform better than it.

Manhattan distance computed distances for each dimension independently. Thus, it implicitly assumes that tags are independent. This may not be the case for certain tags of steam games. For example, the tags 'Local Multi-Player' and 'Multi-player' are clearly dependant. This results in unnecessary increased distances towards games with local multi-player.

In information retrieval, cosine is a widely used distance metric for ranked search because it works well for comparing queries to documents. Due to many similarities between documents and games and a similar search strategy (user profile is the query and games are documents), we should expect good performance for our recommender system as well.

To provide accurate and relevant recommendations to the user, we filter out the reviewed items from the nearest neighbours. This way we attempt to reduce the bias by keeping our train and test data strictly separated.

### Assumptions & expectations
Overall, we make the following assumptions for our recommender system and evaluation process:
- The inventories of users are the ground truth of user's interests (note that in reality this may not be the case)
- A review is an indication of interest, independent from the sentiment that is associated with it (like/dislike)
- Games are very similar to documents, so we expect cosine distance to work well
- Tf-idf prevents bias towards common tags, so we expect it to perform well

## Recommender evaluation
An in-depth evaluation can be found in [evaluation.md](./evaluation.md).
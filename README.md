# AI-Project
Recommender system for the AI Project course based on Content-based Recommender Systems 

## Basic Recommender
In the basic recommender we create a feature vector for each game, this vector consists out of the different tags a game is able to posses and is one-hot encoded. A user will be represented by a similar feature vector, composed of the median for their reviewed games. To recommend items to users we compute the distance from their feature vector to the feature vectors of the different games. 

We utilized different distance metrics and experimented with different tf-idf approaches.
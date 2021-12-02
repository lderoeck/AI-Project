from ast import literal_eval

import numpy as np
import pandas as pd
import scipy
from pandas import DataFrame
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import BallTree, KDTree, NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, Normalizer
from tqdm import tqdm
import re
import warnings
from os.path import exists
import os

# def generate_gt(target: str) -> None:
#     """Generates ground truth data to target

#     Args:
#         target (str): Path to file destination
#     """
#     gt = parse_json("./data/australian_users_items.json")
#     gt['items'] = gt['items'].apply(lambda items: [item['item_id'] for item in items])
#     gt = gt.drop(['user_url'], axis=1)
#     gt.to_parquet(target)

def parse_json(filename_python_json: str, read_max: int = -1) -> DataFrame:
    """Parses json file into a DataFrame

    Args:
        filename_python_json (str): Path to json file
        read_max (int, optional): Max amount of lines to read from json file. Defaults to -1.

    Returns:
        DataFrame: DataFrame from parsed json
    """
    with open(filename_python_json, "r", encoding="utf-8") as f:
        # parse json
        parse_data = []
        # tqdm is for showing progress bar, always good when processing large amounts of data
        for line in tqdm(f):
            # load python nested datastructure
            parsed_result = literal_eval(line)
            parse_data.append(parsed_result)
            if read_max != -1 and len(parse_data) > read_max:
                print(f"Break reading after {read_max} records")
                break
        print(f"Reading {len(parse_data)} rows.")

        # create dataframe
        df = DataFrame.from_dict(parse_data)
        return df


class BaseRecommender(object):
    def __init__(self, items_path: str, train_path: str, test_path: str, val_path: str) -> None:
        items = self._preprocess_items(pd.read_pickle(items_path))
        self.items = self._generate_item_features(items)
        self.train = pd.read_parquet(train_path)
        self.test = pd.read_parquet(test_path)
        self.val = pd.read_parquet(val_path)
        self.recommendations = pd.DataFrame()
        
    def _preprocess_items(self, items: pd.DataFrame):
        items.dropna(subset=["id"], inplace=True)
        items["price"] = items["price"].apply(lambda p: np.float32(p) if re.match(r"\d+(?:.\d{2})?", str(p)) else 0)
        items["metascore"] = items["metascore"].apply(lambda m: m if m != "NA" else np.nan)
        items = items.reset_index(drop=True)
        # TODO: transform id to new id with dict
        return items
    
    def set_user_data(self, train_path: str, test_path: str, val_path: str) -> None:
        self.train = pd.read_parquet(train_path)
        self.test = pd.read_parquet(test_path)
        self.val = pd.read_parquet(val_path)
    
    # def _preprocess_users(self, users: pd.DataFrame):
    #     users.dropna(subset=["user_id", "items"], inplace=True)
    #     users.sort_values("items_count", inplace=True)
    #     users.drop(users[users["items_count"] < 3].index, inplace=True)
    #     users.drop(users[users["items_count"] > 1024].index, inplace=True)
    #     users["item_id"] = users["items"].apply(lambda row: [game["item_id"] for game in row])
    #     users["playtime_forever"] = users["items"].apply(lambda row: [game["playtime_forever"] for game in row])
    #     users["playtime_2weeks"] = users["items"].apply(lambda row: [game["playtime_2weeks"] for game in row])
    #     users = users.drop("items", axis=1)
    #     users = users.reset_index(drop=True)
    #     return users
    
    # def _preprocess_reviews(self, reviews: pd.DataFrame):
    #     reviews.dropna(subset=["user_id", "reviews"], inplace=True)
    #     reviews.drop(reviews[~reviews["reviews"].astype(bool)].index, inplace=True) # filter out empty review sets
    #     reviews["item_id"] = reviews["reviews"].apply(lambda row: [review["item_id"] for review in row])
    #     reviews["recommend"] = reviews["reviews"].apply(lambda row: [review["recommend"] for review in row])
    #     reviews["reviews_count"] = reviews["reviews"].apply(len)
    #     reviews = reviews.drop("reviews", axis=1)
    #     reviews.sort_values("reviews_count", inplace=True)
    #     reviews = reviews.reset_index(drop=True)
    #     return reviews
    
    def _generate_item_features(self, items: pd.DataFrame):
        pass
    
    def evaluate(self, filename=None, qual_eval_folder=None, k=10, val=False) -> dict:
        """Evaluate the recommendations based on ground truth

        Args:
            recommendations (pd.DataFrame): Dataframe consisting of user_id and their respective recommendations
            filename ([type], optional): filename for qualitative evaluation. Defaults to None.
            qual_eval_folder ([type], optional): output folder for qualitative evaluation. Defaults to None.

        Returns:
            dict: a dict containing the recall@k and nDCG@k
        """
        recommendations = self.recommendations
        
        gt = self.val if val else self.test
        gt.rename(columns={"item_id": "items"}, inplace=True)
            
        eval = recommendations.drop(recommendations[~recommendations['recommendations'].astype(bool)].index)  # drop all recommendations that are empty
        eval = eval.merge(gt, left_index=True, right_index=True)

        results_dict = dict()
        # drop all rows with no items (nothing to compare against)
        # eval.drop(eval[~eval['items'].astype(bool)].index, inplace=True)

        if filename and qual_eval_folder:
            if not os.path.exists(qual_eval_folder):
                os.makedirs(qual_eval_folder)
            eval.to_parquet(qual_eval_folder + filename + '.parquet')
            
        # Cap to k recommendations
        eval['recommendations'] = eval['recommendations'].apply(lambda recs: recs[:k])
            
        # Drop reviewed items from ground truth
        # eval['items'] = eval.apply(lambda row: list(set(row['items']).difference(set(row['item_id']))), axis=1)
        # eval.drop(eval[~eval['items'].astype(bool)].index, inplace=True)
        
        # compute HR@k
        eval['HR@k'] = eval.apply(lambda row: int(any(item in row['recommendations'] for item in row['items'])), axis=1)
        results_dict[f'HR@{k}'] = eval['HR@k'].mean()

        # compute nDCG@k
        eval['nDCG@k'] = eval.apply(lambda row: np.sum([int(rec in row['items'])/(np.log2(i+2)) for i, rec in enumerate(row['recommendations'])]), axis=1)
        eval['nDCG@k'] = eval.apply(lambda row: row['nDCG@k']/np.sum([1/(np.log2(i+2)) for i in range(min(len(row['recommendations']), len(row['items'])))]), axis=1)
        results_dict[f'nDCG@{k}'] = eval['nDCG@k'].mean()

        # compute recall@k
        eval['items'] = eval['items'].apply(set)
        eval['recommendations'] = eval['recommendations'].apply(set)
        eval['recall@k'] = eval.apply(lambda row: len(row['recommendations'].intersection(row['items']))/len(row['items']), axis=1)
        results_dict[f'recall@{k}'] = eval['recall@k'].mean()

        eval['ideal_recall@k'] = eval.apply(lambda row: min(len(row['items']), len(row["recommendations"]))/len(row['items']), axis=1)
        results_dict[f'ideal_recall@{k}'] = eval['ideal_recall@k'].mean()
        
        eval['nRecall@k'] = eval.apply(lambda row: row['recall@k']/row['ideal_recall@k'], axis=1)
        results_dict[f'nRecall@{k}'] = eval['nRecall@k'].mean()

        return results_dict
    
    
class ContentBasedRecommender(BaseRecommender):
    def __init__(self, items_path: str, train_path: str, test_path: str, val_path: str, sparse: bool = True, tfidf='default', normalize=False) -> None:
        """Content based recommender

        Args:
            items_path (str): Path to pickle file containing the items
            sparse (bool, optional): If recommender uses a sparse representation. Defaults to True.
            distance_metric (str, optional): Which distance metric to use. Defaults to 'minkowski'.
            dim_red ([type], optional): Which dimensionality reduction to use. Defaults to None.
            tfidf (str, optional): Which tf-idf method to use. Defaults to 'default'.
            use_feedback (bool, optional): Consider positive/negative reviews. Defaults to True.
        """
        self.sparse = sparse
        self.normalize = normalize
        self.recommendations = None
        self.normalizer = Normalizer(copy=False)
        
        # Select tf-idf method to use
        self.tfidf = None
        if tfidf == 'default':
            self.tfidf = TfidfTransformer(smooth_idf=False, sublinear_tf=False)
        elif tfidf == 'smooth':
            self.tfidf = TfidfTransformer(smooth_idf=True, sublinear_tf=False)
        elif tfidf == 'sublinear':
            self.tfidf = TfidfTransformer(smooth_idf=False, sublinear_tf=True)
        elif tfidf == 'smooth_sublinear':
            self.tfidf = TfidfTransformer(smooth_idf=True, sublinear_tf=True)

        # Select algorithm to use for neighbour computation
        algorithm = 'auto'
        self.method = NearestNeighbors(n_neighbors=10, algorithm=algorithm, metric='cosine')
        
        super(ContentBasedRecommender, self).__init__(items_path, train_path, test_path, val_path)
        
    def _process_item_features(self, items):
        columns = ["genres", "tags"]
        items = items.filter(columns)
        return items

    def _generate_item_features(self, items: DataFrame) -> DataFrame:
        """Generates feature vector of items and appends to returned DataFrame

        Args:
            items (DataFrame): dataframe containing the items

        Returns:
            DataFrame: dataframe with feature vector appended
        """
        items = self._process_item_features(items)
        # Combine all features into one column
        columns = items.columns.tolist()
        for col in columns:
            items[col] = items[col].fillna("").apply(set)
        items["tags"] = items.apply(lambda x: list(
            set.union(*([x[col] for col in columns]))), axis=1)
        columns.remove("tags")
        items = items.drop(columns, axis=1)

        # Compute one-hot encoded vector of tags
        mlb = MultiLabelBinarizer(sparse_output=self.sparse)
        if self.sparse:
            items = items.join(DataFrame.sparse.from_spmatrix(mlb.fit_transform(items.pop(
                "tags")), index=items.index, columns=["tag_" + c for c in mlb.classes_]))
        else:
            items = items.join(DataFrame(mlb.fit_transform(items.pop(
                "tags")), index=items.index, columns=["tag_" + c for c in mlb.classes_]))

        return items

    def generate_recommendations(self, amount=10, read_max=None) -> None:
        """Generate recommendations based on user review data

        Args:
            data_path (str): User review data
            amount (int, optional): Amount of times to recommend. Defaults to 10.
            read_max (int, optional): Max amount of users to read. Defaults to None.
        """
        items = self.items
        df = self.train.iloc[:read_max].copy(deep=True) if read_max else self.train

        # Drop id so only feature vector is left
        if self.sparse:
            X = scipy.sparse.csr_matrix(items.values)
        else:
            X = np.array(items.values)

        if self.tfidf:
            # Use tf-idf
            X = self.tfidf.fit_transform(X)
            
        if self.normalize:
            X = self.normalizer.fit_transform(X)

        # Transformed feature vector back into items
        if self.sparse:
            items = DataFrame.sparse.from_spmatrix(X)
        else:
            items = DataFrame(X)

        self.method.set_params(n_neighbors=amount)
        nbrs = self.method.fit(X)

        recommendation_list = []
        for index, row in tqdm(df.iterrows()):
            # Compute uservector and recommendations for all users
            owned_items = items.iloc[row["item_id"],:]

            # If user has no items, no usable data is available
            assert not owned_items.empty

            # Computing average, assuming all user items are indication of interest
            user_vector = owned_items.mean()

            if self.normalize:
                user_vector = self.normalizer.transform([user_vector.to_numpy()])
            else:
                user_vector = [user_vector.to_numpy()]
            # Start overhead of 20%
            gen_am = amount//5
            recommendations = []
            while len(recommendations) < amount:
                # calculate amount of items to be generated
                gen_am += amount - len(recommendations)
                nns = nbrs.kneighbors(user_vector, gen_am, return_distance=True)
                # Filter out items in training set
                recommendations = list(filter(lambda id: id not in row["item_id"], nns[1][0]))

            recommendation_list.append(recommendations[:amount])

        df["recommendations"] = recommendation_list
        self.recommendations = df



class ImprovedRecommender(ContentBasedRecommender):
    def __init__(self, items_path: str, train_path: str, test_path: str, val_path: str, sparse: bool = True, dim_red=None, tfidf='default', use_feedback=True, normalize=False) -> None:
        """Content based recommender

        Args:
            items_path (str): Path to pickle file containing the items
            sparse (bool, optional): If recommender uses a sparse representation. Defaults to True.
            distance_metric (str, optional): Which distance metric to use. Defaults to 'minkowski'.
            dim_red ([type], optional): Which dimensionality reduction to use. Defaults to None.
            tfidf (str, optional): Which tf-idf method to use. Defaults to 'default'.
            use_feedback (bool, optional): Consider positive/negative reviews. Defaults to True.
        """
        self.dim_red = dim_red
        self.use_feedback = use_feedback

        super(ImprovedRecommender, self).__init__(items_path, train_path, test_path, val_path, sparse, tfidf, normalize)
        
    def _process_item_features(self, items):
        columns = ["genres", "tags", "specs", "developer", "publisher"]
        items = items.filter(columns)
        items["developer"].fillna(value='', inplace=True)
        items["publisher"].fillna(value='', inplace=True)
        items["developer"] = items["developer"].apply(lambda my_str: my_str.split(','))
        items["publisher"] = items["publisher"].apply(lambda my_str: my_str.split(','))
        return items

    def generate_recommendations(self, amount=10, read_max=None) -> None:
        """Generate recommendations based on user review data

        Args:
            data_path (str): User review data
            amount (int, optional): Amount of times to recommend. Defaults to 10.
            read_max (int, optional): Max amount of users to read. Defaults to None.
        """
        items = self.items
        
        df = self.train.iloc[:read_max].copy(deep=True) if read_max else self.train

        # Drop id so only feature vector is left
        if self.sparse:
            X = scipy.sparse.csr_matrix(items.values)
        else:
            X = np.array(items.values)

        if self.tfidf:
            # Use tf-idf
            X = self.tfidf.fit_transform(X)

        if self.dim_red:
            # Use dimensionality reduction
            X = self.dim_red.fit_transform(X)
            
        if self.normalize:
            X = self.normalizer.fit_transform(X)

        # Combine transformed feature vector back into items
        if self.sparse and self.dim_red:
            warnings.warn("Sparse was set to 'True' but dimensionality reduction is used, using dense matrix representation instead.", RuntimeWarning)
        if self.sparse and self.dim_red is None:
            items = DataFrame.sparse.from_spmatrix(X)
        else:
            items = DataFrame(X)

        self.method.set_params(n_neighbors=amount)
        nbrs = self.method.fit(X)

        recommendation_list = []
        for index, row in tqdm(df.iterrows()):
            # Compute uservector and recommendations for all users
            reviewed_items = items.iloc[row["item_id"],:]

            # If user has no reviews, no usable data is available
            assert not reviewed_items.empty

            user_vector = None
            if self.use_feedback:
                # TODO implement!
                raise NotImplementedError
            else:
                # Computing average, assuming all reviews are indication of interest
                user_vector = reviewed_items.mean()

            if self.normalize:
                user_vector = self.normalizer.transform([user_vector.to_numpy()])
            else:
                user_vector = [user_vector.to_numpy()]
            # Start overhead of 20%
            gen_am = amount//5
            recommendations = []
            while len(recommendations) < amount:
                # calculate amount of items to be generated
                gen_am += amount - len(recommendations)
                nns = nbrs.kneighbors(user_vector, gen_am, return_distance=True)
                # Filter out items in reviews
                recommendations = list(filter(lambda id: id not in row["item_id"], nns[1][0]))

            recommendation_list.append(recommendations[:amount])

        df["recommendations"] = recommendation_list
        self.recommendations = df


if __name__ == "__main__":
    pass

class PopBasedRecommender(BaseRecommender):
    def __init__(self, train_path: str, test_path: str, val_path: str) -> None:
        self.train = pd.read_parquet(train_path)
        self.test = pd.read_parquet(test_path)
        self.val = pd.read_parquet(val_path)

    def generate_recommendations(self, read_max=None) -> None:
        # reviews = parse_json(data_path) if read_max is None else parse_json(data_path, read_max=read_max)
        # df = self._preprocess_reviews(reviews)
        df = self.train.iloc[:read_max].copy(deep=True) if read_max else self.train

        n_game_pop = df["item_id"].explode()
        n_game_pop.dropna(inplace=True)
        n_game_pop = n_game_pop.value_counts()

        df["recommendations"] = df["item_id"].apply(lambda x: [rec for rec in n_game_pop.index if rec not in x][:10]) 
        self.recommendations = df

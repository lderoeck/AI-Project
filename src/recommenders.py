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

def generate_gt(target: str) -> None:
    """Generates ground truth data to target

    Args:
        target (str): Path to file destination
    """
    gt = parse_json("./data/australian_users_items.json")
    gt['items'] = gt['items'].apply(lambda items: [item['item_id'] for item in items])
    gt = gt.drop(['user_url'], axis=1)
    gt.to_parquet(target)

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
    def __init__(self, items_path: str) -> None:
        items = self._preprocess_items(parse_json(items_path))
        self.items = self._generate_item_features(items)
        self.recommendations = pd.DataFrame()
        
    def _preprocess_items(self, items: pd.DataFrame):
        items.dropna(subset=["id"], inplace=True)
        items["price"] = items["price"].apply(lambda p: np.float32(p) if re.match(r"\d+(?:.\d{2})?", str(p)) else 0)
        items["metascore"] = items["metascore"].apply(lambda m: m if m != "NA" else np.nan)
        items = items.reset_index(drop=True)
        return items
    
    def _preprocess_users(self, users: pd.DataFrame):
        users.dropna(subset=["user_id", "items"], inplace=True)
        users.sort_values("items_count", inplace=True)
        users.drop(users[users["items_count"] < 3].index, inplace=True)
        users.drop(users[users["items_count"] > 1024].index, inplace=True)
        users = users.reset_index(drop=True)
        return users
    
    def _preprocess_reviews(self, reviews: pd.DataFrame):
        reviews.dropna(subset=["user_id", "reviews"], inplace=True)
        reviews["reviews"] = reviews["reviews"].apply(lambda lst: [{"item_id": i["item_id"], "recommend": i["recommend"]} for i in lst]) # explode
        reviews["item_id"] = reviews["reviews"].apply(lambda row: [review["item_id"] for review in row])
        reviews["recommend"] = reviews["reviews"].apply(lambda row: [review["recommend"] for review in row])
        reviews["reviews_count"] = reviews["reviews"].apply(len)
        reviews.sort_values("reviews_count", inplace=True)
        reviews.drop(reviews[~reviews["reviews"].astype(bool)].index, inplace=True) # filter out empty review sets
        reviews = reviews.reset_index(drop=True)
        return reviews
    
    def _generate_item_features(self, items: pd.DataFrame):
        pass
    
    def evaluate(self, filename=None, qual_eval_folder=None, k=10) -> dict:
        """Evaluate the recommendations based on ground truth

        Args:
            recommendations (pd.DataFrame): Dataframe consisting of user_id and their respective recommendations
            filename ([type], optional): filename for qualitative evaluation. Defaults to None.
            qual_eval_folder ([type], optional): output folder for qualitative evaluation. Defaults to None.

        Returns:
            dict: a dict containing the recall@k and nDCG@k
        """
        recommendations = self.recommendations
        gt_file = './data/ground_truth.parquet'
        if not exists(gt_file):
            generate_gt(gt_file)
            
        eval = recommendations.drop(recommendations[~recommendations['recommendations'].astype(bool)].index)  # drop all recommendations that are empty
        gt = pd.read_parquet('./data/ground_truth.parquet')
        eval = eval.merge(gt, on=['user_id'])

        results_dict = dict()
        # drop all rows with no items (nothing to compare against)
        eval.drop(eval[~eval['items'].astype(bool)].index, inplace=True)

        if filename and qual_eval_folder:
            if not os.path.exists(qual_eval_folder):
                os.makedirs(qual_eval_folder)
            eval.to_parquet(qual_eval_folder + filename + '.parquet')
            
        # Cap to k recommendations
        eval['recommendations'] = eval['recommendations'].apply(lambda recs: recs[:k])
            
        # Drop reviewed items from ground truth
        eval['items'] = eval.apply(lambda row: list(set(row['items']).difference(set(row['item_id']))), axis=1)
        eval.drop(eval[~eval['items'].astype(bool)].index, inplace=True)
        
        # compute HR@k
        eval['HR@k'] = eval.apply(lambda row: int(any(rec in row['recommendations'] for rec in row['items'])), axis=1)
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
    def __init__(self, items_path: str, sparse: bool = True, distance_metric='minkowski', tfidf='default', use_feedback=True, normalize=True) -> None:
        """Content based recommender

        Args:
            items_path (str): Path to json file containing the items
            sparse (bool, optional): If recommender uses a sparse representation. Defaults to True.
            distance_metric (str, optional): Which distance metric to use. Defaults to 'minkowski'.
            tfidf (str, optional): Which tf-idf method to use. Defaults to 'default'.
            use_feedback (bool, optional): Consider positive/negative reviews. Defaults to True.
        """
        self.sparse = sparse
        self.use_feedback = use_feedback
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
        if distance_metric in BallTree.valid_metrics:
            algorithm = 'ball_tree'
        elif distance_metric in KDTree.valid_metrics:
            algorithm = 'kd_tree'
        self.method = NearestNeighbors(n_neighbors=10, algorithm=algorithm, metric=distance_metric)
        super(ContentBasedRecommender, self).__init__(items_path)

    def _generate_item_features(self, items: DataFrame) -> DataFrame:
        """Generates feature vector of items and appends to returned DataFrame

        Args:
            items (DataFrame): dataframe containing the items

        Returns:
            DataFrame: dataframe with feature vector appended
        """
        items.drop(["publisher", "app_name", "title", "url", "release_date", "discount_price", "reviews_url",
                    "price", "early_access", "sentiment", "metascore"], axis=1, inplace=True)
        columns = ["genres", "tags"]
        # Combine genres, tags and specs into one column
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

    def generate_recommendations(self, data_path: str, amount=10, read_max=None) -> None:
        """Generate recommendations based on user review data

        Args:
            data_path (str): User review data
            amount (int, optional): Amount of times to recommend. Defaults to 10.
            read_max (int, optional): Max amount of users to read. Defaults to None.
        """
        items = self.items
        df = parse_json(data_path) if read_max is None else parse_json(data_path, read_max=read_max)
        df.drop(df[~df["reviews"].astype(bool)].index,inplace=True)  # filter out empty reviews

        # Process reviews
        df = df.explode("reviews", ignore_index=True)
        df = pd.concat([df.drop(["reviews", "user_url"], axis=1), pd.json_normalize(df.reviews)], 
                axis=1).drop(["funny", "helpful", "posted", "last_edited", "review"], axis=1)
        df = df.groupby("user_id").agg(list).reset_index()

        # Drop id so only feature vector is left
        if self.sparse:
            X = scipy.sparse.csr_matrix(items.drop(["id"], axis=1).values)
        else:
            X = np.array(items.drop(["id"], axis=1).values)

        if self.tfidf:
            # Use tf-idf
            X = self.tfidf.fit_transform(X)
            
        if self.normalize:
            X = self.normalizer.fit_transform(X)

        # Combine transformed feature vector back into items
        if self.sparse:
            items = pd.concat([items["id"], DataFrame.sparse.from_spmatrix(X)], axis=1)
        else:
            items = pd.concat([items["id"], DataFrame(X)], axis=1)

        self.method.set_params(n_neighbors=amount)
        nbrs = self.method.fit(X)

        recommendation_list = []
        for index, row in tqdm(df.iterrows()):
            # Compute uservector and recommendations for all users
            reviewed_items = items[items["id"].isin(row["item_id"])]

            # If user has no reviews, no usable data is available
            if reviewed_items.empty:
                recommendation_list.append([])
                continue

            user_vector = None
            if self.use_feedback:
                # Compute average taking into account if review is positive/negative
                reviewed_item_ids = np.array(row["item_id"])
                recommend = np.array(row["recommend"])

                positive_ids = reviewed_item_ids[recommend]
                negative_ids = reviewed_item_ids[~recommend]

                positive_values = reviewed_items[reviewed_items["id"].isin(positive_ids)].drop(["id"], axis=1).sum()
                negative_values = reviewed_items[reviewed_items["id"].isin(negative_ids)].drop(["id"], axis=1).sum()

                user_vector = positive_values.sub(negative_values).div(reviewed_items.shape[0])
            else:
                # Computing average, assuming all reviews are indication of interest
                user_vector = reviewed_items.drop(["id"], axis=1).mean()

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
                recommendations = list(filter(lambda id: id not in row["item_id"], [items.loc[item]["id"] for item in nns[1][0]]))

            recommendation_list.append(recommendations[:amount])

        df["recommendations"] = recommendation_list
        self.recommendations = df


class ImprovedRecommender(BaseRecommender):
    def __init__(self, items_path: str = './data/steam_games.json', sparse: bool = True, dim_red=None, tfidf='default', use_feedback=True, normalize=False) -> None:
        """Content based recommender

        Args:
            items_path (str): Path to json file containing the items
            sparse (bool, optional): If recommender uses a sparse representation. Defaults to True.
            distance_metric (str, optional): Which distance metric to use. Defaults to 'minkowski'.
            dim_red ([type], optional): Which dimensionality reduction to use. Defaults to None.
            tfidf (str, optional): Which tf-idf method to use. Defaults to 'default'.
            use_feedback (bool, optional): Consider positive/negative reviews. Defaults to True.
        """
        self.sparse = sparse
        self.dim_red = dim_red
        self.use_feedback = use_feedback
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
        
        super(ImprovedRecommender, self).__init__(items_path)

    def _generate_item_features(self, items: DataFrame) -> DataFrame:
        """Generates feature vector of items and appends to returned DataFrame

        Args:
            items (DataFrame): dataframe containing the items

        Returns:
            DataFrame: dataframe with feature vector appended
        """
        items.drop(["publisher", "app_name", "title", "url", "release_date", "discount_price", "reviews_url",
                    "price", "early_access", "sentiment", "metascore"], axis=1, inplace=True)
        columns = ["genres", "tags", "specs", "developer"]
        # items["developer"] = items["developer"].apply(lambda my_str: my_str.split(','))
        # Combine genres, tags and specs into one column
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

    def generate_recommendations(self, data_path: str, amount=10, read_max=None) -> None:
        """Generate recommendations based on user review data

        Args:
            data_path (str): User review data
            amount (int, optional): Amount of times to recommend. Defaults to 10.
            read_max (int, optional): Max amount of users to read. Defaults to None.
        """
        items = self.items
        
        df = self._preprocess_reviews(parse_json(data_path, read_max=read_max))
        # df = parse_json(data_path) if read_max is None else parse_json(data_path, read_max=read_max)
        # df.drop(df[~df["reviews"].astype(bool)].index,inplace=True)  # filter out empty reviews

        # # Process reviews
        # df = df.explode("reviews", ignore_index=True)
        # df = pd.concat([df.drop(["reviews", "user_url"], axis=1), pd.json_normalize(df.reviews)], 
        #         axis=1).drop(["funny", "helpful", "posted", "last_edited", "review"], axis=1)
        # df = df.groupby("user_id").agg(list).reset_index()

        # Drop id so only feature vector is left
        if self.sparse:
            X = scipy.sparse.csr_matrix(items.drop(["id"], axis=1).values)
        else:
            X = np.array(items.drop(["id"], axis=1).values)

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
            items = pd.concat([items["id"], DataFrame.sparse.from_spmatrix(X)], axis=1)
        else:
            items = pd.concat([items["id"], DataFrame(X)], axis=1)

        self.method.set_params(n_neighbors=amount)
        nbrs = self.method.fit(X)

        recommendation_list = []
        for index, row in tqdm(df.iterrows()):
            review = row["reviews"]
            # Compute uservector and recommendations for all users
            reviewed_items = items[items["id"].isin(row["item_id"])]

            # If user has no reviews, no usable data is available
            if reviewed_items.empty:
                recommendation_list.append([])
                continue

            user_vector = None
            if self.use_feedback:
                # Compute average taking into account if review is positive/negative
                reviewed_item_ids = np.array(row["item_id"])
                recommend = np.array(row["recommend"])

                positive_ids = reviewed_item_ids[recommend]
                negative_ids = reviewed_item_ids[~recommend]

                positive_values = reviewed_items[reviewed_items["id"].isin(positive_ids)].drop(["id"], axis=1).sum()
                negative_values = reviewed_items[reviewed_items["id"].isin(negative_ids)].drop(["id"], axis=1).sum()

                user_vector = positive_values.sub(negative_values).div(reviewed_items.shape[0])
            else:
                # Computing average, assuming all reviews are indication of interest
                user_vector = reviewed_items.drop(["id"], axis=1).mean()

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
                recommendations = list(filter(lambda id: id not in row["item_id"], [items.loc[item]["id"] for item in nns[1][0]]))

            recommendation_list.append(recommendations[:amount])

        df["recommendations"] = recommendation_list
        self.recommendations = df


if __name__ == "__main__":
    pass

class PopBasedRecommender(BaseRecommender):
    def __init__(self) -> None:
        pass

    def generate_recommendations(self, data_path: str, read_max=None) -> None:
        # TODO: controleren of sommige items verschillende id's hebben, belangrijk voor pop based recommender
        reviews = parse_json(data_path) if read_max is None else parse_json(data_path, read_max=read_max)
        df = self._preprocess_reviews(reviews)

        n_game_pop = df["item_id"].explode()
        n_game_pop.dropna(inplace=True)
        n_game_pop = n_game_pop.value_counts()

        df["recommendations"] = df["item_id"].apply(lambda x: [rec for rec in n_game_pop.index if rec not in x][:10]) 
        self.recommendations = df

from ast import literal_eval

import numpy as np
import pandas as pd
import scipy
from pandas import DataFrame
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import BallTree, KDTree, NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm


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

# TODO: use seed for SVD


class ContentBasedRec(object):
    def __init__(self, items_path: str, sparse: bool = True, distance_metric='minkowski', dim_red=None, tfidf='default', use_feedback=True) -> None:
        """Content based recommender

        Args:
            items_path (str): Path to json file containing the items
            sparse (bool, optional): If recommender uses a sparse representation. Defaults to True.
            distance_metric (str, optional): Which distance metric to use. Defaults to 'minkowski'.
            dim_red ([type], optional): Which dimensionality reduction to use. Defaults to None.
            tfidf (str, optional): Which tf-idf method to use. Defaults to 'default'.
            use_feedback (bool, optional): Consider positive/negative reviews. Defaults to True.
        """
        super().__init__()
        self.sparse = sparse
        self.dim_red = dim_red
        self.use_feedback = use_feedback
        self.items = self._generate_item_features(parse_json(items_path))
        self.recommendations = None
        
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

    def _generate_item_features(self, items: DataFrame) -> DataFrame:
        """Generates feature vector of items and appends to returned DataFrame

        Args:
            items (DataFrame): dataframe containing the items

        Returns:
            DataFrame: dataframe with feature vector appended
        """
        items.drop(["publisher", "app_name", "title", "url", "release_date", "discount_price", "reviews_url",
                    "price", "early_access", "developer", "sentiment", "metascore"], axis=1, inplace=True)
        items.dropna(subset=["id"], inplace=True)
        items = items.reset_index(drop=True)
        # Combine genres, tags and specs into one column
        items["genres"] = items["genres"].fillna("").apply(set)
        items["tags"] = items["tags"].fillna("").apply(set)
        items["specs"] = items["genres"].fillna("").apply(set)
        items["tags"] = items.apply(lambda x: list(
            set.union(x["genres"], x["tags"], x["specs"])), axis=1)
        items = items.drop(["genres", "specs"], axis=1)

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

        if self.dim_red:
            # Use dimensionality reduction
            X = self.dim_red.fit_transform(X)

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

            # Start overhead of 20%
            gen_am = amount//5
            recommendations = []
            while len(recommendations) < amount:
                # calculate amount of items to be generated
                gen_am += amount - len(recommendations)
                nns = nbrs.kneighbors([user_vector.to_numpy()], gen_am, return_distance=True)
                # Filter out items in reviews
                recommendations = list(filter(lambda id: id not in row["item_id"], [items.loc[item]["id"] for item in nns[1][0]]))

            recommendation_list.append(recommendations)

        df["recommendations"] = recommendation_list
        self.recommendations = df


if __name__ == "__main__":
    rec = ContentBasedRec("./data/steam_games.json", distance_metric="cosine")
    rec.generate_recommendations("./data/australian_user_reviews.json")
    print(rec.recommendations)

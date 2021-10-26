from ast import literal_eval

import numpy as np
import pandas as pd
import scipy
from pandas import DataFrame
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
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


class ContentBasedRec(object):
    def __init__(self, items_path: str, sparse: bool = True, method=NearestNeighbors(n_neighbors=10, algorithm="ball_tree"), dim_red=TruncatedSVD(n_components=50)) -> None:
        super().__init__()
        self.sparse = sparse
        self.method = method
        self.dim_red = dim_red
        self.items = self._generate_item_features(parse_json(items_path))
        self.recommendations = None

    def _generate_item_features(self, items: DataFrame) -> DataFrame:
        """Generates feature vector of items and appends to returned DataFrame

        Args:
            items (DataFrame): dataframe containing the items

        Returns:
            DataFrame: dataframe with feature vector appended
        """
        item_data = {"publisher", "genres", "app_name", "title", "url", "release_date", "tags", "discount_price",
                     "reviews_url", "specs", "price", "early_access", "id", "developer", "sentiment", "metascore"}
        assert all(
            column in item_data for column in items.columns.values.tolist())

        items.drop(list(item_data.difference(
            {"genres", "tags", "id", "specs"})), axis=1, inplace=True)
        items["id"].dropna(inplace=True)
        # Combine genres, tags and specs into one column
        items["genres"] = items["genres"].fillna("").apply(set)
        items["tags"] = items["tags"].fillna("").apply(set)
        items["specs"] = items["genres"].fillna("").apply(set)
        items["tags"] = items.apply(lambda x: list(
            set.union(x["genres"], x["tags"], x["specs"])), axis=1)
        items = items.drop(["genres", "specs"], axis=1)

        mlb = MultiLabelBinarizer(sparse_output=self.sparse)
        if self.sparse:
            items = items.join(DataFrame.sparse.from_spmatrix(mlb.fit_transform(items.pop(
                "tags")), index=items.index, columns=["tag_" + c for c in mlb.classes_]))
        else:
            items = items.join(DataFrame(mlb.fit_transform(items.pop(
                "tags")), index=items.index, columns=["tag_" + c for c in mlb.classes_]))

        return items

    def generate_recommendations(self, data_path: str) -> None:
        """Generate recommendations based on user review data

        Args:
            data_path (str): User review data
        """
        items = self.items
        df = parse_json(data_path)
        df = df.explode("reviews", ignore_index=True)

        df = pd.concat([df.drop(["reviews", "user_url"], axis=1), pd.json_normalize(
            df.reviews)], axis=1).drop(["funny", "helpful", "posted", "last_edited", "review"], axis=1)
        df = df.groupby("user_id")["item_id"].apply(
            list).reset_index(name="item_id")

        if self.sparse:
            X = scipy.sparse.csr_matrix(items.drop(["id"], axis=1).values)
        else:
            X = np.matrix(items.drop(["id"], axis=1).values)

        if self.dim_red:
            X = self.dim_red.fit_transform(X)
        items = pd.concat([items["id"], DataFrame(X)], axis=1)

        nbrs = self.method.fit(X)

        recommendation_list = []
        for index, row in tqdm(df.iterrows()):
            reviewed_items = items[items["id"].isin(row["item_id"])]
            if reviewed_items.empty:
                recommendation_list.append([])
                continue
            user_vector = reviewed_items.drop(["id"], axis=1).mean()
            nns = nbrs.kneighbors([user_vector.to_numpy()],
                                  10, return_distance=False)[0]
            recommendations = [items.loc[item]["id"] for item in nns]
            for recommendation in recommendations:
                assert not isinstance(recommendation, list)
            recommendation_list.append(recommendations)

        df["recommendations"] = recommendation_list
        self.recommendations = df


if __name__ == "__main__":
    test = ContentBasedRec("./data/steam_games.json", sparse=False)

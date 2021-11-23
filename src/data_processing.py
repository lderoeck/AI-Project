from os import remove
from recommender import parse_json
import re
import pandas as pd
import numpy as np

reviews = parse_json("./data/australian_user_reviews.json")
reviews.dropna(subset=["user_id", "reviews"], inplace=True)
reviews["reviews"] = reviews["reviews"].apply(lambda lst: [{"item_id": i["item_id"], "recommend": i["recommend"]} for i in lst])
reviews["reviews_count"] = reviews["reviews"].apply(len)
reviews.sort_values("reviews_count", inplace=True)
reviews.drop(reviews[~reviews["reviews"].astype(bool)].index, inplace=True)
reviews.reset_index()
reviews[["user_id", "reviews"]].to_parquet("./data/australian_user_reviews.parquet")

del reviews

user_items = parse_json("./data/australian_users_items.json")
user_items.dropna(subset=["user_id", "items"], inplace=True)
user_items.sort_values("items_count", inplace=True)
user_items.drop(user_items[user_items["items_count"] < 3].index, inplace=True)
user_items.drop(user_items[user_items["items_count"] > 1024].index, inplace=True)
user_items.reset_index()
user_items.to_parquet("./data/australian_users_items.parquet")

del user_items

steam_games = parse_json("./data/steam_games.json")
steam_games.dropna(subset=["id"], inplace=True)
steam_games["price"] = steam_games["price"].apply(lambda p: np.float32(p) if re.match(r"\d+(?:.\d{2})?", str(p)) else 0)
steam_games["metascore"] = steam_games["metascore"].apply(lambda m: m if m != "NA" else np.nan)
steam_games.reset_index()
steam_games.to_parquet("./data/steam_games.parquet")

del steam_games
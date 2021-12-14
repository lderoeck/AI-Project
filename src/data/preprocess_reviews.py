from ast import literal_eval
from tqdm import tqdm
import pandas as pd

def parse_json(filename_python_json: str, read_max: int = -1) -> pd.DataFrame:
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
        df = pd.DataFrame.from_dict(parse_data)
        return df
    
def unpack_split(df, col):
    return pd.DataFrame({
        'item_id': df[col].apply(lambda x: [game[0] for game in x])})
    
def convert(item_ids, games):
    return games.index[games['id'].isin(item_ids)].tolist()

reviews = parse_json("./data/australian_user_reviews.json")
reviews.dropna(subset=["user_id", "reviews"], inplace=True)
reviews.drop(reviews[~reviews["reviews"].astype(bool)].index, inplace=True) # filter out empty review sets
reviews.drop(columns=['user_url'])
reviews["item_id"] = reviews["reviews"].apply(lambda row: [review["item_id"] for review in row])
reviews["recommend"] = reviews["reviews"].apply(lambda row: [review["recommend"] for review in row])
reviews["reviews_count"] = reviews["reviews"].apply(len)
reviews = reviews.drop("reviews", axis=1)
reviews.sort_values("reviews_count", inplace=True)
reviews = reviews.reset_index(drop=True)
reviews.drop_duplicates(subset=['user_id'], inplace=True)

user_ids = pd.read_parquet('./data/user_ids.parquet')
reviews = user_ids.reset_index().merge(reviews, on='user_id').set_index('index')

users = pd.read_pickle('./data/interactions.pkl')
users = unpack_split(users, 'interactions')
users = users.merge(reviews, left_index=True, right_index=True)

games = pd.read_pickle('./data/games.pkl')

users['item_id_y'] = users['item_id_y'].apply(lambda x: convert(x, games))
users.drop(users[~users['item_id_y'].astype(bool)].index, inplace=True) # filter out empty review sets

# check that most reviews are within the items of the user
users['reviews_in_games'] = users.apply(lambda row: len(set(row['item_id_x']).intersection(set(row['item_id_y'])))/len(row['item_id_y']), axis=1)
print(users['reviews_in_games'].mean())

reviews = users[['item_id_y', 'recommend']].rename(columns={'item_id_y': 'reviews'})

reviews.to_parquet('./data/reviews.parquet')

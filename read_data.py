
import pandas as pd
from pandas.core.frame import DataFrame
from tqdm import tqdm
import ast
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
import scipy


#read file line-by-line and parse json, returns dataframe
def parse_json(filename_python_json:str, read_max:int=-1) -> pd.DataFrame:
    with open(filename_python_json, "r", encoding="utf-8") as f:
        #parse json
        parse_data = []
        for line in tqdm(f): #tqdm is for showing progress bar, always good when processing large amounts of data
            # line = line.decode('utf-8')
            # line = line.replace('true','True') #difference json/python
            # line = line.replace('false','False')
            parsed_result = ast.literal_eval(line) #load python nested datastructure
            parse_data.append(parsed_result)
            if read_max !=-1 and len(parse_data) > read_max:
                print(f'Break reading after {read_max} records')
                break
        print(f"Reading {len(parse_data)} rows.")

        #create dataframe
        df= pd.DataFrame.from_dict(parse_data)
        return df
    
# df = parse_json('./data/steam_reviews.json')
# print(df.head())

def get_item_features():
    items = parse_json('./data/steam_games.json')
    items = items[['genres', 'tags', 'id','specs']]
    items['id'].dropna(inplace=True)
    # print(items['genres'].tolist())
    # print(items)
    # assert False
    items['genres'] = items['genres'].fillna("").apply(set)
    items['tags'] = items['tags'].fillna("").apply(set)
    items['specs'] = items['genres'].fillna("").apply(set)
    items['tags'] = items.apply(lambda x: list(set.union(x['genres'], x['tags'], x['specs'])), axis=1)
    items = items.drop(['genres', 'specs'], axis=1)
    mlb = MultiLabelBinarizer(sparse_output=True)
    items = items.join(pd.DataFrame.sparse.from_spmatrix(mlb.fit_transform(items.pop('tags')), index=items.index, columns=['tag_' + c for c in mlb.classes_]))
    return items

items = get_item_features()
df = parse_json('./data/australian_user_reviews.json')
df = df.explode('reviews', ignore_index=True)

df = pd.concat([df.drop(['reviews', 'user_url'], axis=1), pd.json_normalize(df.reviews)], axis=1).drop(['funny', 'helpful', 'posted', 'last_edited', 'review'], axis=1)
df = df.groupby('user_id')['item_id'].apply(list).reset_index(name='item_id')
print(df)
X = scipy.sparse.csr_matrix(items.drop(['id'], axis=1).values)
nbrs = NearestNeighbors(n_neighbors=10).fit(X)

recommendation_list = []
for index, row in tqdm(df.iterrows()):
    reviewed_items = items[items['id'].isin(row['item_id'])]
    if reviewed_items.empty:
        recommendation_list.append([])
        continue
    user_vector = reviewed_items.drop(['id'], axis=1).mean()
    nns = nbrs.kneighbors(scipy.sparse.csr_matrix(user_vector.values), 10, return_distance=False)[0]
    recommendations = [items.loc[item]['id'] for item in nns]
    recommendation_list.append(recommendations)

df['recommendations'] = recommendation_list

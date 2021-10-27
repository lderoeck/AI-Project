import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
import scipy
from recommender import parse_json
    
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

def generate_recommendations():
    items = get_item_features()
    df = parse_json('./data/australian_user_reviews.json')
    df = df.explode('reviews', ignore_index=True)

    df = pd.concat([df.drop(['reviews', 'user_url'], axis=1), pd.json_normalize(df.reviews)], axis=1).drop(['funny', 'helpful', 'posted', 'last_edited', 'review'], axis=1)
    df = df.groupby('user_id')['item_id'].apply(list).reset_index(name='item_id')
    X = scipy.sparse.csr_matrix(items.drop(['id'], axis=1).values)
    svd = TruncatedSVD(n_components=50)
    Y = svd.fit_transform(X)
    items = pd.concat([items['id'], pd.DataFrame(Y)], axis=1)
    
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(Y)
    
    recommendation_list = []
    for index, row in tqdm(df.iterrows()):
        reviewed_items = items[items['id'].isin(row['item_id'])]
        if reviewed_items.empty:
            recommendation_list.append([])
            continue
        user_vector = reviewed_items.drop(['id'], axis=1).mean()
        nns = nbrs.kneighbors([user_vector.to_numpy()], 10, return_distance=False)[0]
        recommendations = [items.loc[item]['id'] for item in nns]
        for recommendation in recommendations:
            if isinstance(recommendation, list):
                print('BRUH')
                assert False
        recommendation_list.append(recommendations)

    df['recommendations'] = recommendation_list
    return df

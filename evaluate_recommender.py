import pandas as pd
import numpy as np
from recommender import parse_json, ContentBasedRec
import itertools

def generate_gt():
    gt = parse_json("./data/australian_users_items.json")
    gt['items'] = gt['items'].apply(lambda items: [item['item_id'] for item in items])
    gt = gt.drop(['user_url'], axis=1)
    gt.to_parquet('./data/ground_truth.parquet')

def evaluate(recommendations:pd.DataFrame):
    eval = recommendations.drop(recommendations[~recommendations['recommendations'].astype(bool)].index)
    gt = pd.read_parquet('./data/ground_truth.parquet')
    eval = eval.merge(gt, on=['user_id'])

    results_dict = dict()
    # compute recall@k
    eval['recommendations'] = eval['recommendations'].apply(set)
    eval['items'] = eval['items'].apply(set)
    eval.drop(eval[~eval['items'].astype(bool)].index, inplace=True)
    eval['recall@k'] = eval.apply(lambda row: len(row['recommendations'].intersection(row['items']))/len(row['items']), axis=1)
    eval['nDCG@k'] = eval.apply(lambda row: np.sum([(np.power(2, rec in row['items'])-1)/(np.log2(i+1)) for i, rec in enumerate(row['recommendations'])]), axis=1)
    eval['nDCG@k'] = eval.apply(lambda row: row['nDCG@k']/np.sum([1/(np.log2(i+1)) for i in len(row['recommendations'])]), axis=1)
    results_dict['recall@k'] = eval['recall@k'].mean()
    results_dict['recall@k'] = eval['nDCG@k'].mean()
    return results_dict

if __name__ == '__main__':
    metrics = ['cosine']
    tfidf = [None, 'default', 'smooth', 'sublinear', 'smooth_sublinear']
    combinations = list(itertools.product(metrics, tfidf))
    for metric, tfidf in combinations:
        rec = ContentBasedRec("./data/steam_games.json", sparse=False, distance_metric=metric, tfidf=tfidf)
        rec.generate_recommendations("./data/australian_user_reviews.json")
        print(metric, tfidf, evaluate(rec.recommendations))

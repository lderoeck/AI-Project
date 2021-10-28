import pandas as pd
import numpy as np
from recommender import parse_json, ContentBasedRec
import itertools
from multiprocessing import Pool
import os

def generate_gt(target:str):
    gt = parse_json("./data/australian_users_items.json")
    gt['items'] = gt['items'].apply(lambda items: [item['item_id'] for item in items])
    gt = gt.drop(['user_url'], axis=1)
    gt.to_parquet(target)

def evaluate(recommendations:pd.DataFrame):
    eval = recommendations.drop(recommendations[~recommendations['recommendations'].astype(bool)].index)
    gt = pd.read_parquet('./data/ground_truth.parquet')
    eval = eval.merge(gt, on=['user_id'])

    results_dict = dict()
    eval.drop(eval[~eval['items'].astype(bool)].index, inplace=True)
    
    # compute nDCG@k
    eval['nDCG@k'] = eval.apply(lambda row: np.sum([(np.power(2, rec in row['items'])-1)/(np.log2(i+2)) for i, rec in enumerate(row['recommendations'])]), axis=1)
    eval['nDCG@k'] = eval.apply(lambda row: row['nDCG@k']/np.sum([1/(np.log2(i+2)) for i in range(len(row['recommendations']))]), axis=1)
    results_dict['nDCG@k'] = eval['nDCG@k'].mean()
    
    # compute recall@k
    eval['items'] = eval['items'].apply(set)
    eval['recommendations'] = eval['recommendations'].apply(set)
    eval['recall@k'] = eval.apply(lambda row: len(row['recommendations'].intersection(row['items']))/len(row['items']), axis=1)
    results_dict['recall@k'] = eval['recall@k'].mean()
    
    return results_dict

def evaluate_recommender(metric, tfidf):
    rec = ContentBasedRec("./data/steam_games.json", sparse=True, distance_metric=metric, tfidf=tfidf)
    rec.generate_recommendations("./data/australian_user_reviews.json")
    return evaluate(metric, tfidf, rec.recommendations)
        

if __name__ == '__main__':
    gt_file = './data/ground_truth.parquet'
    from os.path import exists
    if not exists(gt_file):
        generate_gt(gt_file)
    metrics = ['euclidean', 'cosine']
    tfidf = [None, 'default', 'smooth', 'sublinear', 'smooth_sublinear']
    combinations = list(itertools.product(metrics, tfidf))
    with Pool(min(os.cpu_count(), len(combinations))) as pool:
        results = [pool.apply_async(evaluate_recommender, args=(metric, tfidf)) for metric, tfidf in combinations]
        output = [p.get() for p in results]
    for result in output:
        print(result[0], result[1] + ':', result[2])

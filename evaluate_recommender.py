import pandas as pd
import numpy as np
from recommender import parse_json, ContentBasedRec
import itertools
from multiprocessing import Pool
import os
import ast


def generate_gt(target: str) -> None:
    """Generates ground truth data to target

    Args:
        target (str): Path to file destination
    """
    gt = parse_json("./data/australian_users_items.json")
    gt['items'] = gt['items'].apply(lambda items: [item['item_id'] for item in items])
    gt = gt.drop(['user_url'], axis=1)
    gt.to_parquet(target)


def evaluate(recommendations: pd.DataFrame, filename=None) -> dict:
    """Evaluate the recommendations based on ground truth

    Args:
        recommendations (pd.DataFrame): Dataframe consisting of user_id and their respective recommendations

    Returns:
        dict: a dict containing the recall@k and nDCG@k 
    """
    eval = recommendations.drop(recommendations[~recommendations['recommendations'].astype(bool)].index)  # drop all recommendations that are empty
    gt = pd.read_parquet('./data/ground_truth.parquet')
    eval = eval.merge(gt, on=['user_id'])

    results_dict = dict()
    # drop all rows with no items (nothing to compare against)
    eval.drop(eval[~eval['items'].astype(bool)].index, inplace=True)
    
    if filename:
        if not os.path.exists('./evaluation'):
            os.mkdir('./evaluation/')
        eval.to_csv('./evaluation/' + filename + '.csv')

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


def evaluate_recommender(metric: str, tfidf: str) -> tuple:
    """Wrapper function to allow for easy evaluation of specific metrics and tf-idf methods

    Args:
        metric (str): distance metric to be used
        tfidf (str): tf-idf method to be used

    Returns:
        tuple: (metric, tf-idf, evaluation)
    """
    rec = ContentBasedRec("./data/steam_games.json",sparse=True, distance_metric=metric, tfidf=tfidf)
    rec.generate_recommendations("./data/australian_user_reviews.json")
    return (metric, tfidf, evaluate(rec.recommendations, '%s_%s' % (metric, tfidf)))

def map_id_to_name(mapping, filename):
    recommendations = pd.read_csv('./evaluation/' + filename)
    recommendations = recommendations[['item_id', 'recommendations', 'items']]
    for col in recommendations:
        recommendations[col] = recommendations[col].apply(lambda x: [mapping.get(i, np.nan) for i in ast.literal_eval(x)])
    recommendations.to_csv('./evaluation/parsed_' + filename)


if __name__ == '__main__':
    import glob
    games = parse_json("./data/steam_games.json")
    games = games[['id', 'title']]
    mapping = dict(zip(games.id, games.title))
    for f in glob.glob('./evaluation/*.csv'):
        map_id_to_name(mapping, os.path.basename(f))
    # gt_file = './data/ground_truth.parquet'
    # from os.path import exists
    # if not exists(gt_file):
    #     generate_gt(gt_file)
    # metrics = ['cosine']
    # tfidf = [None]
    # combinations = list(itertools.product(metrics, tfidf))
    # with Pool(min(os.cpu_count(), len(combinations))) as pool:
    #     results = [pool.apply_async(evaluate_recommender, args=(metric, tfidf)) for metric, tfidf in combinations]
    #     output = [p.get() for p in results]
    # for result in output:
    #     print(result[0], result[1], '\b:', result[2])

import pandas as pd
import numpy as np
from recommender import parse_json, ContentBasedRec
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


def evaluate(recommendations: pd.DataFrame, filename=None, qual_eval_folder=None) -> dict:
    """Evaluate the recommendations based on ground truth

    Args:
        recommendations (pd.DataFrame): Dataframe consisting of user_id and their respective recommendations
        filename ([type], optional): filename for qualitative evaluation. Defaults to None.
        qual_eval_folder ([type], optional): output folder for qualitative evaluation. Defaults to None.

    Returns:
        dict: a dict containing the recall@k and nDCG@k
    """
    eval = recommendations.drop(recommendations[~recommendations['recommendations'].astype(bool)].index)  # drop all recommendations that are empty
    gt = pd.read_parquet('./data/ground_truth.parquet')
    eval = eval.merge(gt, on=['user_id'])

    results_dict = dict()
    # drop all rows with no items (nothing to compare against)
    eval.drop(eval[~eval['items'].astype(bool)].index, inplace=True)

    if filename and qual_eval_folder:
        if not os.path.exists(qual_eval_folder):
            os.makedirs(qual_eval_folder)
        eval.to_csv(qual_eval_folder + filename + '.csv')
        
    # Drop reviewed items from ground truth
    eval['items'] = eval.apply(lambda row: list(set(row['items']).difference(set(row['item_id']))), axis=1)
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


def evaluate_recommender(metric: str, tfidf: str, use_feedback=False, qual_eval_folder='./evaluation', read_max=None) -> tuple:
    """Wrapper function to allow for easy evaluation of specific metrics and tf-idf methods

    Args:
        metric (str): distance metric to be used
        tfidf (str): tf-idf method to be used
        qual_eval_folder (str, optional): output folder for qualitative evaluation. Defaults to './evaluation'.

    Returns:
        tuple: (metric, tf-idf, evaluation)
    """
    rec = ContentBasedRec("./data/steam_games.json", sparse=True, distance_metric=metric, tfidf=tfidf, use_feedback=use_feedback)
    rec.generate_recommendations("./data/australian_user_reviews.json", read_max=read_max)
    return (metric, tfidf, use_feedback, evaluate(rec.recommendations, '%s_%s_%s' % (metric, tfidf, use_feedback), qual_eval_folder + '/source/'))


def map_id_to_name(mapping: dict, filename: str) -> None:
    """Replaces ids in input file to application names

    Args:
        mapping (dict): a map from id -> application name
        filename (str): input file name
    """
    recommendations = pd.read_csv(filename)
    recommendations = recommendations[['item_id', 'recommendations', 'items']]
    for col in recommendations:
        recommendations[col] = recommendations[col].apply(
            lambda x: [mapping.get(i, 'unknown game') for i in ast.literal_eval(x)])
    recommendations.to_csv(filename)


if __name__ == '__main__':
    pass

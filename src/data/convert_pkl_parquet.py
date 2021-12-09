from sys import argv
import pandas as pd 
from ast import literal_eval
from glob import glob
import numpy as np

def unpack_split(df, col):
    return pd.DataFrame({
        'item_id': df[col].apply(lambda x: x[0]),
        'playtime_forever': df[col].apply(lambda x: x[1]),
        'playtime_2weeks': df[col].apply(lambda x: x[2])})
    
def load_interactions(f, n_splits=5):
    df = pd.read_pickle(f)
    df = df.applymap(lambda x: np.array(x, dtype=np.int32))
    interactions_dict = {}
    for split in range(n_splits):
        for column in ['train', 'val', 'test']:
            interactions_dict[split, column] = pd.DataFrame({
                'item_id': df[column].apply(lambda x: x[split, 0]),
                'playtime_forever': df[column].apply(lambda x: x[split, 1]),
                'playtime_2weeks': df[column].apply(lambda x: x[split, 2])})
    return interactions_dict

def main(f: str, n_splits=5) -> None:
    interactions = load_interactions(f, n_splits)
    for i in range(n_splits):
        for part in ['train', 'val', 'test']:
            part_file = interactions[i, part].applymap(lambda x: x.tolist())
            print(f"{f} -> {f[:-4]}_{i}_{part}.parquet")
            part_file.to_parquet(f"{f[:-4]}_{i}_{part}.parquet")


if __name__ == "__main__":
    main(argv[1])

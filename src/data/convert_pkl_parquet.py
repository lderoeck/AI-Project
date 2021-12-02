from sys import argv
import pandas as pd 
from ast import literal_eval
from glob import glob

def unpack_split(df, col):
    return pd.DataFrame({
        'item_id': df[col].apply(lambda x: x[0]),
        'playtime_forever': df[col].apply(lambda x: x[1]),
        'playtime_2weeks': df[col].apply(lambda x: x[2])})

def main(base_path: str) -> None:
    for f in glob(f'{base_path}/*.pkl'):
        data = pd.read_pickle(f)
        for part in ['train', 'val', 'test']:
            part_file = unpack_split(data, part).applymap(lambda x: x.tolist())
            print(type(part_file.iloc[0]['item_id']))
            print(f"{f} -> {f[:-4]}_{part}.parquet")
            part_file.to_parquet(f"{f[:-4]}_{part}.parquet")


if __name__ == "__main__":
    main(argv[1])

from sys import argv
import pandas as pd 
from ast import literal_eval
from glob import glob

def main(base_path: str) -> None:
    for f in glob(f'{base_path}/*.csv'):
        data = pd.read_csv(f, index_col=0)
        for col in ["item_id", "playtime_forever", "playtime_2weeks"]:
            data[col] = data[col].apply(literal_eval)

        print(f"{f} -> {f[:-3]}parquet")
        data.to_parquet(f"{f[:-3]}parquet")


if __name__ == "__main__":
    main(argv[1])
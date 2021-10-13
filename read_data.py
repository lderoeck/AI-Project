
import pandas as pd
from pandas.core.frame import DataFrame
from tqdm import tqdm
import ast
import numpy as np


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

df = parse_json('./data/australian_user_reviews.json')
df = df.explode('reviews', ignore_index=True)


df = pd.concat([df.drop(['reviews', 'user_url'], axis=1), pd.json_normalize(df.reviews)], axis=1).drop(['funny', 'helpful', 'posted', 'last_edited', 'review'], axis=1)
print(df)


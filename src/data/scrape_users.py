import urllib
import xmltodict
import pandas as pd
import numpy as np
from random import sample
from sklearn.model_selection import train_test_split

def split_data(data, seed):
    train, test = train_test_split(data, test_size=0.3, random_state=seed)
    if len(test) < 2:
        train, val = train_test_split(train, test_size=0.5, random_state=seed+1)
    else:
        test, val = train_test_split(test, test_size=0.5, random_state=seed+1)
    return [train, test, val]

def get_user_info(user):
    file = urllib.request.urlopen(f'https://steamcommunity.com/id/{user}/games?tab=all&xml=1')
    data = file.read()
    file.close()

    return xmltodict.parse(data)

if __name__ == '__main__':
    users = ['epiquesam', 'Vyolex']
    cols = ['item_id', 'playtime_forever', 'playtime_2weeks']
    train = pd.DataFrame(columns=cols)
    test = pd.DataFrame(columns=cols)
    val = pd.DataFrame(columns=cols)
    for user in users:
        user_info = get_user_info(user)
        item_id = []
        playtime_forever = []
        playtime_2weeks = []
        data = {'item_id': item_id, 'playtime_forever': playtime_forever, 'playtime_2weeks': playtime_2weeks}
        for game in user_info['gamesList']['games']['game']:
            item_id.append(game['appID'])
            if 'hoursOnRecord' in game:
                playtime_forever.append(game['hoursOnRecord'])
            else:
                playtime_forever.append(0)
            if 'hoursLast2Weeks' in game:
                playtime_2weeks.append(game['hoursLast2Weeks'])
            else:
                playtime_2weeks.append(0)
        l = 100 #length of data 
        f = 50  #number of elements you need
        
        splits_data = [dict() for _ in range(3)]
        for key in data:
            splits = split_data(data[key], 42)
            for i, split in enumerate(splits):
                splits_data[i][key] = split
        
        train = train.append(splits_data[0], ignore_index=True)
        test = test.append(splits_data[1], ignore_index=True)
        val = val.append(splits_data[2], ignore_index=True)
    train.to_parquet('./data/train_users.parquet')
    test.to_parquet('./data/test_users.parquet')
    val.to_parquet('./data/val_users.parquet')

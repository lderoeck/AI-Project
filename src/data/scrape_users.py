import urllib
import xmltodict
import pandas as pd
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
    games = pd.read_pickle('./data/all_games.pkl.xz')
    users = ['epiquesam', 'Vyolex']
    cols = ['item_id', 'playtime_forever', 'playtime_2weeks']
    all = pd.DataFrame(columns=cols)
    train = pd.DataFrame(columns=cols)
    test = pd.DataFrame(columns=cols)
    val = pd.DataFrame(columns=cols)
    for user in users:
        user_info = get_user_info(user)
        item_id = []
        playtime_forever = []
        playtime_2weeks = []
        for game in user_info['gamesList']['games']['game']:
            id = game['appID']
            new_id = games.index[games['id'] == id].tolist()
            if len(new_id) > 0:
                item_id.append(new_id[0])
            else:
                print(game)
                continue
            if 'hoursOnRecord' in game:
                playtime_forever.append(float(game['hoursOnRecord'].replace(',', '')))
            else:
                playtime_forever.append(0)
            if 'hoursLast2Weeks' in game:
                playtime_2weeks.append(float(game['hoursLast2Weeks'].replace(',', '')))
            else:
                playtime_2weeks.append(0)
        
        data = {'item_id': item_id, 'playtime_forever': playtime_forever, 'playtime_2weeks': playtime_2weeks}
        splits_data = [dict() for _ in range(3)]
        for key in data:
            splits = split_data(data[key], 42)
            for i, split in enumerate(splits):
                splits_data[i][key] = split
        
        all = all.append(data, ignore_index=True)
        train = train.append(splits_data[0], ignore_index=True)
        test = test.append(splits_data[1], ignore_index=True)
        val = val.append(splits_data[2], ignore_index=True)
    
    all.to_parquet('./data/train_all_users.parquet')
    train.to_parquet('./data/train_users.parquet')
    test.to_parquet('./data/test_users.parquet')
    val.to_parquet('./data/val_users.parquet')

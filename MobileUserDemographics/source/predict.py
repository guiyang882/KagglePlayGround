import datetime
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import random
import zipfile
import time
import shutil
from sklearn.metrics import log_loss

random.seed(2016)

def run_xgb(train, test, features, target, random_state=0):
    eta = 0.1
    max_depth = 5
    subsample = 0.7
    colsample_bytree = 0.7
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "multi:softprob",
        "num_class": 12,
        "booster" : "gbtree",
        "eval_metric": "mlogloss",
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state,
    }
    num_boost_round = 500
    early_stopping_rounds = 50
    test_size = 0.3

    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    print('Length train:', len(X_train.index))
    print('Length valid:', len(X_valid.index))
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration)
    score = log_loss(y_valid.tolist(), check)

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_iteration)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction.tolist(), score


def create_submission(score, test, prediction):
    # Make Submission
    now = datetime.datetime.now()
    sub_file = '../submission/submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('device_id,F23-,F24-26,F27-28,F29-32,F33-42,F43+,M22-,M23-26,M27-28,M29-31,M32-38,M39+\n')
    total = 0
    test_val = test['device_id'].values
    for i in range(len(test_val)):
        str1 = str(test_val[i])
        for j in range(12):
            str1 += ',' + str(prediction[i][j])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()

def map_column(table, f):
    labels = sorted(table[f].unique())
    mappings = dict()
    for i in range(len(labels)):
        mappings[labels[i]] = i
    table = table.replace({f: mappings})
    return table

def read_train_test():
    import datetime
    def rules_for_timestamp(x):
        weekday_data, hour_data = [], []
        for ind in x.index:
            str_tmp = x.loc[ind]
            tmp = datetime.datetime.strptime(str(str_tmp), "%Y-%m-%d %H:%M:%S")
            weekday_data.append(tmp.weekday())
            hour_data.append(tmp.hour)
        return weekday_data, hour_data

    def rules_for_geo(geo_info):
        geo_index = []
        for i in geo_info.index:
            t_key = (int(geo_info.loc[i]["longitude"]), int(geo_info.loc[i]["latitude"]))
            info = 0;
            if geoMap.has_key(t_key):
                info = geoMap[t_key]
            else:
                info = 0
            geo_index.append(info)
        return geo_index

    # Events
    print('Read events...')
    events = pd.read_csv("../data/events.csv", dtype={'device_id': np.str})
    events['counts'] = events.groupby(['device_id'])['event_id'].transform('count')
    events.drop_duplicates('device_id', keep='first', inplace=True)
    print events.head()

    print("Deal with timestamp...")
    weekday_data, hour_data = rules_for_timestamp(events['timestamp'])
    del events["timestamp"]
    events["weekday"] = pd.Series(data=weekday_data, index=events.index)
    events["hour"] = pd.Series(data=hour_data, index=events.index)

    print("Deal with geo info...")
    pd_geoClass = pd.read_csv("../data/geo_class_features.csv")
    geoMap = {}
    for ind in range(len(pd_geoClass)):
        longtitude = pd_geoClass["longtitude"][ind]
        latitude = pd_geoClass["latitude"][ind]
        class_index = pd_geoClass["class"][ind]
        geoMap[(longtitude, latitude)] = class_index

    tmp = events[["longitude", "latitude"]]
    geo_class = rules_for_geo(tmp)
    del events["longitude"]
    del events["latitude"]
    events["geo"] = pd.Series(data=geo_class, index=events.index)

    # Phone brand
    print('Read brands...')
    pbd = pd.read_csv("../data/phone_brand_device_model.csv", dtype={'device_id': np.str})
    pbd.drop_duplicates('device_id', keep='first', inplace=True)
    pbd = map_column(pbd, 'phone_brand')
    pbd = map_column(pbd, 'device_model')

    # Read table prepare app
    print("Deal with app event")
    app_event_label = pd.read_csv("../data/table_prepare_app.csv", dtype={"event_id": np.int64})
    print type(app_event_label["event_id"][0])
    print type(events["event_id"][0])
    events = pd.merge(events, app_event_label, how="left", on="event_id", left_index=True)

    # Train
    print('Read train...')
    train = pd.read_csv("../data/gender_age_train.csv", dtype={'device_id': np.str})
    train = map_column(train, 'group')
    train = train.drop(['age'], axis=1)
    train = train.drop(['gender'], axis=1)
    train = pd.merge(train, pbd, how='left', on='device_id', left_index=True)
    train = pd.merge(train, events, how='left', on='device_id', left_index=True)
    train.fillna(-1, inplace=True)

    # Test
    print('Read test...')
    test = pd.read_csv("../data/gender_age_test.csv", dtype={'device_id': np.str})
    test = pd.merge(test, pbd, how='left', on='device_id', left_index=True)
    test = pd.merge(test, events, how='left', on='device_id', left_index=True)
    test.fillna(-1, inplace=True)

    train.to_csv("train_sample.csv", index=False)
    test.to_csv("test_sample.csv", index=False)

def prepare_Sample():
    train = pd.read_csv("./train_sample.csv", dtype={"device_id": np.str})
    test = pd.read_csv("./test_sample.csv", dtype={"device_id": np.str})

    # Features
    features = list(test.columns.values)
    features.remove('device_id')

    return train, test, features

train, test, features = prepare_Sample()
print('Length of train: ', len(train))
print('Length of test: ', len(test))
print('Features [{}]: {}'.format(len(features), sorted(features)))
test_prediction, score = run_xgb(train, test, features, 'group')
print("LS: {}".format(round(score, 5)))
create_submission(score, test, test_prediction)
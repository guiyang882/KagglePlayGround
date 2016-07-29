import datetime
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import random
import time
from sklearn.metrics import log_loss

random.seed(114)

def run_xgb(train, test, features, target, random_state=0):
    eta = 0.2
    max_depth = 5
    subsample = 0.7
    colsample_bytree = 0.7
    start_time = time.time()

    # print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    # params = {
    #     "objective": "multi:softprob",
    #     "num_class": 12,
    #     "booster" : "gbtree",
    #     "eval_metric": "mlogloss",
    #     "eta": eta,
    #     "max_depth": max_depth,
    #     "subsample": subsample,
    #     "colsample_bytree": colsample_bytree,
    #     "silent": 1,
    #     "seed": random_state,
    # }

    params = {
        "objective": "multi:softprob",
        "num_class": 12,
        "eta": 0.01,
        "lambda": 5,
        "lambda_bias": 0,
        "alpha": 2,
        "booster": "gblinear",
        "eval_metric": "mlogloss",
        "silent": 1,
        "seed": random_state,
    }

    num_boost_round = 280
    early_stopping_rounds = 100
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
    # check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration)
    check = gbm.predict(xgb.DMatrix(X_valid[features]))
    score = log_loss(y_valid.tolist(), check)

    print("Predict test set...")
    # test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_iteration)
    test_prediction = gbm.predict(xgb.DMatrix(test[features]))

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

    # Events
    print('Read events...')
    events = pd.read_csv("../data/events.csv", dtype={'device_id': np.str})
    events['counts'] = events.groupby(['device_id'])['event_id'].transform('count')

    # Read table prepare app
    print("Deal with app event")
    app_event_label = pd.read_csv("../data/table_prepare_app.csv", dtype={"event_id": np.int64})
    events = pd.merge(events, app_event_label, how="left", on="event_id", left_index=True)
    del events["event_id"]
    events.sort_values(by=["device_id", "timestamp"], inplace=True)
    events["timestamp"] = events["timestamp"].apply(lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S").time().hour * 1.0 + datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S").time().minute / 60.0)
    del events["longitude"]
    del events["latitude"]
    events_counts = events[["device_id", "counts"]].drop_duplicates(subset={"device_id"})
    events_counts.set_index(["device_id"], inplace=True)
    del events["counts"]
    events = events.groupby(["device_id"]).sum()
    events["counts"] = pd.Series(data=events_counts["counts"].values, index=events_counts.index)
    events.index.set_names("id", inplace=True)
    events["device_id"] = pd.Series(data=events.index.tolist(), index=events.index)
    print events.head()

    # print("Deal with timestamp...")
    # weekday_data, hour_data = rules_for_timestamp(events['timestamp'])
    # del events["timestamp"]
    # events["weekday"] = pd.Series(data=weekday_data, index=events.index)
    # events["hour"] = pd.Series(data=hour_data, index=events.index)

    # Phone brand
    print('Read brands...')
    pbd = pd.read_csv("../data/phone_brand_device_model.csv", dtype={'device_id': np.str})
    pbd.drop_duplicates('device_id', keep='first', inplace=True)
    pbd = map_column(pbd, 'phone_brand')
    pbd = map_column(pbd, 'device_model')

    # Train
    print('Read train...')
    train = pd.read_csv("../data/gender_age_train.csv", dtype={'device_id': np.str})
    train = map_column(train, 'group')
    print("deal age and gender")
    # train["age"] = train["age"].apply(lambda x: int(x)/5)
    # train["gender"] = train["gender"].apply(lambda x: 0 if x == "M" else 1)
    del train["age"]
    del train["gender"]
    print("start merge")
    train = pd.merge(train, pbd, how='left', on='device_id', left_index=True)
    train = pd.merge(train, events, how='left', on='device_id', left_index=True)

    group_columns = ["Custom", "Education", "Family",
                     "Finance", "Fun", "Industry tag",
                     "Music", "Other", "Productivity",
                     "Property", "Religion", "Services",
                     "Video", "Games", "Shopping", "Tencent", "Travel", "Vitality", "Sports"]

    train.fillna(-1, inplace=True)

    # Test
    print('Read test...')
    test = pd.read_csv("../data/gender_age_test.csv", dtype={'device_id': np.str})
    test = pd.merge(test, pbd, how='left', on='device_id', left_index=True)
    test = pd.merge(test, events, how='left', on='device_id', left_index=True)
    test.fillna(-1, inplace=True)

    train.to_csv("train_sample.csv", index=False)
    test.to_csv("test_sample.csv", index=False)

def prepare_Sample_DropNA():
    train = pd.read_csv("train_sample.csv", dtype={"device_id": np.str})
    test = pd.read_csv("test_sample.csv", dtype={"device_id": np.str})
    # print train.head()
    # Features
    group_columns = ["Custom", "Education", "Family",
                     "Finance", "Fun", "Industry tag",
                     "Music", "Other", "Productivity",
                     "Property", "Religion", "Services",
                     "Video", "Games", "Shopping", "Tencent", "Travel", "Vitality", "Sports"]
    remove_columns = ["Other", "Vitality", "Sports", "Travel"]
    features = list(test.columns.values)
    features.remove('device_id')
    # for item in group_columns:
    #     features.remove(item)
    print features
    return train, test, features

# read_train_test()

train, test, features = prepare_Sample_DropNA()
print('Length of train: ', len(train))
print('Length of test: ', len(test))
print('Features [{}]: {}'.format(len(features), sorted(features)))
test_prediction, score = run_xgb(train, test, features, 'group')
print("LS: {}".format(round(score, 5)))
create_submission(score, test, test_prediction)
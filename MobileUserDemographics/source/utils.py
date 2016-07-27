import pandas as pd
import numpy as np
from datetime import datetime
import rules_for_categorize_label as rules

def load_data_pd(filename):
    pd_data = pd.read_csv(filename)
    print pd_data.head()

def map_column(table, f):
    labels = sorted(table[f].unique())
    mappings = dict()
    for i in range(len(labels)):
        mappings[labels[i]] = i
    table = table.replace({f: mappings})
    return table

def calc_dist_longitude2latitude(longitudeA, latitudeA, longitudeB, latitudeB):
    R = 6371.004 * 10
    alpha = np.sin(latitudeA) * np.sin(latitudeB) * np.cos(longitudeA-longitudeB) + np.cos(longitudeA) * np.cos(longitudeB)
    dist = R * np.arccos(alpha) * np.pi/180
    return dist

def deal_timestamp(str_timestamp):
    tmp = datetime.strptime(str_timestamp, "%Y-%m-%d %H:%M:%S")
    return tmp.hour + tmp.minute / 60.0

def prepare_categoring_labels():
    print("Read app_labels and label_categories")
    labels = pd.read_csv("../data/label_categories.csv", dtype={'app_id':np.str})
    app_labels = pd.read_csv("../data/app_labels.csv")
    apps = pd.merge(app_labels, labels, how='left', on='label_id')
    apps['general_groups'] = apps['category']
    apps['general_groups'] = apps['general_groups'].apply(rules.to_Games)
    apps['general_groups'] = apps['general_groups'].apply(rules.to_Property)
    apps['general_groups'] = apps['general_groups'].apply(rules.to_Family)
    apps['general_groups'] = apps['general_groups'].apply(rules.to_Fun)
    apps['general_groups'] = apps['general_groups'].apply(rules.to_Productivity)
    apps['general_groups'] = apps['general_groups'].apply(rules.to_Finance)
    apps['general_groups'] = apps['general_groups'].apply(rules.to_Religion)
    apps['general_groups'] = apps['general_groups'].apply(rules.to_Services)
    apps['general_groups'] = apps['general_groups'].apply(rules.to_Travel)
    apps['general_groups'] = apps['general_groups'].apply(rules.to_Custom)
    apps['general_groups'] = apps['general_groups'].apply(rules.to_Video)
    apps['general_groups'] = apps['general_groups'].apply(rules.to_Shopping)
    apps['general_groups'] = apps['general_groups'].apply(rules.to_Education)
    apps['general_groups'] = apps['general_groups'].apply(rules.to_Vitality)
    apps['general_groups'] = apps['general_groups'].apply(rules.to_Sports)
    apps['general_groups'] = apps['general_groups'].apply(rules.to_Music)
    apps['general_groups'] = apps['general_groups'].apply(rules.to_Other)
    del apps['category']
    app_events = pd.read_csv("../data/app_events.csv", dtype={'app_id': np.int64})
    app_events = pd.merge(app_events, apps, how='left', on='app_id', left_index=True)
    app_events.dropna(inplace=True)
    app_events.to_csv("../data/prepare_app_events.csv", header=True,index=False,encoding='utf-8')
    return app_events

def prepare_train_data():
    print('Read events...')
    events = pd.read_csv("../data/events.csv", dtype={'device_id': np.str})
    pbd = pd.read_csv("../data/phone_brand_device_model.csv", dtype={'device_id': np.str})
    pbd.drop_duplicates('device_id', keep='first', inplace=True)
    events = pd.merge(events, pbd, how='left', on='device_id')
    train = pd.read_csv("../data/gender_age_train.csv", dtype={'device_id': np.str})
    train = pd.merge(train, events, how='left', on='device_id', left_index=True)
    train.dropna(inplace=True)
    train.to_csv("../data/prepare_train.csv", header=True, index=False, encoding='utf-8')
    print train.head()

def prepare_geoposition():
    print("Read Geo Classify...")
    geoClass = pd.read_csv("../data/geo_class_features.csv")
    geoMap = {}
    for ind in range(len(geoClass)):
        longtitude = geoClass["longtitude"][ind]
        latitude = geoClass["latitude"][ind]
        class_index = geoClass["class"][ind]
        geoMap[(longtitude, latitude)] = class_index

if __name__ == "__main__":
    prepare_geoposition()
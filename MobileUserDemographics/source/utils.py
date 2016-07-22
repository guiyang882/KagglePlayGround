import pandas as pd
import rules_for_categorize_label as rules

def load_data_pd(filename):
    pd_data = pd.read_csv(filename)
    print pd_data.head()

def categoring_labels():
    labels = pd.read_csv("../data/label_categories.csv")
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
    print apps['general_groups'].value_counts()

if __name__ == "__main__":
    categoring_labels()
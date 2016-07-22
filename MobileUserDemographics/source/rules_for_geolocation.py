import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns

df_events = pd.read_csv("../data/events.csv", dtype={'device_id': np.str, 'timestamp': datetime})
print df_events.head()
print df_events.dtypes
df_events_sample = df_events.sample(n=100000)

df_at0 = df_events[(df_events["longitude"]==0) & (df_events["latitude"]==0)]
df_near0 = df_events[(df_events["longitude"]>-1) &\
                     (df_events["longitude"]<1) &\
                     (df_events["latitude"]>-1) &\
                     (df_events["latitude"]<1)]

print("# events:", len(df_events))
print("# at (0,0)", len(df_at0))
print("# near (0,0)", len(df_near0))

# Sample it down to only the China region
lon_min, lon_max = 75, 135
lat_min, lat_max = 15, 55

idx_china = (df_events["longitude"] > lon_min) & \
            (df_events["longitude"] < lon_max) & \
            (df_events["latitude"] > lat_min) & \
            (df_events["latitude"] < lat_max)

df_events_china = df_events[idx_china].sample(n=100000)

# Sample it down to only the Beijing region
lon_min, lon_max = 116, 117
lat_min, lat_max = 39.75, 40.25

idx_beijing = (df_events["longitude"] > lon_min) & \
              (df_events["longitude"] < lon_max) & \
              (df_events["latitude"] > lat_min) & \
              (df_events["latitude"] < lat_max)

df_events_beijing = df_events[idx_beijing]
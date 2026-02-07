import pandas as pd

"""
Features: 
- heartrate
- acc_x, acc_y, acc_z
- mean heartrate by test subject (8 hour window containing awake and sleep heartrates)
- take absolute value, then mean acceleration over the same window

Label:
- is_sleep (True or False)
"""

# Import reusable classifier
from dtsc330_26 import reusable_classifier
reusable_classifier.ReusableClassifier()

# Import reader to load in data
from dtsc330_26.readers import har
har.HAR('data/motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0')

# Assign a variable to the data we just read in
data = har.HAR('data/motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0')

# convert data into a usable dataframe
df = data.df

# Only include every 50th row
# New dataframe includes roughly only one row per second
df = df.iloc[::50]

# Take absolute value of the acceleration to avoid issues from averaging positive and negative values
df['acc_x'] = df['acc_x'].abs()
df['acc_y'] = df['acc_y'].abs()
df['acc_z'] = df['acc_z'].abs()

# sets window size equal to the nubmer of seconds for the first person's experiment
window_size = len(df[df['person'] == 0.0]) 

# Create new features:
# - Mean heartrate per experiment subject's 8 hour window
# - Mean accelerations in x,y,z directions over the same window
df['rolling_hr'] = df['hr'].rolling(window=window_size, min_periods=1).mean()
df['rolling_acc_x'] = df['acc_x'].rolling(window=window_size, min_periods=1).mean()
df['rolling_acc_y'] = df['acc_y'].rolling(window=window_size, min_periods=1).mean()
df['rolling_acc_z'] = df['acc_z'].rolling(window=window_size, min_periods=1).mean()

# train test split
labels = df['is_sleep']
features = df.drop(columns=['timestamp', 'person', 'is_sleep'])

# assess classifier performance
rc = reusable_classifier.ReusableClassifier(model_type='logistic_regression')
print(rc.assess(features, labels))

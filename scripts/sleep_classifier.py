import pandas as pd
import random

"""
Features: 
- heartrate
- absolute value of acc_x, acc_y, acc_z
- mean heartrate by test subject (8 hour window containing awake and sleep heartrates)
- mean of absolute value of acceleration over the same window

Label:
- is_sleep (True or False)
"""

# Import reusable classifier
from dtsc330_26 import reusable_classifier
reusable_classifier.ReusableClassifier()

# Import reader to load in data
from dtsc330_26.readers import har

# Assign a variable to the data we just read in
data = har.HAR('data/motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0', 10)

# convert data into a usable dataframe
full_df = data.df
full_df.index = pd.to_timedelta(full_df["timestamp"], unit="s")

people = pd.unique(full_df["person"])

features, labels, test_features, test_labels = [], [], [], []

for person in people:
    print(f'Computing person {person + 1}')

    df = full_df.loc[full_df["person"] == person]

    df['acc_x'] = df['acc_x'].abs()
    df['acc_y'] = df['acc_y'].abs()
    df['acc_z'] = df['acc_z'].abs()

    for window in ["10s", "1min", "10min", "1h", "6h"]:
        for column in ["hr", "acc_x", "acc_y", "acc_z"]:
            df[f"rolling_{column}_{window}"] = df[column].rolling(window).mean()
    
    df = df.resample("10s").first().dropna(how="any")

    # Extract features, labels, and classify
    fs = df.drop(columns=["timestamp", "person", "is_sleep"])
    ls = df["is_sleep"]


    if person < 1:
        test_features.append(fs)
        test_labels.append(ls)
    else:
        features.append(fs)
        labels.append(ls)

rc = reusable_classifier.ReusableClassifier("xgboost")
rc.train(pd.concat(features), pd.concat(labels))

pred_labels = rc.predict(pd.concat(test_features))
test_labels = pd.concat(test_labels)

count_equal = (pred_labels.astype(int) == test_labels.to_numpy().astype(int)).sum()
print(count_equal / len(test_labels))

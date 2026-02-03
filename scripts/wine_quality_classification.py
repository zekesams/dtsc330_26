import pandas as pd
import zipfile

# BOTH of these will work, but they affect the downstream
#import dtsc330_26.reusable_classifier
#dtsc330_26.reusable_classifier.ReusableClassifier()

from dtsc330_26 import classifier
classifier.ReusableClassifier()

#from dtsc330_26.reusable_classifier import ReusableClassifier
# NO
#from dtsc330_26.reusable_classifier import *
# You're fired.

# In a script, no name = main if you don't want
# It can be ugly
# don't add good notes
# you're good

# you will regret when you don't add good notes

zf = zipfile.ZipFile('data/wine_quality.zip')
df = pd.read_csv(zf.open('WineQT.csv'), sep=';')
print(df)

# train test split


rc = classifier.ReusableClassifier()
rc.train()
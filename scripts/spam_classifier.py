import pandas as pd
import fasttext as ft
import re
from sklearn.model_selection import train_test_split

from dtsc330_26.classifiers import reusable_classifier

# read in data
df = pd.read_csv('data/spam_email_dataset.csv')

### Used Claude to help generate some features
def caps_ratio(text):
    """
    This function gets the percentage of capital letters in the email text

    Args: 'text': will be the full email text
    """
    letters = [c for c in text if c.isalpha()]
    if len(letters) == 0:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)

# Apply to dataframe
df['caps_ratio'] = df['email_text'].apply(caps_ratio)

def url_count(text):
    """
    Gets the number of individual urls in an email

    Args: 'text': will be the full email text
    """
    pattern = r'https?://\S+|www\.\S+'
    return len(re.findall(pattern, text))

# Apply to dataframe
df['url_count'] = df['email_text'].apply(url_count)

# get features and labels
features = df.drop(columns=['email_id', 'subject', 'email_text', 'sender_email', 'sender_domain','label'])
labels = df['label']

# train-test split
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=1
)

rc = reusable_classifier.ReusableClassifier("random_forest")
rc.train(train_features, train_labels)

pred_labels = rc.predict(test_features)
test_labels = test_labels

count_equal = (pred_labels == test_labels).sum()
print(count_equal / len(test_labels))

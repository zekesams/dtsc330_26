from dtsc330_26 import merged_data, entity_resolution_features
from dtsc330_26.classifiers import entity_resolution_classifier

def train():
    # add training data
    erc = entity_resolution_classifier.EntityResolutionClassifier()
    erc.train(features,labels)
    
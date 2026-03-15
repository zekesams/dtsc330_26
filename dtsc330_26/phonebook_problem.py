import pandas as pd
import fasttext as ft

from dtsc330_26.readers import grants, articles


class Phonebook():
    """
    Authors - forename, surname, initials, affiliation(s)
    Grantees - forename, surname, initials, affiliation(s)

    """
    def __init__(self, path:str):
        pass

    def retreive_articles(self):
        pass

    def retreive_grants(self):
        pass

    def prematch(self):
        pass

    def combine(self):
        for i,row in self.art_df.iterrows():
            for j,row in self.grant_df.iterrows():
                pass


if __name__ == "__main__":
    phonebook = Phonebook()
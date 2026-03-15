import pandas as pd
from dtsc330_26.readers import grants, articles

class MergedData():
    def get_merged_data(grant_path:str,article_path:str) -> pd.DataFrame:
        
        art = articles.Articles(article_path)
        auth_df = art.get_authors()

        self.art_df = self.retreive_articles(article_path)
        self.grant_df = self.retreive_grants(grant_path)


    if __name__ == '__main__':
        get_merged_data()

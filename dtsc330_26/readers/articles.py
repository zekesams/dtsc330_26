import gzip
import pandas as pd
import xml.etree.ElementTree as ET
import sqlalchemy


class Articles():
    def __init__(self, path: str):
        """Read in a pubmed articles XML file that is gzipped

        Args:
            path (str): location of GZIPped file on the disk
        """
        self.article_df = None
        self.author_df = None
        self._parse(path) 

    def _parse(self, path: str):
        """Parse the Pubmed file"""
        articles = []
        authors = []
        # One trick for creating a dataframe is to create a list of
        # dicts with the same naming format
        with gzip.open(path, 'rb') as fp:
            # _ means throw it away
            # for test in ET.iterparse(fp, events=('end',)):
            # test = (index, value itself)

            for _, article in ET.iterparse(fp, events=('end',)):
                if article.tag == 'PubmedArticle':
                    article_row, article_authors = self._parse_article(article)
                    articles.append(article_row)
                    authors.extend(article_authors)  # be careful extend vs append
                    # append: [[auth1, auth2], [auth3, auth4, auth5]]
                    # extend: [auth1, auth2, ..., auth5]
                    article.clear()

        self.article_df = pd.DataFrame(articles)
        self.author_df = pd.DataFrame(authors)

    def _parse_article(self, article: ET.Element):
        """Parse an XML PubmedArticle element"""
        row = {}
        tags = [
            'PMID',
            'ArticleTitle',
            'PubDate',
            'DateCompleted',
            'Affiliation',
            'Year',
            'Month',
            'Day'
        ]
        # <PubDate>
        #   <Year>2001</Year>
        #   <Month>2</Month>
        #   <Day>28</Day>
        #   <Hour>10</Hour>
        #   <Minute>0</Minute>
        # </PubDate>
        for el in article.iter():
            if el.tag in tags:
                if el.tag.find("Date") > -1:
                    for el2 in el.iter():
                        if el2.tag in tags:
                            row[el2.tag] = el2.text
                row[el.tag] = el.text
                # <LastName>Bettcher</LastName>
                # el.text pulls out what's INSIDE the pair of tags
                # el.tag is the name of the tag
                # remember that a tag is combination of a start and end
                # <el.tag></el.tag>

            ### My old HW and other code to reference later
            # HW:
            # if el.tag == 'PubDate':
                # for x in el:
                  #  if x.tag == 'Year':
                    #    year = x.text
                   # elif x.tag == 'Month':
                    #    month = x.text
                   # elif x.tag == 'Day':
                     #   day = x.text

                # row['PubDate'] = f'{year}-{month}-{day}'

            # if el.tag == 'DateCompleted':
                #for y in el:
                   # if y.tag == 'Year':
                    #    year = y.text
                    #elif y.tag == 'Month':
                     #   month = y.text
                    #elif y.tag == 'Day':
                     #   day = y.text

                # row['DateCompleted'] = f'{year}-{month}-{day}'
            
        if 'PMID' not in row.keys():
            return {}, {}
        
        # In XML, strictly speaking, there's no rule about order
        # <AuthorList></AuthorList><PMID></PMID>
        # <PMID></PMID><AuthorList></AuthorList>
        authors = []
        tags = ['LastName', 'ForeName', 'Initials', 'Affiliation']
        for author in article.findall('.//Author'):
            auth_row = {'PMID': row['PMID']}
            for el in author.iter():
                if el.tag in tags:
                    auth_row[el.tag] = el.text.lower().strip()
            authors.append(auth_row)
        
        return row, authors

    def to_db(self,path:str='data/article_grant_db.sqlite'):
        """Send the read-in data to the database"""
        #Define the connection
        engine = sqlalchemy.create_engine('sqlite:///data/article_grant_db.sqlite')
        connection = engine.connect()

        self.df[['PMID',
                'PubDate',
                'DateCompleted']].to_sql('articles', 
                                    connection,
                                    if_exists='append',
                                    index=False)

    def _from_db(self):
        """Load the data from the database"""
        engine = sqlalchemy.create_engine("sqlite:///data/article_grant_db.sqlite")
        connection = engine.connect()
        df = pd.read_sql("SELECT * FROM articles", connection)
        return df
    
    def get_authors(self):
        """Get parsed grantees"""
        return self.author_df.rename(
            columns={
                'LastName':'surname',
                'ForeName':'forename',
                'Initials':'initials',
                'Affiliation':'affiliation'
            }
        )    
    
    def get_entries(self):
        """Get parsed articles"""
        return self.article_df
        
    
if __name__ == '__main__':
    articles = Articles('data/pubmed25n1275.xml.gz')

# CamelCase
# snake_case -- this is what python programmers use
import pandas as pd


class Grants():  # class names in python are camel case (e.g. GrantReader)
    def __init__(self, path: str):
        """Create and parse a Grants file

        Args:
            path (str): the location of the file on the disk
        """
        # What is self?
        # "Self is the specific instance of the object" - Computer Scientist 
        # Store shared variables in self
        self.path = path
        self.df = self._parse(path)

    def _parse(self, path: str):
        """Parse a grants file"""
        df = pd.read_csv(path, compression='zip')
        
        mapper = {
            'APPLICATION_ID': 'application_id',  # _id means an id
            'BUDGET_START': 'start_at', #  _at means a date
            'ACTIVITY': 'grant_type',
            'TOTAL_COST': 'total_cost',
            'PI_NAMEs': 'pi_names',  # you will notice, homework references this
            'ORG_NAME': 'organization',
            'ORG_CITY': 'city',
            'ORG_STATE': 'state',
            'ORG_COUNTRY': 'country', 
        }
        # make column names lowercase
        # maybe combine for budget duration?
        df = df.rename(columns=mapper)[mapper.values()]

        # HW2 Question 1
        # This takes every missing date in the 'start_at' column and fills with the date in the row above
        for i in range(1, len(df)):
            if pd.isna(df.loc[i, 'start_at']):
                df.loc[i, 'start_at'] = df.loc[i-1, 'start_at']

        # HW2 Question 2
        df['grantee'] = df['pi_names'].str.split(';')
        df = df.explode('grantee')
        df['grantee'] = df['grantee'].str.strip()


        return df
    
    def get(self):
        """Get parsed grants"""
        return self.df        


if __name__ == '__main__':
    # This is for debugging
    grants = Grants('data/RePORTER_PRJ_C_FY2025.zip')
    
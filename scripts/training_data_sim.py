import pandas as pd

# create article author names
authors_data = {
    'author_forename':['john','alan','veronica','owen','lisa','joseph','mary','annaliese','matthew','andrew'],
    'author_surname':['smith','jefferson','johnson','white','gray','sams','jones','rodriguez','jackson','gill']
    }

# convert to dataframe format
authors_df = pd.DataFrame(authors_data)

print(authors_df)

# create grantee names
grants_data = {
    'grantee_forename': ['john','jack', 'xavier','alex','lisa', 'joseph', 'naomi', 'anna', 'matt','mark'],
    'grantee_surname': ['smith', 'johnson', 'ramirez','whitaker','wood','grey','morrison','rodriguez','jakson','washington']
}

# convert to dataframe format
grantees_df = pd.DataFrame(grants_data)

# POTENTIAL ERRORS:
# shortened versions of first names -> annaliese/anna, matt/matthew
# different spellings of last names -> gray/grey
# typos -> jackson/jakson
# last name changing after marriage

print(grantees_df)

# merge dataframes together using a cross merge
# a cross merge compares everything in the authors_df to everything in the grantees_df

full_df = pd.merge(authors_df, grantees_df, how='cross')

print(full_df)

# create new columns containing true/false values for if forenames and surnames are the same
full_df['forename_match'] = full_df['author_forename'] == full_df['grantee_forename']
full_df['surname_match'] = full_df['author_surname'] == full_df['grantee_surname']

# create a new column with a true/false value for if both full author names match full grantee names
full_df['match'] = (full_df['forename_match'] == True) & (full_df['surname_match'] == True)

# drop forename and surname match columns
df = full_df.drop(columns=['forename_match','surname_match'])

print(df)

print(df[df['match'] == True])
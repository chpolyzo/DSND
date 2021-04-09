import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_clean(messages_filepath, categories_filepath):
    """
    Input:
    messages_path = the path of the messages.csv file,
    categories_path = the path of the categories.csv file
    
    Output:
    cleaned merged dataframe using both messages and categories data
    
    Description:
    Function to load and clean data in compatibility terms with the ML application
    1. Loads data using messages and categories paths
    3. Drops dupplicates in the messages dataframe
    3. Drops 'original' column in the messages dataframe
    4. Creates a list of column names which we find in the category dataframe, as category column
    5. Loops into all reccords of the categories data frame category column to split the category 
    column and distribute new columns in the category dataframe with the respected labels
    6. Fixes the 'related' column by replacing "2" to 1
    7. Drops 'child_alone' column which has only one value, there fore would bring no value to our model
    8. Drops categories column after having created the columns for each variable found there
    9. Merges two dataframes
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # drop dupplicates in messages dataframe 
    messages.drop_duplicates(inplace = True)
    
    # drop original column in the messages dataframe
    messages.drop(['original'], axis = 1, inplace = True)
    
    # create column names using list comprehensions taken from the categories.categories Series
    cols = [i.split('-')[0] for i in categories.iloc[0, 1:2].str.split(";")[0]]

    # create a loop to create columns with the right data for every category found in the category.category Series
    import time
    start = time.time()

    for col in range(0, len(cols)):
        print(f"Creating column {col + 1}: {cols[col]}")
        categories[cols[col]] = categories.categories.apply(lambda x: x.split(";")[col].split('-')[1]).astype('int64')
    stop = time.time()

    print(f"Calculation time: {round(stop-start, 2)} seconds")

    # fix the related categories binary issue
    categories.loc[(categories['related']==2)] = 1

    # drop the child alone category which is the same and would not add any value to our model
    categories.drop(['child_alone'], axis = 1, inplace = True)

    # drop categories column
    categories.drop(['categories'], axis = 1, inplace = True)

    # drop dupplicates in messages dataframe 
    categories.drop_duplicates(inplace = True)

    # merge datasets
    df = messages.merge(categories, on='id', how = 'inner')
    
    # drop nan values
    df.dropna(inplace = True)
    
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('response_table', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data and cleaning data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_clean(messages_filepath, categories_filepath)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    " Load the data from 2 csv files 
    "
    " Args:
    "   messages_filepath: file path of the csv file containing the messages
    "   categories_filepath: file path of the csv file containing the categories
    "
    " Returns:
    "   a dataframe
    "
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    """
    " clean a dataframe to make it ready for machine learning algorithm
    " 
    " Args:
    "   df: the dataframe to clean
    "
    " Returns:
    "   df: the cleaned dataframe
    "
    """
    
    # split categories into separate columns
    df.iloc[:,4].str.split(";", expand=True)
    df = pd.concat([df,df.iloc[:,4].str.split(";", expand=True)],axis=1)
    
    columns_names = ['id', 'message', 'original', 'genre', 'categories']    
    for i in range(0,36):
        columns_names.append(df.iloc[0,i+5].split("-")[0])
    df.columns = columns_names
    
    for col_name in columns_names:
        if col_name not in ['id', 'message', 'original', 'genre', 'categories']:
            df[col_name] = df[col_name].str.split("-", expand=True)[1]
            df[col_name] = pd.to_numeric(df[col_name])
            
    df = df.drop(['categories'], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """
    " save a dataframe to a database
    " 
    " Args:
    "   df: the dataframe to save
    "   database_filename: the name of the file containing the database
    "  
    " Returns:
    "   nothing
    "
    """
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Disaster_Response', engine, index=False)  


def main():
    """
    " main function. Load data from command line arguments, clean it, 
    " save it into a database
    "
    " Args:
    "   none
    "
    " Returns:
    "   nothing
    "
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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

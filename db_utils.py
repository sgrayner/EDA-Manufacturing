import yaml
import pandas as pd
from sqlalchemy import create_engine


class DatabaseConnector:
    '''
    This class connects to the RDS database, saves the data to the local machine and loads the data.
    '''
    def __init__(self, creds):
        self.creds = creds

    def __read_creds(self):
        '''
        This function is used to read and return credentials of a database.

        Returns:
            data (DataFrame): The credentials for the database.
        '''
        with open(self.creds, 'r') as file:
            data = yaml.safe_load(file)
        return data
    
    def __init_db_engine(self):
        '''
        This function initiates an SQLalchemy database engine.

        Returns:
            engine: An SQLalchemy engine.
        '''
        data = self.__read_creds()
        DATABASE_TYPE = 'postgresql'
        DBAPI = 'psycopg2'
        HOST = data['HOST']
        USER = data['USER']
        PASSWORD = data['PASSWORD']
        DATABASE = data['DATABASE']
        PORT = data['PORT']
        engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")
        return engine
    
    def read_rds_table(self, table):
        '''
        This function reads data from an RDS database.
        
        Parameters:
            table: Name of table to extract data from.
            
        Returns:
            df (DataFrame): the extracted data as a DataFrame.
        '''
        df = pd.read_sql_table(table, self.__init_db_engine(), index_col=0)
        return df
    
    def save_data(self, table, filename):
        '''
        This function saves data to a csv file.

        Parameters:
            table: Name of table to extract data from.
            filename: Name of csv file to save the data as.
        '''
        df = self.read_rds_table(table)
        df.to_csv(filename, index=False)

    def load_data(filename):
        '''
        This function loads the data from a csv file.

        Parameters:
            filename: Name of csv file.
        
        Returns:
            df (DataFrame): The loaded data.
        '''
        df = pd.read_csv(filename)
        return df
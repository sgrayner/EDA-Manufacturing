import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox, yeojohnson
from seaborn import pairplot
import statsmodels.formula.api as smf

from sqlalchemy import create_engine

class RDSDatabaseConnector:
    '''
    This class connects to the RDS database, saves the data to the local machine and loads the data.
    '''
    def __init__(self, creds):
        self.creds = creds

    def read_creds(self):
        '''
        This function is used to read and return credentials of a database.

        Returns:
            data (DataFrame): The credentials for the database.
        '''
        with open(self.creds, 'r') as file:
            data = yaml.safe_load(file)
        return data
    
    def init_db_engine(self):
        '''
        This function initiates an SQLalchemy database engine.

        Returns:
            engine: An SQLalchemy engine.
        '''
        data = self.read_creds()
        DATABASE_TYPE = 'postgresql'
        DBAPI = 'psycopg2'
        HOST = data['RDS_HOST']
        USER = data['RDS_USER']
        PASSWORD = data['RDS_PASSWORD']
        DATABASE = data['RDS_DATABASE']
        PORT = data['RDS_PORT']
        engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")
        return engine
    
    def read_rds_table(self, table):
        '''
        This function reads data from an RDS database.
        
        Args:
            table: Name of table to extract data from.
            
        Returns:
            df (DataFrame): the extracted data as a DataFrame.
        '''
        df = pd.read_sql_table(table, self.init_db_engine(), index_col=0)
        return df
    
    def save_data(self):
        '''
        This function saves the data to a csv file.
        '''
        df = self.read_rds_table('failure_data')
        df.to_csv('failure_data.csv', index=False)

    def load_data():
        '''
        This function loads the data from a csv file.
        
        Returns:
            df (DataFrame): The loaded data.
        '''
        df = pd.read_csv('failure_data.csv')
        return df
    
class DataTransform:

    def cleaning():
        df = RDSDatabaseConnector.load_data()
        df['Type'] = df['Type'].astype('category')
        return df

    def impute_nulls():
        df = DataTransform.cleaning()
        without_nulls = df.dropna(subset=('Air temperature [K]', 'Process temperature [K]'))
        process_temp_gradient, process_temp_intercept = np.polyfit(without_nulls['Air temperature [K]'], without_nulls['Process temperature [K]'], deg=1)
        air_temp_gradient, air_temp_intercept = np.polyfit(without_nulls['Process temperature [K]'], without_nulls['Air temperature [K]'], deg=1)
        process_temp_nulls = df[df['Air temperature [K]'].notnull() & df['Process temperature [K]'].isnull()]
        process_temp_nulls['Process temperature [K]'] = np.round(process_temp_gradient * process_temp_nulls['Air temperature [K]'] + process_temp_intercept, 1)
        air_temp_nulls = df[df['Air temperature [K]'].isnull() & df['Process temperature [K]'].notnull()]
        air_temp_nulls['Air temperature [K]'] = np.round(air_temp_gradient * air_temp_nulls['Process temperature [K]'] + air_temp_intercept, 1)
        df.update(air_temp_nulls)
        df.update(process_temp_nulls)
        df = df.dropna(subset=('Air temperature [K]', 'Process temperature [K]'))
        df['Tool wear [min]'] = df['Tool wear [min]'].fillna(df['Tool wear [min]'].mean())
        return df

    def remove_skew():
        df = DataTransform.impute_nulls()
        df['Rotational speed [rpm]'] = boxcox(df['Rotational speed [rpm]'])[0]
        df['Machine failure'] = yeojohnson(df['Machine failure'] + 0.1)[0]
        df['TWF'] = yeojohnson(df['TWF'] + 0.1)[0]
        df['HDF'] = yeojohnson(df['HDF'] + 0.1)[0]
        df['PWF'] = yeojohnson(df['PWF'] + 0.1)[0]
        df['OSF'] = yeojohnson(df['OSF'] + 0.1)[0]
        df['RNF'] = yeojohnson(df['RNF'] + 0.1)[0]
        return df
    
    def remove_collinearity():
        df = DataTransform.remove_skew()
        df.drop('Rotational speed [rpm]', axis=1, inplace=True) # VIF = 6.25
    
class DataFrameInfo:

    def check_dtypes():
        df = DataTransform.remove_skew()
        return df.info()
    
    def stats(column):
        df = DataTransform.remove_skew()
        mean = df[column].mean()
        median = df[column].median()
        LQ = df[column].quantile(0.25)
        UQ = df[column].quantile(0.75)
        range = df[column].max() - df[column].min()
        sd = df[column].std()
        return {'mean': mean, 'median': median, 'LQ': LQ, 'UQ': UQ, 'range': range, 'sd': sd}
    
    def drop_outliers(column):
        df = DataTransform.remove_skew()
        df['z-score'] = (df[column] - df[column].mean())/df[column].std()
        outliers = df[abs(df['z-score']) > 3][column]
        df = df[~df[column].isin(outliers)]
        df.drop('z-score', axis=1, inplace=True)
        return df
    
    def skew():
        df = DataTransform.remove_skew()
        return df.skew()
    
    def distinct_vals(column):
        df = DataTransform.remove_skew()
        return df[column].value_counts()
    
    def dimensions():
        df = DataTransform.remove_skew()
        dimensions = df.shape
        return dimensions
    
    def percentage_null():
        df = DataTransform.remove_skew()
        percent_null = df.isnull().sum()*100/len(df)
        return percent_null
    


class Plotter:

    def histogram(column):
        df = DataTransform.remove_skew()
        histogram = df[column].plot.hist(bins=50)
        plt.show()

    def pairplot():
        df = DataTransform.remove_skew()
        plot = pairplot(df)
        plt.show()
    
    def scatter(x_column, y_column):
        df = DataTransform.remove_skew()
        scattergraph = plt.scatter(df[x_column], df[y_column])
        plt.show()

    def boxplot(column):
        df = DataTransform.remove_skew()
        plot = plt.boxplot(df[column])
        plt.show()

    

#print(DataFrameInfo.outliers('Torque [Nm]'))
df = DataTransform.remove_skew()
print(Plotter.boxplot('Machine failure'))

print(df.head())
#print(DataFrameInfo.percentage_null())
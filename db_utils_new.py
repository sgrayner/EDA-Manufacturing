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
    
    def save_data(self, table, filename):
        '''
        This function saves the data to a csv file.
        '''
        df = self.read_rds_table(table)
        df.to_csv(filename, index=False)

    def load_data(filename):
        '''
        This function loads the data from a csv file.
        
        Returns:
            df (DataFrame): The loaded data.
        '''
        df = pd.read_csv(filename)
        return df
    
class DataTransform:

    def cleaning(df):
        '''
        This function cleans the data.

        Returns:
            df (DataFrame): The cleaned dataframe.
        '''
        df['Type'] = df['Type'].astype('category')
        return df

    def impute_nulls(df):
        '''
        This function imputes null-values with either a linear regression function, or an average value.

        Returns:
            df (DataFrame): The imputed dataframe.
        '''
        without_nulls = df.dropna(subset=('Air temperature [K]', 'Process temperature [K]'))
        process_temp_gradient, process_temp_intercept = np.polyfit(without_nulls['Air temperature [K]'], without_nulls['Process temperature [K]'], deg=1)
        air_temp_gradient, air_temp_intercept = np.polyfit(without_nulls['Process temperature [K]'], without_nulls['Air temperature [K]'], deg=1)
        process_temp_nulls = df[df['Air temperature [K]'].notnull() & df['Process temperature [K]'].isnull()]
        process_temp_nulls.loc[:, 'Process temperature [K]'] = np.round(process_temp_gradient * process_temp_nulls['Air temperature [K]'] + process_temp_intercept, 1)
        air_temp_nulls = df[df['Air temperature [K]'].isnull() & df['Process temperature [K]'].notnull()]
        air_temp_nulls.loc[:, 'Air temperature [K]'] = np.round(air_temp_gradient * air_temp_nulls['Process temperature [K]'] + air_temp_intercept, 1)
        df.update(air_temp_nulls)
        df.update(process_temp_nulls)
        df = df.dropna(subset=('Air temperature [K]', 'Process temperature [K]'))
        df.loc[:, 'Tool wear [min]'] = df['Tool wear [min]'].fillna(df['Tool wear [min]'].mean())
        return df

    def remove_skew(df):
        '''
        This function transforms the data to remove skew.
        
        Returns:
            df (DataFrame): The tranformed dataframe.
        '''
        df.loc[:, 'Rotational speed [rpm]'] = boxcox(df['Rotational speed [rpm]'])[0]
        return df
    
    def drop_outliers(df, column):
        '''
        This utility function removes outliers from a column of the data.

        Parameters:
            column (str): The variable to drop outliers from.

        Returns:
            df (DataFrame): The dataframe with the outliers removed from 'column'.
        '''
        df.loc[:, 'z-score'] = (df[column] - df[column].mean())/df[column].std()
        outliers = df[abs(df['z-score']) > 3][column]
        df = df[~df[column].isin(outliers)]
        df.drop('z-score', axis=1, inplace=True)
        return df
    
    def remove_collinearity(df):
        '''
        This function removes collinear variables from the data.
        
        Returns:
            df (DataFrame): The tranformed dataframe.
        '''
        df.drop('Rotational speed [rpm]', axis=1, inplace=True) # VIF = 6.25
        return df
    
class DataFrameInfo:
    
    def stats(series):
        '''
        This utility function returns central tendency and dispersion statistics.
        '''
        mean = series.mean()
        median = series.median()
        LQ = series.quantile(0.25)
        UQ = series.quantile(0.75)
        range = series.max() - series.min()
        sd = series.std()
        return {'mean': mean, 'median': median, 'LQ': LQ, 'UQ': UQ, 'range': range, 'sd': sd}

class DataAnalysis:

    def ranges_table(df):
        data = {('Air_temp', 'Max'): [], ('Air_temp', 'Min'): [], ('Process_temp', 'Max'): [], ('Process_temp', 'Min'): [],\
                ('Torque', 'Max'): [], ('Torque', 'Min'): [], ('Tool_wear', 'Max'): [], ('Tool_wear', 'Min'): []}
        for column in range(3, 7):
            all_max = df[df.columns[column]].max()
            all_min = df[df.columns[column]].min()
            H_max = df[df['Type'] == 'H'][df.columns[column]].max()
            H_min = df[(df['Type'] == 'H') & (df[df.loc[:, column]])].min()
            M_max = df[(df['Type'] == 'M') & (df[df.loc[:, column]])].max()
            M_min = df[(df['Type'] == 'M') & (df[df.loc[:, column]])].min()
            L_max = df[(df['Type'] == 'L') & (df[df.loc[:, column]])].max()
            L_min = df[(df['Type'] == 'L') & (df[df.loc[:, column]])].min()
            data.loc[:, list(data.keys())[2*column - 6]] = [all_max, H_max, M_max, L_max]
            data.loc[:, list(data.keys())[2*column - 5]] = [all_min, H_min, M_min, L_min]
        range_table = pd.DataFrame(data, index=['All', 'High', 'Medium', 'Low'])
        return range_table
    
    def failure_over_column_range(df, failure, column):
        frequencies = df[df[failure] != 0.0][column]
        plt.hist(frequencies, bins=50, density=True)
        plt.hist(df[column], bins=50, density=True, alpha=0.5)
        plt.title(f'{failure} failures and {column}')
        plt.xlabel(f'{column}')
        plt.ylabel('Relative frequency')
        plt.xlim([df[column].min(), df[column].max()])
        plt.legend([failure, column])
        plt.show()

    def failure_plots(df):
        num_failures = len(df[df['Machine failure'] != 0.0])
        percent_failures = num_failures*100/len(df)
        print('Number of failures =', num_failures, "\n" 'Percentage of failures =', percent_failures)
        H_failures = df[(df['Machine failure'] != 0.0) & (df['Type'] == 'H')]
        M_failures = df[(df['Machine failure'] != 0.0) & (df['Type'] == 'M')]
        L_failures = df[(df['Machine failure'] != 0.0) & (df['Type'] == 'L')]
        TWF_failures = df[df['TWF'] != 0.0]
        HDF_failures = df[df['HDF'] != 0.0]
        PWF_failures = df[df['PWF'] != 0.0]
        OSF_failures = df[df['OSF'] != 0.0]
        RNF_failures = df[df['RNF'] != 0.0]

        ax1_heights = (len(df) - num_failures, num_failures)
        ax1_bars = ('Passed', 'Failed')
        ax2_heights = (len(H_failures), len(M_failures), len(L_failures))
        ax2_bars = ('High', 'Medium', 'Low')
        ax3_heights = (len(TWF_failures), len(HDF_failures), len(PWF_failures), len(OSF_failures), len(RNF_failures))
        ax3_bars = ('TWF', 'HDF', 'PWF', 'OSF', 'RNF')
        failure_types = [TWF_failures, HDF_failures, PWF_failures, OSF_failures, RNF_failures]
        ax4_heights = {'High': [], 'Medium': [], 'Low': []}
        for failure_type in failure_types:
            H_failure = len(failure_type[df['Type'] == 'H'])
            M_failure = len(failure_type[df['Type'] == 'M'])
            L_failure = len(failure_type[df['Type'] == 'L'])
            ax4_heights['High'].append(H_failure)
            ax4_heights['Medium'].append(M_failure)
            ax4_heights['Low'].append(L_failure)
        ax4_bars = ('TWF', 'HDF', 'PWF', 'OSF', 'RNF')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))

        ax1.bar(x=ax1_bars, height=ax1_heights)
        ax1.set_title('Machine failures')
        ax1.set_xlabel('Type')
        ax1.set_ylabel('Frequency')
        ax1.set_ylim([0, 11000])
        for index, data in enumerate(ax1_heights):
            ax1.text(x=index, y=data+200, s=f'{data}', ha='center', size=10)

        ax2.bar(x=ax2_bars, height=ax2_heights)
        ax2.set_title('Machine failures of each product quality type')
        ax2.set_xlabel('Type')
        ax2.set_ylabel('Frequency')
        ax2.set_ylim([0, 275])
        for index, data in enumerate(ax2_heights):
            ax2.text(x=index, y=data+5, s=f'{data}', ha='center', size=10)

        ax3.bar(x=ax3_bars, height=ax3_heights)
        ax3.set_title('Machine failures by failure type')
        ax3.set_xlabel('Type')
        ax3.set_ylabel('Frequency')
        ax3.set_ylim([0, 130])
        for index, data in enumerate(ax3_heights):
            ax3.text(x=index, y=data+2, s=f'{data}', ha='center', size=10)

        x = np.arange(len(ax4_bars))
        bar_width = 0.2
        for index in range(3):
            ax4.bar(x + index*bar_width, list(ax4_heights.values())[index], width=bar_width)
            ax4.text(0 + index/5, list(ax4_heights.values())[index][0] + 1, s=f'{list(ax4_heights.values())[index][0]}', ha='center')
            ax4.text(1 + index/5, list(ax4_heights.values())[index][1] + 1, s=f'{list(ax4_heights.values())[index][1]}', ha='center')
            ax4.text(2 + index/5, list(ax4_heights.values())[index][2] + 1, s=f'{list(ax4_heights.values())[index][2]}', ha='center')
            ax4.text(3 + index/5, list(ax4_heights.values())[index][3] + 1, s=f'{list(ax4_heights.values())[index][3]}', ha='center')
            ax4.text(4 + index/5, list(ax4_heights.values())[index][4] + 1, s=f'{list(ax4_heights.values())[index][4]}', ha='center')
        plt.xticks(x + 0.2, ax4_bars)
        ax4.set_title('Machine failures by failure types and product quality')
        ax4.set_xlabel('Type')
        ax4.set_ylabel('Frequency')
        ax4.set_ylim([0, 90])
        ax4.legend(list(ax4_heights.keys()))

        plt.tight_layout()
        plt.show()

    
df = RDSDatabaseConnector.load_data('failure_data.csv')

clean_df = DataTransform.cleaning(df)
imputed_df = DataTransform.impute_nulls(clean_df)
no_skew_df = DataTransform.remove_skew(imputed_df)
no_rpm_outliers_df = DataTransform.drop_outliers(no_skew_df, 'Rotational speed [rpm]')
no_Nm_outliers_df = DataTransform.drop_outliers(no_rpm_outliers_df, 'Torque [Nm]')
transformed_df = DataTransform.remove_collinearity(no_Nm_outliers_df)

print(DataAnalysis.ranges_table(df))
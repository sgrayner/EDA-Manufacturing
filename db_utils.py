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

    def __read_creds__(self):
        '''
        This function is used to read and return credentials of a database.

        Returns:
            data (DataFrame): The credentials for the database.
        '''
        with open(self.creds, 'r') as file:
            data = yaml.safe_load(file)
        return data
    
    def __init_db_engine__(self):
        '''
        This function initiates an SQLalchemy database engine.

        Returns:
            engine: An SQLalchemy engine.
        '''
        data = self.__read_creds__()
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
        
        Parameters:
            table: Name of table to extract data from.
            
        Returns:
            df (DataFrame): the extracted data as a DataFrame.
        '''
        df = pd.read_sql_table(table, self.__init_db_engine__(), index_col=0)
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


class DataTransform:
    '''
    This class cleans and transforms the data, ready for analysis.
    '''
    def cleaning(df):
        '''
        This function cleans the data.

        Parameters:
            df: The dataframe to input.

        Returns:
            df (DataFrame): The cleaned dataframe.
        '''
        df.loc[:, 'Type'] = df['Type'].astype('category')
        return df

    def impute_nulls(df):
        '''
        This function imputes null-values with either a linear regression function, or an average value.

        Parameters:
            df: The dataframe to input.

        Returns:
            df (DataFrame): The imputed dataframe.
        '''
        without_nulls = df.dropna(subset=('Air temperature [K]', 'Process temperature [K]'))
        process_temp_gradient, process_temp_intercept = \
            np.polyfit(without_nulls['Air temperature [K]'], without_nulls['Process temperature [K]'], deg=1)
        air_temp_gradient, air_temp_intercept = \
            np.polyfit(without_nulls['Process temperature [K]'], without_nulls['Air temperature [K]'], deg=1)
        process_temp_nulls = df[df['Air temperature [K]'].notnull() & df['Process temperature [K]'].isnull()]
        process_temp_nulls.loc[process_temp_nulls.index, 'Process temperature [K]'] = \
            np.round(process_temp_gradient * process_temp_nulls.loc[process_temp_nulls.index, 'Air temperature [K]'] + process_temp_intercept, 1)
        air_temp_nulls = df[df['Air temperature [K]'].isnull() & df['Process temperature [K]'].notnull()]
        air_temp_nulls.loc[air_temp_nulls.index, 'Air temperature [K]'] = \
            np.round(air_temp_gradient * air_temp_nulls.loc[air_temp_nulls.index, 'Process temperature [K]'] + air_temp_intercept, 1)
        df.update(air_temp_nulls)
        df.update(process_temp_nulls)
        df = df.dropna(subset=('Air temperature [K]', 'Process temperature [K]'))
        df.loc[df.index, 'Tool wear [min]'] = df['Tool wear [min]'].fillna(df['Tool wear [min]'].mean())
        return df

    def remove_skew(df):
        '''
        This function transforms the data to remove skew.

        Parameters:
            df: The dataframe to input.
        
        Returns:
            df (DataFrame): The tranformed dataframe.
        '''
        df.loc[:, 'Rotational speed [rpm]'] = boxcox(df['Rotational speed [rpm]'])[0]
        return df
    
    def drop_outliers(df, column):
        '''
        This function removes outliers from a column of the data.

        Parameters:
            df: The dataframe to input.
            column: The variable to drop outliers from.

        Returns:
            df (DataFrame): The dataframe with the outliers removed from 'column'.
        '''
        df.loc[df.index, 'z-score'] = (df[column] - df[column].mean())/df[column].std()
        outliers = df[abs(df['z-score']) > 3][column]
        df = df[~df[column].isin(outliers)]
        df = df.drop('z-score', axis=1)
        return df
    
    def remove_collinearity(df):
        '''
        This function removes collinear variables from the data.

        Parameters:
            df: The dataframe to input.
        
        Returns:
            df (DataFrame): The tranformed dataframe.
        '''
        df.drop('Rotational speed [rpm]', axis=1, inplace=True) # VIF = 6.25
        return df


class DataAnalysis:
    '''
    This class analyses and visualises different features of the data.
    '''
    def ranges_table(df):
        '''
        This function calculates the operating ranges of the explanatory variables, 
        as well as for the different product qualities.

        Parameters:
            df: The dataframe to input.

        Returns:
            range_table (DataFrame): Calculated results in a table.
        '''
        data = {('Air_temp', 'Max'): [], ('Air_temp', 'Min'): [], ('Process_temp', 'Max'): [], \
                ('Process_temp', 'Min'): [], ('Torque', 'Max'): [], ('Torque', 'Min'): [], \
                ('Tool_wear', 'Max'): [], ('Tool_wear', 'Min'): []}
        for column in range(3, 7):
            all_max = df[df.columns[column]].max()
            all_min = df[df.columns[column]].min()
            H_max = df[df['Type'] == 'H'][df.columns[column]].max()
            H_min = df[df['Type'] == 'H'][df.columns[column]].min()
            M_max = df[df['Type'] == 'M'][df.columns[column]].max()
            M_min = df[df['Type'] == 'M'][df.columns[column]].min()
            L_max = df[df['Type'] == 'L'][df.columns[column]].max()
            L_min = df[df['Type'] == 'L'][df.columns[column]].min()
            data[list(data.keys())[2*column - 6]] = [all_max, H_max, M_max, L_max]
            data[list(data.keys())[2*column - 5]] = [all_min, H_min, M_min, L_min]
        range_table = pd.DataFrame(data, index=['All', 'High', 'Medium', 'Low'])
        return range_table
    
    def failure_over_column_range(df, failure, column):
        '''
        This function plots a histogram of a failure type 
        superimposed with an explanatory variable

        Parameters:
            df: The dataframe to input.
            failure: The failure type to plot.
            column: The explanatory variable to plot.
        '''
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
        '''
        This function plots bar charts detailing information on
        machine failure types as well as failures of each product quality type.

        Parameters:
            df: The dataframe to input.
        '''
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
        failure_types = [TWF_failures, HDF_failures, PWF_failures, OSF_failures, RNF_failures]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))

        ax1_heights = (len(df) - num_failures, num_failures)
        ax1_bars = ('Passed', 'Failed')
        ax1.bar(x=ax1_bars, height=ax1_heights)
        ax1.set_title('Machine failures')
        ax1.set_xlabel('Type')
        ax1.set_ylabel('Frequency')
        ax1.set_ylim([0, 11000])
        for index, data in enumerate(ax1_heights):
            ax1.text(x=index, y=data+200, s=f'{data}', ha='center', size=10)

        ax2_heights = (len(H_failures), len(M_failures), len(L_failures))
        ax2_bars = ('High', 'Medium', 'Low')
        ax2.bar(x=ax2_bars, height=ax2_heights)
        ax2.set_title('Machine failures of each product quality type')
        ax2.set_xlabel('Type')
        ax2.set_ylabel('Frequency')
        ax2.set_ylim([0, 275])
        for index, data in enumerate(ax2_heights):
            ax2.text(x=index, y=data+5, s=f'{data}', ha='center', size=10)

        ax3_heights = [len(failure_type) for failure_type in failure_types]
        ax3_bars = ('TWF', 'HDF', 'PWF', 'OSF', 'RNF')
        ax3.bar(x=ax3_bars, height=ax3_heights)
        ax3.set_title('Machine failures by failure type')
        ax3.set_xlabel('Type')
        ax3.set_ylabel('Frequency')
        ax3.set_ylim([0, 130])
        for index, data in enumerate(ax3_heights):
            ax3.text(x=index, y=data+2, s=f'{data}', ha='center', size=10)

        ax4_heights = {'High': [], 'Medium': [], 'Low': []}
        for failure_type in failure_types:
            H_failure = len(failure_type.loc[df['Type'] == 'H'])
            M_failure = len(failure_type.loc[df['Type'] == 'M'])
            L_failure = len(failure_type.loc[df['Type'] == 'L'])
            ax4_heights['High'].append(H_failure)
            ax4_heights['Medium'].append(M_failure)
            ax4_heights['Low'].append(L_failure)
        positions = np.arange(len(ax3_bars))
        bar_width = 0.2
        for index in range(3):
            ax4.bar(positions + index*bar_width, list(ax4_heights.values())[index], width=bar_width)
            ax4.text(0 + index/5, list(ax4_heights.values())[index][0] + 1, \
                     s=f'{list(ax4_heights.values())[index][0]}', ha='center', size=10)
            ax4.text(1 + index/5, list(ax4_heights.values())[index][1] + 1, \
                     s=f'{list(ax4_heights.values())[index][1]}', ha='center', size=10)
            ax4.text(2 + index/5, list(ax4_heights.values())[index][2] + 1, \
                     s=f'{list(ax4_heights.values())[index][2]}', ha='center', size=10)
            ax4.text(3 + index/5, list(ax4_heights.values())[index][3] + 1, \
                     s=f'{list(ax4_heights.values())[index][3]}', ha='center', size=10)
            ax4.text(4 + index/5, list(ax4_heights.values())[index][4] + 1, \
                     s=f'{list(ax4_heights.values())[index][4]}', ha='center', size=10)
        plt.xticks(positions + 0.2, ax3_bars)
        ax4.set_title('Machine failures by failure types and product quality')
        ax4.set_xlabel('Type')
        ax4.set_ylabel('Frequency')
        ax4.set_ylim([0, 100])
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
import pandas as pd
import numpy as np
from scipy.stats import boxcox


class DataTransform:
    '''
    This class cleans and transforms the data, ready for analysis.
    '''
    def cleaning(df: pd.DataFrame):
        '''
        This function cleans the data.

        Parameters:
            df: The dataframe to input.

        Returns:
            df (DataFrame): The cleaned dataframe.
        '''
        df.loc[:, 'Type'] = df['Type'].astype('category')
        return df

    def lin_reg_impute(df: pd.DataFrame, col1: str, col2: str):
        '''
        This function imputes null-values in col1 and col2 with a linear regression function.

        Parameters:
            df: The dataframe to input.
            col1: The first column in df.
            col2: The second column in df.

        Returns:
            df (DataFrame): The imputed dataframe.
        '''
        without_nulls = df.dropna(subset=(col1, col2))
        col2_gradient, col2_intercept = \
            np.polyfit(without_nulls[col1], without_nulls[col2], deg=1)
        col1_gradient, col1_intercept = \
            np.polyfit(without_nulls[col2], without_nulls[col1], deg=1)
        col2_nulls = df[df[col1].notnull() & df[col2].isnull()]
        col2_nulls.loc[col2_nulls.index, col2] = \
            np.round(col2_gradient * col2_nulls.loc[col2_nulls.index, col1] + col2_intercept, 1)
        col1_nulls = df[df[col1].isnull() & df[col2].notnull()]
        col1_nulls.loc[col1_nulls.index, col1] = \
            np.round(col1_gradient * col1_nulls.loc[col1_nulls.index, col2] + col1_intercept, 1)
        df.update(col1_nulls)
        df.update(col2_nulls)
        df = df.dropna(subset=(col1, col2))
        return df
    
    def mean_impute(df: pd.DataFrame, col: str):
        '''
        This function imputes null-values in col with the mean value.

        Parameters:
            df: The dataframe to impute.
            col: The column to impute.

        Returns:
            df (DataFrame): The imputed dataframe.
        '''
        df.loc[df.index, col] = df[col].fillna(df[col].mean())
        return df

    def boxcox_skew_transform(df: pd.DataFrame, col: str):
        '''
        This function applies a Box-Cox transform to the data in the 'col' column to remove skew.

        Parameters:
            df: The dataframe to input.
            col: The skewed column.
        
        Returns:
            df (DataFrame): The tranformed dataframe.
        '''
        df.loc[:, col] = boxcox(df[col])[0]
        return df
    
    def drop_outliers(df: pd.DataFrame, col: str):
        '''
        This function removes outliers from a column of the data.

        Parameters:
            df: The dataframe to input.
            column: The variable to drop outliers from.

        Returns:
            df (DataFrame): The dataframe with the outliers removed from 'column'.
        '''
        df.loc[df.index, 'z-score'] = (df[col] - df[col].mean())/df[col].std()
        outliers = df[abs(df['z-score']) > 3][col]
        df = df[~df[col].isin(outliers)]
        df = df.drop('z-score', axis=1)
        return df
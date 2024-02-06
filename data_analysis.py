import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DataAnalysis:
    '''
    This class analyses the values of the variables.
    '''
    def ranges_table(df: pd.DataFrame):
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
            data[list(data.keys())[2*column - 6]].append(all_max)
            data[list(data.keys())[2*column - 5]].append(all_min)
            for quality in ('H', 'M', 'L'):
                max = df[df['Type'] == quality][df.columns[column]].max()
                min = df[df['Type'] == quality][df.columns[column]].min()
                data[list(data.keys())[2*column - 6]].append(max)
                data[list(data.keys())[2*column - 5]].append(min)
        range_table = pd.DataFrame(data, index=['All', 'High', 'Medium', 'Low'])
        return range_table
    

class DataPlotter:
    '''
    This class creates plots of different features of the data.
    '''
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.num_failures = len(df[df['Machine failure'] != 0.0])
        self.percent_failures = self.num_failures*100/len(df)
        self.H_failures = df[(df['Machine failure'] != 0.0) & (df['Type'] == 'H')]
        self.M_failures = df[(df['Machine failure'] != 0.0) & (df['Type'] == 'M')]
        self.L_failures = df[(df['Machine failure'] != 0.0) & (df['Type'] == 'L')]
        self.TWF_failures = df[df['TWF'] != 0.0]
        self.HDF_failures = df[df['HDF'] != 0.0]
        self.PWF_failures = df[df['PWF'] != 0.0]
        self.OSF_failures = df[df['OSF'] != 0.0]
        self.RNF_failures = df[df['RNF'] != 0.0]
        self.failure_types = [self.TWF_failures, self.HDF_failures, self.PWF_failures, self.OSF_failures, self.RNF_failures]
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(15, 8))

    def failure_over_column_range(self, failure: str, column: str):
        '''
        This function plots a histogram of a failure type 
        superimposed with an explanatory variable

        Parameters:
            failure: The failure type to plot.
            column: The explanatory variable to plot.
        '''
        frequencies = self.df[self.df[failure] != 0.0][column]
        plt.hist(frequencies, bins=50, density=True)
        plt.hist(self.df[column], bins=50, density=True, alpha=0.5)
        plt.title(f'{failure} failures and {column}')
        plt.xlabel(f'{column}')
        plt.ylabel('Relative frequency')
        plt.xlim([self.df[column].min(), self.df[column].max()])
        plt.legend([failure, column])
        plt.show()

    def __machine_failures(self):
        '''
        This function creates a plot of a bar chart of passes and failures of the machine.
        '''
        ax1_heights = (len(self.df) - self.num_failures, self.num_failures)
        ax1_bars = ('Passed', 'Failed')
        self.ax1.bar(x=ax1_bars, height=ax1_heights)
        self.ax1.set_title('Machine failures')
        self.ax1.set_xlabel('Type')
        self.ax1.set_ylabel('Frequency')
        self.ax1.set_ylim([0, 11000])
        for index, data in enumerate(ax1_heights):
            self.ax1.text(x=index, y=data+200, s=f'{data}', ha='center', size=10)

    def __failures_by_quality(self):
        '''
        This function creates a plot of a bar chart of failures for each product quality type.
        '''
        ax2_heights = (len(self.H_failures), len(self.M_failures), len(self.L_failures))
        ax2_bars = ('High', 'Medium', 'Low')
        self.ax2.bar(x=ax2_bars, height=ax2_heights)
        self.ax2.set_title('Machine failures of each product quality type')
        self.ax2.set_xlabel('Type')
        self.ax2.set_ylabel('Frequency')
        self.ax2.set_ylim([0, 275])
        for index, data in enumerate(ax2_heights):
            self.ax2.text(x=index, y=data+5, s=f'{data}', ha='center', size=10)

    def __failures_by_type(self):
        '''
        This function creates a plot of a bar chart of the different types of failures.
        '''
        ax3_heights = [len(failure_type) for failure_type in self.failure_types]
        ax3_bars = ('TWF', 'HDF', 'PWF', 'OSF', 'RNF')
        self.ax3.bar(x=ax3_bars, height=ax3_heights)
        self.ax3.set_title('Machine failures by failure type')
        self.ax3.set_xlabel('Type')
        self.ax3.set_ylabel('Frequency')
        self.ax3.set_ylim([0, 130])
        for index, data in enumerate(ax3_heights):
            self.ax3.text(x=index, y=data+2, s=f'{data}', ha='center', size=10)

    def __failures_by_qualityandtype(self):
        '''
        This function creates a plot of a grouped bar chart of the different types of failures
            for each product quality type.
        '''
        ax4_heights = {'High': [], 'Medium': [], 'Low': []}
        ax4_bars = ('TWF', 'HDF', 'PWF', 'OSF', 'RNF')
        for failure_type in self.failure_types:
            H_failure = len(failure_type.loc[self.df['Type'] == 'H'])
            M_failure = len(failure_type.loc[self.df['Type'] == 'M'])
            L_failure = len(failure_type.loc[self.df['Type'] == 'L'])
            ax4_heights['High'].append(H_failure)
            ax4_heights['Medium'].append(M_failure)
            ax4_heights['Low'].append(L_failure)
        positions = np.arange(len(ax4_bars))
        bar_width = 0.2
        for index in range(3):
            self.ax4.bar(positions + index*bar_width, list(ax4_heights.values())[index], width=bar_width)
            for position in range(5):
                self.ax4.text(position + index/5, list(ax4_heights.values())[index][position] + 1, \
                        s=f'{list(ax4_heights.values())[index][position]}', ha='center', size=10)
        plt.xticks(positions + 0.2, ax4_bars)
        self.ax4.set_title('Machine failures by failure types and product quality')
        self.ax4.set_xlabel('Type')
        self.ax4.set_ylabel('Frequency')
        self.ax4.set_ylim([0, 100])
        self.ax4.legend(list(ax4_heights.keys()))

    def failure_plots(self):
        '''
        This function plots bar charts detailing information on
        machine failure types as well as failures of each product quality type.
        '''
        print('Number of failures =', self.num_failures, "\n" 'Percentage of failures =', self.percent_failures)
        self.__machine_failures()
        self.__failures_by_quality()
        self.__failures_by_type()
        self.__failures_by_qualityandtype()
        plt.tight_layout()
        plt.show()
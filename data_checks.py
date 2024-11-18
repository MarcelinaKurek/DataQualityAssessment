import os
import pandas as pd
import numpy as np
import warnings
import re
from visualizer import Visualizer
from language_validator import LanguageValidator
from scipy.stats import mode
warnings.filterwarnings('ignore')


class PandasValidator:
    def __init__(self, df, data_path):
        self.susp_values_list = []
        self.checks_done = 0
        self.df = df
        self.types_dict = {}
        self.nrows = self.df.shape[0]
        self.result_path = self.make_necessary_folders(data_path)
        self.visualizer = Visualizer(data_path, self.df)
        self.language_validator = None

    @staticmethod
    def make_necessary_folders(data_path):
        """
        Function prepares necessary folders for data check results
        :param data_path: Path to dataset
        :return:
        """
        filepath = data_path.split(".")[0]
        filepath_0 = '/'.join(filepath.split("/")[1:])
        folder_name = f"results/{filepath_0}/tables"
        if not os.path.exists(f"{folder_name}"):
            os.makedirs(f"{folder_name}")
        return folder_name


    def missing_values_check(self, columns):
        """
        Function checks missing values in dataframe
        :param columns: Columns to check
        """
        missing_values = self.df[columns].isna().sum().tolist()
        if any(missing_values):
            idx = np.where(missing_values)[0]
            print(
                f"Column(s) with missing values: {', '.join([columns[i] + ' : ' + str(missing_values[i]) + ' (' + '{:.2f}'.format(missing_values[i] / self.nrows) + '%)' for i in idx])}")

    def categorical_values_check(self, categories_df):
        """
        Function to perform validation on values in categorical columns, using Regexes and language models
        :param categories_df:
        :return:
        """
        # susp_values_dict = self.regex_check(categories_df)
        self.language_validator = LanguageValidator(categories_df['Names'])
        self.language_validator.run_language_validator()
        return

    def run_categorical_validator(self):
        """
        Perform data validation for categorical columns
        """
        print("\nRunning categorical validator...")
        categorical_columns = self.df.select_dtypes(include='object').columns.tolist()
        if any(categorical_columns):
            print("Categorical columns:", categorical_columns)
            #prepare summary dataframe
            summary_df = self.df[categorical_columns].apply(lambda x: [len(x.value_counts()),list(x.value_counts().index),list(x.value_counts().values)], axis=0)
            summary_df = pd.DataFrame(summary_df).T
            summary_df.columns = ['Categories nr', 'Names', 'Counted']
            self.categorical_values_check(summary_df)
            summary_df.to_csv(f"{self.result_path}/categorical_summary_df.csv")
            print("Categorical summary statistics saved.")
            self.missing_values_check(categorical_columns)
            unique_categories = np.where(summary_df['Categories nr'] == self.df.shape[0])[0]
            if any(unique_categories):
                idx = np.where(unique_categories)[0]
                print(f"Categorical column with unique values: {', '.join([categorical_columns[i] for i in idx])}")
            self.visualizer.plot_categories_count(categorical_columns)
            print("Running categorical validator finished.")
        else:
            print("No categorical columns found.")
        return

    def run_numeric_validator(self):
        """
        Perform data validation for numeric columns
        """
        print("\nRunning numeric validator...")
        numeric_columns = self.df.select_dtypes(include='number').columns.tolist()
        print("Numeric columns:", numeric_columns)
        if any(numeric_columns):
            summary_df = self.df[numeric_columns].describe()
            additional_quantiles = self.df[numeric_columns].quantile([.05, .95])
            additional_quantiles.index = ['5%', '95%']
            mode_df = pd.DataFrame(mode(self.df[numeric_columns], axis=0), columns=numeric_columns)
            mode_df.index = ['mode', 'mode_ratio']
            mode_df.loc['mode_ratio',:] /= np.round(self.df.shape[0], 3)
            summary_df = pd.concat([summary_df, additional_quantiles, mode_df], axis=0, ignore_index = False, keys = None)
            summary_df = summary_df.round(decimals=2)
            summary_df.to_csv(f"{self.result_path}/numeric_summary_df.csv")
            self.print_numeric_summary(summary_df)
            print("Numeric summary statistics saved.")
            self.missing_values_check(numeric_columns)
            self.visualizer.plot_histograms(numeric_columns)
            print("Running numeric validator finished.")
        else:
            print("No numeric columns found.")
        return

    def print_numeric_summary(self, summary_df):
        """
        print warnings based on numeric summary statistics
        :param summary_df: numeric summary table
        """
        negative_value_columns = np.where(summary_df.loc['min'] < 0)[0]
        small_mode = np.where(summary_df.loc['mode'] <= summary_df.loc['5%'])[0]
        large_mode = np.where(summary_df.loc['mode'] >= summary_df.loc['95%'])[0]

        if any(negative_value_columns):
            print(f"Columns with negative values: {', '.join(summary_df.columns[negative_value_columns])}")
        if any(small_mode):
            print(f"Columns with mode below 5th quantile: {', '.join(summary_df.columns[small_mode])}")
        if any(large_mode):
            print(f"Columns with mode above 95th quantile: {', '.join(summary_df.columns[large_mode])}")

    def run_datetime_validator(self):
        """
        Perform data validation for datetime columns
        """
        print("\nRunning datetime validator...")
        date_columns = self.df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
        numeric_columns = self.df.select_dtypes(include=['number']).columns.tolist()
        print("Datetime columns:", date_columns)
        if any(date_columns):
            self.missing_values_check(date_columns)
            for column in date_columns:
                date_interval = max(self.df[column]) - min(self.df[column])
                print(f"Range for {column}: {min(self.df[column])} - {max(self.df[column])} ({date_interval})")
                print(f"Number of unique days: {len(self.df[column].value_counts())}", end='\n')
                self.visualizer.plot_time_series(column, numeric_columns)
            print("Running datetime validator finished.")
        else:
            print("No datetime columns found.")
        return

import logging
import numpy as np
import os
import pandas as pd
import warnings

from scipy.stats import mode, chisquare

from visualizer import Visualizer
from language_validator import LanguageValidator

# warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)
logging.captureWarnings(False)

# Define a custom warning
class NumericValuesWarning(Warning):
    pass

class DateWarning(Warning):
    pass

class ObjectDataWarning(Warning):
    pass

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

    def duplicate_values_check(self, columns):
        duplicates = self.df[columns].duplicated()
        if any(duplicates):
            idx = np.where(duplicates)[0]
            print(f"Duplicated rows indices: {idx}")

    def categorical_values_check(self, categories_df):
        """
        Function to perform validation on values in categorical columns, using Regexes and language models
        :param categories_df:
        :return:
        """
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
            self.missing_values_check(categorical_columns)
            summary_df = self.prepare_categorical_summary_df(categorical_columns)
            self.print_categorical_summary(summary_df, categorical_columns)
            self.categorical_values_check(summary_df)
            self.visualizer.plot_categories_count(categorical_columns)
            print("Running categorical validator finished.")
        else:
            raise Exception("No categorical columns found.")
        return

    def print_categorical_summary(self, summary_df, categorical_columns):
        unique_categories = np.where(summary_df['Categories nr'] == self.df.shape[0])[0]
        single_category = np.where(summary_df['Categories nr'] == 1)[0]
        if any(unique_categories):
            idx = np.where(unique_categories)[0]
            warnings.warn(
                f"Categorical column with unique values: {', '.join([categorical_columns[i] for i in idx])}",
                category=ObjectDataWarning, stacklevel=4)
        if any(single_category):
            idx = np.where(single_category)[0]
            warnings.warn(
                f"Categorical column with constant value: {', '.join([categorical_columns[i] for i in idx])}",
                category=ObjectDataWarning, stacklevel=4)

    def prepare_categorical_summary_df(self, categorical_columns):
        summary_df = self.df[categorical_columns].apply(
            lambda x: [len(x.value_counts()), list(x.value_counts().index), list(x.value_counts().values)], axis=0)
        summary_df = pd.DataFrame(summary_df).T
        summary_df.columns = ['Categories nr', 'Names', 'Counted']
        summary_df.to_csv(f"{self.result_path}/categorical_summary_df.csv")
        print("Categorical summary statistics saved.")
        return summary_df

    def run_numeric_validator(self):
        """
        Perform data validation for numeric columns
        """
        print("\nRunning numeric validator...")
        numeric_columns = self.df.select_dtypes(include='number').columns.tolist()
        print("Numeric columns:", numeric_columns)
        if any(numeric_columns):
            summary_df = self.prepare_numeric_summary(numeric_columns)
            self.print_numeric_summary(summary_df)
            self.missing_values_check(numeric_columns)
            self.visualizer.plot_histograms(numeric_columns)
            print("Running numeric validator finished.")
        else:
            raise Exception("No numeric columns found.")
        return

    def prepare_numeric_summary(self, numeric_columns):
        summary_df = self.df[numeric_columns].describe()
        additional_quantiles = self.df[numeric_columns].quantile([.01, .05, .1, .9, .95, .99])
        additional_quantiles.index = ['1%', '5%', '10%', '90%', '95%', '99%']
        mode_df = pd.DataFrame(np.array(mode(self.df[numeric_columns], axis=0, nan_policy='omit'), dtype=float), columns=numeric_columns)
        mode_df.index = ['mode', 'mode_ratio']
        mode_df.loc['mode_ratio', :] = mode_df.loc['mode_ratio', :].astype(float)
        mode_df.loc['mode_ratio', :] /= np.round(self.df.shape[0], 3)
        sorted_arr = np.sort(self.df[numeric_columns], axis=0)
        row_differences = np.diff(sorted_arr, axis=0)
        max_row_difference = np.nanmax(row_differences, axis=0)
        min_row_difference = np.nanmin(row_differences, axis=0)
        diff_df = pd.DataFrame([min_row_difference, max_row_difference], columns=numeric_columns,
                               index=['min_row_diff', 'max_row_diff'])
        summary_df = pd.concat([summary_df, additional_quantiles, mode_df, diff_df], axis=0, ignore_index=False,
                               keys=None)
        summary_df = summary_df.round(decimals=2)
        summary_df.to_csv(f"{self.result_path}/numeric_summary_df.csv")
        print("Numeric summary statistics saved.")
        return summary_df

    def print_numeric_summary(self, summary_df):
        """
        print warnings based on numeric summary statistics
        :param summary_df: numeric summary table
        """
        negative_value_columns = np.where(summary_df.loc['min'] < 0)[0]
        small_percentiles = ['1%', '5%', '10%']
        large_percentiles = ['90%', '95%', '99%']
        small_mode = np.where(summary_df.loc['50%'] <= summary_df.loc[small_percentiles,:])
        large_mode = np.where(summary_df.loc['50%'] >= summary_df.loc[large_percentiles,:])

        if any(negative_value_columns):
            warnings.warn(f"Columns with negative values: {', '.join(summary_df.columns[negative_value_columns])}",
                          category=NumericValuesWarning, stacklevel=4)
        if any(small_mode[0]):
            small_mode = pd.DataFrame(small_mode).T.groupby(small_mode[1]).min().astype(int)[0]
            for i in pd.unique(small_mode):
                cols_idx = small_mode.iloc[np.where(small_mode == i)[0]].index
                warnings.warn(f"Columns with mode below {small_percentiles[i]}th quantile: {', '.join(summary_df.columns[cols_idx])}",
                              category=NumericValuesWarning, stacklevel=4)
        if any(large_mode[0]):
            large_mode = pd.DataFrame(large_mode).T.groupby(large_mode[1]).min().astype(int)[0]
            for i in pd.unique(large_mode):
                cols_idx = large_mode.iloc[np.where(large_mode == i)[0]].index
                warnings.warn(
                    f"Columns with mode above {large_percentiles[i]}th quantile: {', '.join(summary_df.columns[cols_idx])}",
                    category=NumericValuesWarning, stacklevel=4)

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
                date_counts = self.df[column].value_counts()
                print(f"Number of unique days: {len(date_counts)}", end='\n')
                print(f"Maximum number of timestamps per date: {date_counts.iloc[0]}", end='\n')
                self.visualizer.plot_time_series(column, numeric_columns)
                bins = pd.cut(self.df[column].astype(int), bins=date_interval.days)
                bin_counts = bins.value_counts()
                chi2, p_value = chisquare(bin_counts)
                if p_value < 0.05:
                    warnings.warn("Dates are not uniformly distributed.", category=DateWarning, stacklevel=4)
            print("Running datetime validator finished.")
        else:
            raise Exception("No datetime columns found.")
        return

import itertools
import logging
import numpy as np
import os
import pandas as pd
import warnings

from dateutil import relativedelta
from scipy.stats import mode, chisquare
from time import time

from visualizer import Visualizer
from language_validator import LanguageValidator
from utils import make_necessary_folders

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
        self.data_path = data_path
        self.result_path = make_necessary_folders(data_path)
        self.visualizer = Visualizer(data_path, self.df)
        self.language_validator = None
        self.linters_dict = {"duplicates": 0, "missing_values": 1, "num_negative_values": 2, "num_small_mode": 3,
                             "num_large_mode": 4, "num_small_mode_cnt": 5, "num_skewness": 6,
                             "num_wide_range": 7, "num_max_gap": 8, "obj_unique": 9, "obj_same": 10, "obj_long_avg": 11,
                             "obj_long_max": 12, "lang_special_chars": 13, "lang_numeric_vals": 14,
                             "lang_oov_invalid": 15,
                             "lang_oov_custom": 16, "lang_abbr": 17, "time_format": 18, "time_monotonic": 19,
                             "time_not_uniform": 20, "time_long": 21}
        self.linter_count = np.zeros(22).astype(int)

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
            self.linter_count[self.linters_dict["missing_values"]] += len(idx)
        columns_with_all_na = self.df.isna().all()
        if columns_with_all_na.any():
            print(f"Dropping columns with all NaN values: {columns_with_all_na}")
            self.df = self.df.dropna(axis=1, how="all")

    def duplicate_values_check(self, columns):
        """
        Function checks duplicate values in dataframe
        """
        duplicates = self.df[columns].duplicated()
        if any(duplicates):
            idx = np.where(duplicates)[0]
            print(f"Duplicated rows: {len(idx)} ({len(idx) / self.nrows:.2f}%)")
            self.linter_count[self.linters_dict["duplicates"]] += 1

    def categorical_values_check(self, categories_df):
        """
        Function to perform validation on values in categorical columns, using Regexes and language models
        :param categories_df: summary dataframe
        :return:
        """
        self.language_validator = LanguageValidator(categories_df, self.data_path, self.visualizer, self.linter_count,
                                                    self.linters_dict)
        self.language_validator.run_language_validator()
        return

    def run_categorical_validator(self):
        """
        Perform data validation for categorical columns
        """
        print("\nRunning categorical validator...")
        start_time = time()
        categorical_columns = self.df.select_dtypes(include='object').columns.tolist()
        if any(categorical_columns):
            print("Categorical columns:", categorical_columns)
            self.missing_values_check(categorical_columns)
            self.duplicate_values_check(categorical_columns)
            summary_df = self.prepare_categorical_summary_df(categorical_columns)
            self.print_categorical_summary(summary_df, categorical_columns)
            language_df = summary_df.loc[(summary_df['Avg Token Length'] < 128) & (
                    summary_df['Categories nr'] <= self.df.shape[0] * 0.98), :]
            self.categorical_values_check(language_df)
            end_time = time()
            print(f"Running categorical validator finished in {end_time - start_time:.2f} seconds")
        else:
            raise Exception("No categorical columns found.")
        return

    def print_categorical_summary(self, summary_df, categorical_columns):
        """
        Print data linters for categorical data
        :param summary_df: summary dataframe
        :param categorical_columns: list of columns with categorical values
        :return:
        """
        unique_categories = np.where(summary_df['Categories nr'] >= self.df.shape[0] * 0.98)[0]
        single_category = np.where(summary_df['Categories nr'] == 1)[0]
        avg_long_entries = np.where(summary_df['Avg Token Length'] > 128)[0]
        max_long_entries = np.where(summary_df['Max Token Length'] > 128)[0]
        if any(unique_categories):
            warnings.warn(
                f"Categorical column with unique values: {', '.join([categorical_columns[i] for i in unique_categories])}",
                category=ObjectDataWarning, stacklevel=4)
            self.linter_count[self.linters_dict["obj_unique"]] += len(unique_categories)
        if any(single_category):
            warnings.warn(
                f"Categorical column with constant value: {', '.join([categorical_columns[i] for i in single_category])}",
                category=ObjectDataWarning, stacklevel=4)
            self.linter_count[self.linters_dict["obj_same"]] += len(single_category)
        if any(avg_long_entries):
            warnings.warn(
                f"Categorical column with long entries in average(>128 characters): {', '.join([categorical_columns[i] for i in avg_long_entries])}",
                category=ObjectDataWarning, stacklevel=4)
            self.linter_count[self.linters_dict["obj_long_avg"]] += len(avg_long_entries)
        elif any(max_long_entries):
            warnings.warn(
                f"Categorical column with long entries present (>128 characters): {', '.join([categorical_columns[i] for i in max_long_entries])}",
                category=ObjectDataWarning, stacklevel=4)
            self.linter_count[self.linters_dict["obj_long_max"]] += len(max_long_entries)

    def prepare_categorical_summary_df(self, categorical_columns):
        """
        Prepare table with categorical data statistics
        :param categorical_columns: list of columns included in categorical data summary table
        :return: summary dataframe
        """
        summary_df = self.df[categorical_columns].apply(
            lambda x: [len(x.value_counts()), list(x.value_counts().index),
                       list(x.value_counts().values),
                       np.mean(list(map(lambda y: len(y), list(x.value_counts().index)))),
                       np.max(list(map(lambda y: len(y), list(x.value_counts().index))))], axis=0)
        summary_df = pd.DataFrame(summary_df).T
        summary_df.columns = ['Categories nr', 'Names', 'Counted', 'Avg Token Length', 'Max Token Length']
        summary_df.to_csv(f"{self.result_path}/categorical_summary_df.csv")
        print(f"Categorical summary statistics saved to {self.result_path}/categorical_summary_df.csv.")
        return summary_df

    def run_numeric_validator(self):
        """
        Perform data validation for numeric columns
        """
        print("\nRunning numeric validator...")
        start_time = time()
        numeric_columns = self.df.select_dtypes(include='number').columns.tolist()
        self.missing_values_check(numeric_columns)
        numeric_columns = self.df.select_dtypes(include='number').columns.tolist()
        print("Numeric columns:", numeric_columns)
        if any(numeric_columns):
            summary_df = self.prepare_numeric_summary(numeric_columns)
            self.print_numeric_summary(summary_df)
            end_time = time()
            print(f"Running numeric validator finished in {end_time - start_time:.2f} seconds.")
        else:
            raise Exception("No numeric columns found.")
        return

    def prepare_numeric_summary(self, numeric_columns):
        """
        Prepare data linters for numeric columns
        :param numeric_columns: list of columns with numeric types
        :return:
        """
        summary_df = self.df[numeric_columns].describe()
        skewed_df = pd.DataFrame([self.df[numeric_columns].skew()], index=['skew'])
        additional_quantiles = self.df[numeric_columns].quantile([.01, .05, .1, .9, .95, .99])
        additional_quantiles.index = ['1%', '5%', '10%', '90%', '95%', '99%']
        mode_df = pd.DataFrame(np.array(mode(self.df[numeric_columns], axis=0, nan_policy='omit'), dtype=float),
                               columns=numeric_columns)
        mode_df.index = ['mode', 'mode_ratio']
        mode_df.loc['mode_ratio', :] = mode_df.loc['mode_ratio', :].astype(float)
        mode_df.loc['mode_ratio', :] /= np.round(self.df.shape[0], 3)
        sorted_arr = np.sort(self.df[numeric_columns], axis=0)
        row_differences = np.diff(sorted_arr, axis=0)
        non_0_difference = row_differences.astype(float)
        non_0_difference[non_0_difference == 0] = np.nan
        try:
            max_row_difference = np.nanmax(row_differences, axis=0)
            min_row_difference = np.nanmin(non_0_difference, axis=0)
        except:
            max_row_difference = 0
            min_row_difference = 0
        diff_df = pd.DataFrame([min_row_difference, max_row_difference],
                               columns=numeric_columns, index=['min_row_diff', 'max_row_diff'])
        summary_df = pd.concat([summary_df, additional_quantiles, mode_df, diff_df, skewed_df], axis=0,
                               ignore_index=False,
                               keys=None)
        summary_df = summary_df.round(decimals=2)
        summary_df.to_csv(f"{self.result_path}/numeric_summary_df.csv")
        print(f"Numeric summary statistics saved to {self.result_path}/numeric_summary_df.csv.")
        return summary_df

    def print_numeric_summary(self, summary_df):
        """
        print warnings based on numeric summary statistics
        :param summary_df: numeric summary table
        """
        affected_columns = []
        negative_value_columns = np.where(summary_df.loc['min'] < 0)[0]
        small_percentiles = ['1%', '5%', '10%']
        large_percentiles = ['90%', '95%', '99%']
        small_mode = np.where(summary_df.loc['50%'] <= summary_df.loc[small_percentiles, :])
        large_mode = np.where(summary_df.loc['50%'] >= summary_df.loc[large_percentiles, :])
        skewed_columns = np.where(np.abs(summary_df.loc['skew']) > 2)[0]
        wide_range = \
        np.where(summary_df.loc['max'] - summary_df.loc['min'] > 6 * (summary_df.loc['75%'] - summary_df.loc['25%']))[0]
        high_row_diff = np.where(summary_df.loc['max_row_diff'] > summary_df.loc['std'])[0]
        zero_mode = np.where(summary_df.loc['mode_ratio'] < 0.01)[0]

        if any(negative_value_columns):
            warnings.warn(f"Columns with negative values: {', '.join(summary_df.columns[negative_value_columns])}",
                          category=NumericValuesWarning, stacklevel=4)
            affected_columns.append(negative_value_columns)
            self.linter_count[self.linters_dict["num_negative_values"]] += len(negative_value_columns)
        if any(small_mode[0]):
            small_mode = pd.DataFrame(small_mode).T.groupby(small_mode[1]).min().astype(int)[0]
            for i in pd.unique(small_mode):
                cols_idx = small_mode.iloc[np.where(small_mode == i)[0]].index
                warnings.warn(
                    f"Columns with mode below {small_percentiles[i]}th quantile: {', '.join(summary_df.columns[cols_idx])}",
                    category=NumericValuesWarning, stacklevel=4)
                affected_columns.append(np.array(cols_idx))
                self.linter_count[self.linters_dict["num_small_mode"]] += len(cols_idx)
        if any(large_mode[0]):
            large_mode = pd.DataFrame(large_mode).T.groupby(large_mode[1]).min().astype(int)[0]
            for i in pd.unique(large_mode):
                cols_idx = large_mode.iloc[np.where(large_mode == i)[0]].index
                warnings.warn(
                    f"Columns with mode above {large_percentiles[i]}th quantile: {', '.join(summary_df.columns[cols_idx])}",
                    category=NumericValuesWarning, stacklevel=4)
                affected_columns.append(np.array(cols_idx))
                self.linter_count[self.linters_dict["num_large_mode"]] += len(cols_idx)
        if any(skewed_columns):
            warnings.warn(
                f"Skewed columns: {', '.join(summary_df.columns[skewed_columns])}",
                category=NumericValuesWarning, stacklevel=4)
            affected_columns.append(skewed_columns)
            self.linter_count[self.linters_dict["num_skewness"]] += len(skewed_columns)
        if any(wide_range):
            warnings.warn(f"Columns with wide data range: {', '.join(summary_df.columns[wide_range])}",
                          category=NumericValuesWarning, stacklevel=4)
            affected_columns.append(wide_range)
            self.linter_count[self.linters_dict["num_wide_range"]] += len(wide_range)
        if any(high_row_diff):
            warnings.warn(
                f"Columns with high row difference between two following sorted elements: {', '.join(summary_df.columns[high_row_diff])}",
                category=NumericValuesWarning, stacklevel=4)
            affected_columns.append(high_row_diff)
            self.linter_count[self.linters_dict["num_max_gap"]] += len(high_row_diff)
        if any(zero_mode):
            warnings.warn(
                f"Columns with modal value occurrence less than 1%: {', '.join(summary_df.columns[zero_mode])}",
                category=NumericValuesWarning, stacklevel=4)
            affected_columns.append(zero_mode)
            self.linter_count[self.linters_dict["num_small_mode_cnt"]] += len(zero_mode)
        affected_columns = list(itertools.chain.from_iterable(affected_columns))
        affected_columns = summary_df.columns[affected_columns]
        affected_columns = pd.unique(affected_columns)
        self.visualizer.plot_histograms(affected_columns)

    def run_datetime_validator(self):
        """
        Perform data validation for datetime columns
        """
        print("\nRunning datetime validator...")
        start_time = time()
        date_columns = self.df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
        numeric_columns = self.df.select_dtypes(include=['number']).columns.tolist()
        print("Datetime columns:", date_columns)
        if any(date_columns):
            self.missing_values_check(date_columns)
            for column in date_columns:
                date_interval = max(self.df[column]) - min(self.df[column])
                delta = relativedelta.relativedelta(max(self.df[column]), min(self.df[column]))
                print(f"Range for {column}: {min(self.df[column])} - {max(self.df[column])}")
                date_counts = self.df[column].value_counts()
                print(f"Number of unique days: {len(date_counts)}", end='\n')
                print(f"Maximum number of timestamps per date: {date_counts.iloc[0]}", end='\n')
                self.visualizer.plot_time_series(column, numeric_columns)
                if delta.years >= 20:
                    warnings.warn(f"Long period of datetime data: {delta.years} years", category=DateWarning,
                                  stacklevel=4)
                    self.linter_count[self.linters_dict["time_long"]] += 1
                if self.df[column].is_monotonic_increasing or self.df[column].is_monotonic_decreasing:
                    warnings.warn(f"Datetime column is monotonic - it is probably important", category=DateWarning,
                                  stacklevel=4)
                    self.linter_count[self.linters_dict["time_monotonic"]] += 1
                    self.check_uniform_distribution(column, date_interval)
            end_time = time()
            print(f"Running datetime validator finished in {end_time - start_time:.2f} seconds.")
        else:
            raise Exception("No datetime columns found.")
        return

    def check_uniform_distribution(self, column, date_interval):
        """
        Check uniform distribution of dates
        :param column: column to be checked
        :param date_interval: range of dates
        :return:
        """
        bins = pd.cut(self.df[column].astype('int64'), bins=date_interval.days)
        bin_counts = bins.value_counts()
        chi2, p_value = chisquare(bin_counts)
        if p_value < 0.05:
            warnings.warn("Dates are not uniformly distributed.", category=DateWarning, stacklevel=4)
            self.linter_count[self.linters_dict["time_not_uniform"]] += 1

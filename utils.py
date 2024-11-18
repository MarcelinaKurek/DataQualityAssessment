import pandas as pd
import numpy as np
import re


def read_data(filepath):
    """Function to read data from file in csv format"""
    if filepath.lower().endswith("csv"):
        df = pd.read_csv(filepath)
        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)
        return df
    else:
        raise Exception("File type not supported")


def check_and_correct_types(df):
    """Function to check and correct data types"""
    types_dict = dict(zip(df.columns, df.dtypes))
    df, types_dict = check_if_datetime(df, types_dict)
    return df


def check_if_datetime(df, types_dict):
    """Function to check if an object column is possibly a datetime format"""
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    for column in categorical_columns:
        pd_date_pattern = r"\b\d{2}[-/]\d{2}[-/]\d{4}\b"
        pattern1 = r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{4}(?:\s\d{2}:\d{2}:\d{2})?\b"
        pattern2 = r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:\s\d{2}:\d{2}:\d{2})?\b"
        valid_idx = df[column].first_valid_index()
        if bool(re.match(pattern1, df[column][valid_idx])) or bool(re.match(pattern2, df[column][valid_idx])):
            df[column] = pd.to_datetime(df[column])
            types_dict[column] = np.dtype('datetime64[ns]')
    return df, types_dict

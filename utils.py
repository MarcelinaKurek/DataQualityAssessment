import numpy as np
import pandas as pd
import os
import re


def read_data(filepath):
    """Function to read data from file in csv format"""
    if filepath.lower().endswith("csv"):
        df = pd.read_csv(filepath, low_memory=False)
        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)
        return df
    else:
        raise ValueError("File type not supported")


def check_and_correct_types(df):
    """Function to check and correct data types"""
    types_dict = dict(zip(df.columns, df.dtypes))
    df, types_dict, not_consistent = check_if_datetime(df, types_dict)
    return df, not_consistent


def check_if_datetime(df, types_dict):
    """Function to check if an object column is possibly a datetime format"""
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    not_consistent = False
    for column in categorical_columns:
        pd_date_pattern = r"\b\d{2}[-/]\d{2}[-/]\d{4}\b"
        pattern1 = r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{4}(?:\s\d{2}:\d{2}:\d{2})?\b"
        pattern2 = r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:\s\d{2}:\d{2}:\d{2})?\b"
        valid_idx = df[column].first_valid_index()
        if bool(re.match(pattern1, df[column][valid_idx])) or bool(re.match(pattern2, df[column][valid_idx])):
            try:
                df[column] = pd.to_datetime(df[column])
                types_dict[column] = np.dtype('datetime64[ns]')
            except:
                df[column] = pd.to_datetime(df[column], errors='coerce')
                not_consistent = True
                # raise ValueError("Datetime format might not be consistent")
    return df, types_dict, not_consistent


def make_necessary_folders(data_path):
    """
    Function prepares necessary folders for data check results
    :param data_path: Path to dataset
    :return: name of the created directory
    """
    filepath = data_path.split(".")[0]
    filepath_0 = '/'.join(filepath.split("/")[1:])
    folder_name = f"results/{filepath_0}/tables"
    if not os.path.exists(f"{folder_name}"):
        os.makedirs(f"{folder_name}")
    return folder_name

import numpy as np
import pandas as pd
import re
import os
import pickle
from transformers import BertTokenizer

from data_checks import PandasValidator
from outlier_detection import OutlierDetector
from utils import read_data, check_and_correct_types
from itertools import chain
from collections import Counter
from extract_uci_data import save_dataframe
from pathlib import Path

# Sample data linters
all_files = sorted(os.listdir('data_uci'))
sample_files = ['Adult', 'Apartment_for_Rent_Classified', 'Cylinder_Bands', 'Banknote_Authentication']

for fname in sample_files:
        print(fname)
        data_path = f"data_uci/{fname}/data.csv"
        df = read_data(data_path)
        df, not_consistent = check_and_correct_types(df)
        pvalid = PandasValidator(df, data_path)
        date_columns = pvalid.df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
        numeric_columns = pvalid.df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = pvalid.df.select_dtypes(include=['object']).columns.tolist()
        if categorical_columns:
            pvalid.run_categorical_validator()
        if numeric_columns:
            pvalid.run_numeric_validator()
        if date_columns:
            pvalid.run_datetime_validator()
        print("\n")


# Sample anomaly detection
data_path = r"kaggle_datasets/new-york-housing-market/NY-House-Dataset.csv"

if_parameters_dict = {"n_estimators":100, "max_samples":'auto', "contamination":0.01, "random_state":np.random.RandomState(42)}
knn_parameters_dict = {"n_neighbors":5}
lof_parameters_dict = {"n_neighbors":20, "contamination":0.01}
dbscan_parameters_dict = {"eps":'knn', "min_samples":"n_features"}


df = read_data(data_path)
df, not_consistent = check_and_correct_types(df)
od = OutlierDetector(df, data_path)
od.run_knn(**knn_parameters_dict, outlier_ratio=.01)
od.run_isolation_forest(**if_parameters_dict)
od.run_local_outlier_factor(**lof_parameters_dict)
od.run_dbscan(**dbscan_parameters_dict)
od.save_summary_table()


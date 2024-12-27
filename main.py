import numpy as np
import pandas as pd
import re

from transformers import BertTokenizer

from data_checks import PandasValidator
from outlier_detection import OutlierDetector
from utils import read_data, check_and_correct_types


data_path = r"data_uci/Adult/data.csv"
# data_path = r"kaggle_datasets/stock-market-dataset/stocks/A.csv"
# data_path = r"data_uci/Automobile/data.csv"

df = read_data(data_path)
df, not_consistent = check_and_correct_types(df)


# pvalid = PandasValidator(df, data_path)
# pvalid.run_categorical_validator()
# pvalid.run_numeric_validator()
# print(np.arange(21))
# print(pvalid.linter_count)

# pvalid.run_datetime_validator()

if_parameters_dict = {"n_estimators":100, "max_samples":'auto', "contamination":0.01, "random_state":np.random.RandomState(42)}
knn_parameters_dict = {"n_neighbors":10}
lof_parameters_dict = {"n_neighbors":10, "contamination":0.01}
dbscan_parameters_dict = {"eps":2, "min_samples":5}
#
#
od = OutlierDetector(df, data_path)
od.calcualate_z_score()
od.run_knn(**knn_parameters_dict)
# od.run_isolation_forest(**if_parameters_dict)
# od.run_local_outlier_factor(**lof_parameters_dict)
od.run_dbscan(**dbscan_parameters_dict)
# od.save_summary_table()
# od.visualizer.plot_outliers_pca_all(od.df_preprocessed, od.outliers_dict)
# od.visualizer.plot_outliers_umap_all(od.df_preprocessed, od.outliers_dict)

import numpy as np
import pandas as pd
import os

from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.compose import make_column_transformer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from time import time

from visualizer import Visualizer
from utils import make_necessary_folders

class OutlierDetector:
    def __init__(self, df, data_path):
        """
        Initialize the outlier detector
        :param df: input dataframe
        :param data_path: path to data folder
        """
        self.df_preprocessed = None
        self.df = df
        self.numeric_columns = None
        self.visualizer = Visualizer(data_path, self.df)
        self.preprocess()
        self.outlier_summary_df = None
        self.result_path = make_necessary_folders(data_path)
        self.outliers_dict = {}
        self.elbow_point = None

    def preprocess(self):
        """Preprocessing for outlier detection. It applies StandardScaler for numeric columns,
        OrdinalEncoder for columns with basic levels and OneHotEncoder for other categorical columns.
        Columns with more than six categories are not considered.
        """
        print("\nPreprocessing for outlier detection started...")
        start_time = time()
        categorical_columns = self.df.select_dtypes(include='object').columns.tolist()
        numeric_columns = self.df.select_dtypes(include='number').columns.tolist()
        ohe_columns = []
        for column in categorical_columns:
            vals = pd.unique(self.df[column])
            vals = [str(v).lower() for v in vals]
            if len(self.df.loc[:,column].value_counts()) <= 6:
                #column to be preprocessed by OneHotEncoder
                ohe_columns.append(column)

        #apply transformation
        column_trans = make_column_transformer((OneHotEncoder(handle_unknown='ignore'), ohe_columns),
                                               (StandardScaler(), numeric_columns))
        X = column_trans.fit_transform(self.df)
        self.df_preprocessed = pd.DataFrame(X, columns=column_trans.get_feature_names_out())
        self.df_preprocessed = self.df_preprocessed.dropna(axis=0, how='any')
        end_time = time()
        print(f"Preprocessing done in {round(end_time - start_time, 2)} seconds")
        print(f"One hot encoded columns: {ohe_columns}\nStandard scaled columns: {numeric_columns}")
        return

    def calcualate_z_score(self, threshold=3):
        """
        Calculates Z-score for each numeric column separately.
        :param threshold: Z-score threshold to be considered as outlier
        :return:
        """
        print("\nCalculating Z-score started...")
        start_time = time()
        self.numeric_columns = self.df.select_dtypes(include='number').columns.tolist()
        # Calculate Z-score
        mean_scores = np.mean(self.df[self.numeric_columns], axis=0)
        sd_scores = np.std(self.df[self.numeric_columns], axis=0)
        Z_scores = np.abs((self.df[self.numeric_columns] - mean_scores) / sd_scores)
        outlier_indices = np.where(Z_scores > threshold)
        end_time = time()
        self.prepare_summary_table(outlier_indices)
        print(f"Calculating Z-score finished in {round(end_time - start_time, 2)} seconds")
        return

    def prepare_summary_table(self, outlier_indices):
        outliers_df = pd.DataFrame(outlier_indices[1]).value_counts().sort_index().reset_index()
        outliers_df.columns = ["Column name", "Number of outliers"]
        outliers_df['Column name'] = np.array(self.numeric_columns)[outliers_df['Column name']]
        outliers_df['Outliers ratio'] = outliers_df['Number of outliers'] / self.df.shape[0]
        outliers_mean = outliers_df['Number of outliers'].mean()
        outliers_percent_mean = outliers_df['Outliers ratio'].mean()
        self.outlier_summary_df = outliers_df._append({"Column name": "Average", "Number of outliers": outliers_mean,
                                                       "Outliers ratio": outliers_percent_mean}, ignore_index=True).fillna(0)

    def update_summary_table(self, outlier_dict):
        self.outlier_summary_df = self.outlier_summary_df._append(outlier_dict, ignore_index=True)

    def run_isolation_forest(self, **kwargs):
        """
        Running Isolation Forest outlier detection.
        :param kwargs: Keyword arguments for Isolation Forest model
        :return:
        """
        print("\nRunning Isolation Forest started...")
        start_time = time()
        model = IsolationForest(**kwargs)
        model.fit(self.df_preprocessed)
        y = model.predict(self.df_preprocessed)
        end_time = time()
        outliers = np.where(y == -1)[0]
        print(f"Isolation Forest outlier ratio: {(len(np.where(y == -1)[0]) / self.df.shape[0]) * 100:.2f}% ({len(np.where(y == -1)[0])} outliers found)")
        self.visualizer.plot_outliers_pca(self.df_preprocessed, outliers, algorithm_name="Isolation Forest")
        self.visualizer.plot_outliers_umap(self.df_preprocessed, outliers, algorithm_name="Isolation Forest")
        outlier_dict = {"Column name": "Isolation Forest", "Number of outliers": len(outliers),
             "Outliers ratio": len(outliers) / self.df.shape[0]}
        self.update_summary_table(outlier_dict)
        self.outliers_dict["Isolation Forest"] = outliers
        print(f"Running Isolation Forest finished in {round(end_time - start_time, 2)} seconds")
        return

    def run_knn(self, outlier_ratio=.01, **kwargs):
        """
        Running K-Nearest Neighbor outlier detection.
        :param outlier_ratio: ratio of expected outliers
        :param kwargs: Keyword arguments for K-Nearest Neighbor model
        :return:
        """
        print("\nRunning K-Nearest Neighbors...")
        start_time = time()
        knn = NearestNeighbors(**kwargs)
        knn.fit(self.df_preprocessed.dropna())
        distances, indexes = knn.kneighbors(self.df_preprocessed.dropna())
        end_time = time()
        distances_mean = np.mean(distances, axis=1)
        outliers = np.where(distances_mean > np.percentile(distances_mean, 100-(outlier_ratio*100)))[0]
        distances_sorted = np.sort(distances_mean)
        kneedle = KneeLocator(np.arange(len(distances_sorted)), distances_sorted, curve="convex", direction="increasing")
        knee_point = distances_sorted[kneedle.knee]
        self.elbow_point = np.round(knee_point, 1)
        print(
            f"KNN outlier ratio: {(len(outliers) / self.df.shape[0]) * 100:.2f}% ({len(outliers)} outliers found)")
        self.visualizer.plot_outliers_pca(self.df_preprocessed, outliers, algorithm_name="Nearest Neighbors")
        self.visualizer.plot_outliers_umap(self.df_preprocessed, outliers, algorithm_name="Nearest Neighbors")
        knn_outlier_dict = {"Column name": "KNN", "Number of outliers": len(outliers),
             "Outliers ratio": len(outliers) / self.df.shape[0]}
        self.update_summary_table(knn_outlier_dict)
        self.outliers_dict["Nearest Neighbors"] = outliers
        print(f"Running K-Nearest Neighbor done in {round(end_time - start_time, 2)} seconds")
        return

    def run_local_outlier_factor(self, **kwargs):
        """
        Running Local Outlier Factor outlier detection.
        :param kwargs: Keyword arguments for Local Outlier Factor model
        :return:
        """
        print("\nRunning Local Outlier Factor...")
        start_time = time()
        lof = LocalOutlierFactor(**kwargs)
        scores = lof.fit_predict(self.df_preprocessed)
        end_time = time()
        outliers = np.where(scores == -1)[0]
        print(f"Local outlier factor outlier ratio: {(len(np.where(scores == -1)[0]) / self.df.shape[0]) * 100:.2f}% ({len(np.where(scores == -1)[0])} outliers found)")
        self.visualizer.plot_outliers_pca(self.df_preprocessed, outliers, algorithm_name="Local Outlier Factor")
        self.visualizer.plot_outliers_umap(self.df_preprocessed, outliers, algorithm_name="Local Outlier Factor")
        lof_dict = {"Column name": "Local Outlier Factor", "Number of outliers": len(outliers),
             "Outliers ratio": len(outliers) / self.df.shape[0]}
        self.update_summary_table(lof_dict)
        self.outliers_dict["Local Outlier Factor"] = outliers
        print(f"Running Local Outlier Factor done in {round(end_time - start_time, 2)} seconds ")
        return

    def run_dbscan(self, **kwargs):
        """
        Running DBSCAN outlier detection.
        :param kwargs: Keyword arguments for DBSCAN model
        :return:
        """
        print("\nRunning DBSCAN...")

        # Sort distances for the k-distance plot
        start_time = time()
        if self.elbow_point != None:
            kwargs["eps"] = self.elbow_point
            kwargs["min_samples"] = self.df_preprocessed.shape[1]
        print(kwargs)
        dbscan = DBSCAN(**kwargs)
        dbscan.fit(self.df_preprocessed)
        end_time = time()
        scores = dbscan.labels_
        outliers = np.where(scores == -1)[0]
        print(f"DBSACAN outlier ratio: {(len(np.where(scores == -1)[0]) / self.df.shape[0]) * 100:.2f}% ({len(np.where(scores == -1)[0])} outliers found)")
        self.visualizer.plot_outliers_pca(self.df_preprocessed, outliers, algorithm_name="DBSCAN")
        self.visualizer.plot_outliers_umap(self.df_preprocessed, outliers, algorithm_name="DBSCAN")
        dbscan_dict = {"Column name": "DBSCAN", "Number of outliers": len(outliers),
             "Outliers ratio": len(outliers) / self.df.shape[0]}
        self.update_summary_table(dbscan_dict)
        self.outliers_dict["DBSCAN"] = outliers
        print(f"Running DBSCAN done in {round(end_time - start_time, 2)} seconds")
        return

    def save_summary_table(self):
        self.outlier_summary_df = np.round(self.outlier_summary_df, 2)
        self.outlier_summary_df.to_csv(f"{self.result_path}/outlier_summary_df.csv")
        print(f"Summary table saved to {self.result_path}/outlier_summary_df.csv")
        return



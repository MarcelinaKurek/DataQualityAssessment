import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
from visualizer import Visualizer
from sklearn.compose import make_column_transformer
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN


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

    def preprocess(self):
        """Preprocessing for outlier detection. It applies StandardScaler for numeric columns,
        OrdinalEncoder for columns with basic levels and OneHotEncoder for other categorical columns.
        Columns with more than six categories are not considered.
        """
        print("\nPreprocessing for outlier detection started...")
        categorical_columns = self.df.select_dtypes(include='object').columns.tolist()
        numeric_columns = self.df.select_dtypes(include='number').columns.tolist()
        ohe_columns = []
        le_columns = []
        for column in categorical_columns:
            vals = pd.unique(self.df[column])
            vals = [str(v).lower() for v in vals]
            if 'low' in vals or 'medium' in vals or 'high' in vals:
                #column to be preprocessed by Ordinal Encoder
                le_columns.append(column)
            else:
                if len(self.df.loc[:,column].value_counts()) <= 6:
                    #column to be preprocessed by OneHotEncoder
                    ohe_columns.append(column)

        #apply transformation
        column_trans = make_column_transformer((OneHotEncoder(handle_unknown='ignore'), ohe_columns),
                                               (OrdinalEncoder(), le_columns),
                                               (StandardScaler(), numeric_columns))
        X = column_trans.fit_transform(self.df)
        self.df_preprocessed = pd.DataFrame(X, columns=column_trans.get_feature_names_out())
        print("Preprocessing done.")
        print(f"One hot encoded columns: {ohe_columns}\nOrdinal encoded columns: {le_columns}\nStandard scaled columns: {numeric_columns}")
        return

    def calcualate_z_score(self, threshold=3):
        """
        Calculates Z-score for each numeric column separately.
        :param threshold: Z-score threshold to be considered as outlier
        :return:
        """
        print("\nCalculating Z-score...")
        self.numeric_columns = self.df.select_dtypes(include='number').columns.tolist()
        # Calculate Z-score
        mean_scores = np.mean(self.df[self.numeric_columns], axis=0)
        sd_scores = np.std(self.df[self.numeric_columns], axis=0)
        Z_scores = np.abs((self.df[self.numeric_columns] - mean_scores) / sd_scores)
        outlier_indices = np.where(Z_scores > threshold)
        self.prepare_summary_table(outlier_indices)
        return

    def prepare_summary_table(self, outlier_indices):
        outliers_df = pd.DataFrame(outlier_indices[1]).value_counts().sort_index().reset_index()
        outliers_df.columns = ["Column name", "Number of outliers"]
        outliers_df['Column name'] = self.numeric_columns
        outliers_df['Outliers percent'] = outliers_df['Number of outliers'] / self.df.shape[0]
        outliers_mean = outliers_df['Number of outliers'].mean()
        outliers_percent_mean = outliers_df['Outliers percent'].mean()
        self.outlier_summary_df = outliers_df._append({"Column name": "Average", "Number of outliers": outliers_mean,
                                                       "Outliers percent": outliers_percent_mean}, ignore_index=True)

    def update_summary_table(self, outlier_dict):
        self.outlier_summary_df = self.outlier_summary_df._append(outlier_dict, ignore_index=True)

    def run_isolation_forest(self, **kwargs):
        """
        Running Isolation Forest outlier detection.
        :param kwargs: Keyword arguments for Isolation Forest model
        :return:
        """
        print("\nRunning Isolation Forest...")
        model = IsolationForest(**kwargs)
        model.fit(self.df_preprocessed)
        y = model.predict(self.df_preprocessed)
        outliers = np.where(y == -1)[0]
        print(f"Isolation Forest outlier ratio: {(len(np.where(y == -1)[0]) / self.df.shape[0]) * 100:.2f}% ({len(np.where(y == -1)[0])} outliers found)")
        self.visualizer.plot_outliers_pca(self.df_preprocessed, outliers, algorithm_name="Isolation Forest")
        self.visualizer.plot_outliers_umap(self.df_preprocessed, outliers, algorithm_name="Isolation Forest")
        outlier_dict = {"Column name": "Isolation Forest", "Number of outliers": len(outliers),
             "Outliers percent": len(outliers) / self.df.shape[0]}
        self.update_summary_table(outlier_dict)

        print("Running Isolation Forest done.")
        return

    def run_knn(self, **kwargs):
        """
        Running K-Nearest Neighbor outlier detection.
        :param kwargs: Keyword arguments for K-Nearest Neighbor model
        :return:
        """
        print("\nRunning K-Nearest Neighbors...")
        knn = NearestNeighbors(**kwargs)
        knn.fit(self.df_preprocessed)
        distances, indexes = knn.kneighbors(self.df_preprocessed)
        distances = pd.DataFrame(distances)
        distances_mean = distances.mean(axis=1)
        outliers = np.where(distances_mean > np.percentile(distances_mean, 99))[0]
        self.visualizer.plot_outliers_pca(self.df_preprocessed, outliers, algorithm_name="Nearest Neighbors")
        self.visualizer.plot_outliers_umap(self.df_preprocessed, outliers, algorithm_name="Nearest Neighbors")
        knn_outlier_dict = {"Column name": "KNN", "Number of outliers": len(outliers),
             "Outliers percent": len(outliers) / self.df.shape[0]}
        self.update_summary_table(knn_outlier_dict)
        print("Running K-Nearest Neighbor done.")
        return

    def run_local_outlier_factor(self, **kwargs):
        """
        Running Local Outlier Factor outlier detection.
        :param kwargs: Keyword arguments for Local Outlier Factor model
        :return:
        """
        print("\nRunning Local Outlier Factor...")
        lof = LocalOutlierFactor(**kwargs)
        scores = lof.fit_predict(self.df_preprocessed)
        outliers = np.where(scores == -1)[0]
        print(f"Local outlier factor outlier ratio: {(len(np.where(scores == -1)[0]) / self.df.shape[0]) * 100:.2f}% ({len(np.where(scores == -1)[0])} outliers found)")
        self.visualizer.plot_outliers_pca(self.df_preprocessed, outliers, algorithm_name="Local Outlier Factor")
        self.visualizer.plot_outliers_umap(self.df_preprocessed, outliers, algorithm_name="Local Outlier Factor")
        lof_dict = {"Column name": "Local Outlier Factor", "Number of outliers": len(outliers),
             "Outliers percent": len(outliers) / self.df.shape[0]}
        self.update_summary_table(lof_dict)
        print("Running Local Outlier Factor done.")
        return

    def run_dbscan(self, **kwargs):
        """
        Running DBSCAN outlier detection.
        :param kwargs: Keyword arguments for DBSCAN model
        :return:
        """
        print("\nRunning DBSCAN...")
        dbscan = DBSCAN(**kwargs)
        dbscan.fit(self.df_preprocessed)
        scores = dbscan.labels_
        outliers = np.where(scores == -1)[0]
        print(f"DBSACAN outlier ratio: {(len(np.where(scores == -1)[0]) / self.df.shape[0]) * 100:.2f}% ({len(np.where(scores == -1)[0])} outliers found)")
        self.visualizer.plot_outliers_pca(self.df_preprocessed, outliers, algorithm_name="DBSCAN")
        self.visualizer.plot_outliers_umap(self.df_preprocessed, outliers, algorithm_name="DBSCAN")
        dbscan_dict = {"Column name": "DBSCAN", "Number of outliers": len(outliers),
             "Outliers percent": len(outliers) / self.df.shape[0]}
        self.update_summary_table(dbscan_dict)
        print("Running DBSCAN done.")
        return



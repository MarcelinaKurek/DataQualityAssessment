import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import pandas as pd
import umap.umap_ as umap

from collections import Counter
from itertools import chain
from sklearn.decomposition import PCA

plt.style.use('seaborn-v0_8-whitegrid')


class Visualizer:
    def __init__(self, result_path, df):
        self.counter = 0
        self.result_path = result_path
        self.df = df
        self.result_path = self.make_necessary_folders(result_path)

    @staticmethod
    def make_necessary_folders(data_path):
        """
        Make folders for saving visualizations
        """
        filepath = data_path.split(".")[0]
        filepath_0 = '/'.join(filepath.split("/")[1:])
        folder_name = f"results/{filepath_0}"
        if not os.path.exists(f"{folder_name}/plots_general"):
            os.makedirs(f"{folder_name}/plots_general")
        if not os.path.exists(f"{folder_name}/plots_outliers"):
            os.makedirs(f"{folder_name}/plots_outliers")
        return folder_name

    def plot_categories_count(self, categorical_columns, horizontal=False):
        """
        Plot barplot with categories count
        :param categorical_columns: list of columns for categories count plots (each column has a separate plot)
        :param horizontal: plot orientation
        """
        for categorical_column in categorical_columns:
            vals = self.df[categorical_column].value_counts()
            plt.figure()
            if horizontal:
                vals = vals.sort_values(ascending=True)
                plt.barh(vals.index, vals)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
            else:
                plt.bar(vals.index, vals)
                plt.xticks(rotation=30, ha='right', fontsize=14)
                plt.yticks(fontsize=14)
            plt.title(categorical_column, fontsize=16)
            plt.tight_layout()
            if '/' or '\\' in categorical_column:
                categorical_column = categorical_column.replace('/', '_or_').replace('\\', '_')
            plt.savefig(f"{self.result_path}/plots_general/{categorical_column}_count.png")
        return

    def plot_histograms(self, numeric_columns):
        """
        Plot histograms
        :param numeric_columns: list of numeric columns for histogram plots (each column has a separate plot)
        """
        for numeric_column in numeric_columns:
            plt.figure()
            vals = self.df[numeric_column].values
            plt.hist(vals)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.title(f"{numeric_column} - histogram", fontsize=16)
            if '/' or '\\' or '?' in numeric_column:
                numeric_column = numeric_column.replace('/', '_or_').replace('\\', '_').replace("?",'_')
            plt.savefig(f"{self.result_path}/plots_general/{numeric_column}_hist.png")
        return

    def plot_time_series(self, date_column, numeric_columns):
        """
        Plot time series with datetime column on x-axis and numeric columns on y-axis
        """
        for numeric_column in numeric_columns:
            plt.figure()
            vals = self.df[numeric_column].values
            plt.plot(self.df[date_column], vals)
            plt.title(f"{numeric_column}", fontsize=14)
            plt.savefig(f"{self.result_path}/plots_general/{numeric_column}_plot.png")

    def plot_outliers_pca(self, df_preprocessed, outliers, algorithm_name=""):
        """
        Plot outliers using PCA dimensionality reduction
        :param df_preprocessed: dataframe to fit PCA dimensionality reduction
        :param outliers: outliers indices
        :param algorithm_name: algorithm name (included in the plot title)
        :return:
        """
        pca = PCA(2)
        pca.fit(df_preprocessed)
        res = pd.DataFrame(pca.transform(df_preprocessed))
        figsize = (12, 7)
        plt.figure(figsize=figsize)
        plt.title(f"{algorithm_name} - PCA")
        plt.scatter(res[0], res[1], c='blue', s=40, label="normal points")
        plt.scatter(res.iloc[outliers, 0], res.iloc[outliers, 1], c='red',
                    s=40, edgecolor="red", label="predicted outliers")
        plt.legend(loc="upper right")
        plt.savefig(f"{self.result_path}/plots_outliers/{algorithm_name}_pca.png")
        return

    def plot_outliers_umap(self, df_preprocessed, outliers, algorithm_name=""):
        # implemented but not used in the thesis
        reducer = umap.UMAP(n_neighbors=30)
        res = reducer.fit_transform(df_preprocessed)
        figsize = (12, 7)
        plt.figure(figsize=figsize)
        plt.title(f"{algorithm_name} - UMAP", fontsize=17)
        plt.scatter(res[:, 0], res[:, 1], c='blue', s=40, label="normal points")
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.scatter(res[outliers, 0], res[outliers, 1], c='red',
                    s=40, edgecolor="red", label="predicted outliers")

        plt.legend(loc="upper right", fontsize=13)
        plt.savefig(f"{self.result_path}/plots_outliers/{algorithm_name}_umap.png")

    def plot_outliers_pca_all(self, df_preprocessed, outliers_dict):
        """
        Plot outliers using PCA dimensionality reduction for 4 algorithms (Isolation Forest, KNN, Local Outlier Factor,
        DBSCAN)
        :param df_preprocessed: dataframe after preprocessing to fit PCA dimensionality reduction
        :param outliers_dict: dictionary of outliers indices
        :return:
        """
        pca = PCA(2)
        pca.fit(df_preprocessed)
        res = pd.DataFrame(pca.transform(df_preprocessed))
        figsize = (12, 9)
        all_values_flat = list(chain.from_iterable(outliers_dict.values()))
        all_outliers_count = Counter(all_values_flat)
        df = pd.DataFrame(all_outliers_count.items(), columns=['Outlier', 'Outlier_count'])
        colors = ["#ffcccc", "#ff6666", "#ff0000", "#990000"]  # Light red to dark red
        plt.figure(figsize=figsize)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.title(f"Wyniki detekcji anomalii - PCA", fontsize=17)
        plt.scatter(res[0], res[1], c='blue', s=40, label="normalny punkt", edgecolor="none")
        main_labels = ["anomalia na podstawie predykcji 1 algorytmu", "anomalia na podstawie predykcji 2 algorytmów",
                       "anomalia na podstawie predykcji 3 algorytmów", "anomalia na podstawie predykcji 4 algorytmów"]
        for i in range(4):
            outlier_idx = list(df.loc[df['Outlier_count'] == i+1, 'Outlier'])
            # plt.scatter(res.iloc[outlier_idx, 0], res.iloc[outlier_idx, 1], c=colors[i],
            #             s=40, edgecolor="none", label=f"outlier predicted by {i+1} algorithm(s)")
            plt.scatter(res.iloc[outlier_idx, 0], res.iloc[outlier_idx, 1], c=colors[i],
                        s=40, edgecolor="none", label=main_labels[i])
            if i == 3:
                for j, label in enumerate(outlier_idx):
                    plt.annotate(label, (res.iloc[outlier_idx[j], 0], res.iloc[outlier_idx[j], 1]),
                                 textcoords="offset points", xytext=(0,5), ha='center', size=12)
        plt.xlabel("I PCA component", fontsize=15)
        plt.ylabel("II PCA component", fontsize=15)
        plt.legend(fontsize=13, loc="upper right", bbox_to_anchor=(1, -0.1))
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        plt.savefig(f"{self.result_path}/plots_outliers/outliers_pca.png")
        return

    def plot_outliers_umap_all(self, df_preprocessed, outliers_dict):
        # implemented but not used in the thesis
        reducer = umap.UMAP(n_neighbors=30)
        res = pd.DataFrame(reducer.fit_transform(df_preprocessed))
        all_values_flat = list(chain.from_iterable(outliers_dict.values()))
        all_outliers_count = Counter(all_values_flat)
        df = pd.DataFrame(all_outliers_count.items(), columns=['Outlier', 'Outlier_count'])
        colors = ["#ffcccc", "#ff6666", "#ff0000", "#990000"]  # Light red to dark red
        colors = ["#f88379", "#ff6347", "#dc143c", "#800020"]
        custom_cmap = mcolors.ListedColormap(colors)
        figsize = (12, 7)
        plt.figure(figsize=figsize)
        plt.title(f"Outliers summary - UMAP", fontsize=17)
        plt.scatter(res.iloc[:, 0], res.iloc[:, 1], c='blue', s=40, label="normal points", edgecolor="none")
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        for i in range(4):
            outlier_idx = list(df.loc[df['Outlier_count'] == i+1, 'Outlier'])
            plt.scatter(res.iloc[outlier_idx, 0], res.iloc[outlier_idx, 1], c=colors[i],
                        s=40, edgecolor="none", label=f"outlier predicted by {i+1} algorithm(s)")
            if i == 3:
                for j, label in enumerate(outlier_idx):
                    plt.annotate(label, (res.iloc[outlier_idx[j], 0], res.iloc[outlier_idx[j], 1]), textcoords="offset points", xytext=(0, 5), ha='center')
        plt.legend(fontsize=13)
        plt.savefig(f"{self.result_path}/plots_outliers/outliers_umap.png")

    def plot_k_distance(self, distances, knee_point):
        """
        Plot k distance for choosing an optimal epsilon value in DBSCAN algorithm
        :param distances: average distance to k nearest neighbors
        :param knee_point: optimal epsilon value according to heuristics
        """
        distances_sorted = np.sort(distances)
        plt.figure(figsize=(12, 7))
        plt.plot(distances_sorted, linewidth=2)
        plt.scatter(knee_point, distances_sorted[knee_point], c='red', label='punkt łokciowy')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.title(f"Wykres odległości do k najbliższych sąsiadów (k={2*self.df.shape[1]-1})", fontsize=18)
        plt.xlabel('Obserwacje posortowane według odległości do k sąsiadów', fontsize=16)
        plt.ylabel('średnia odległość do k sąsiadów', fontsize=16)
        plt.legend(fontsize=15)
        plt.savefig("sample_k_distance.png")


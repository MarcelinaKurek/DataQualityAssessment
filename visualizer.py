import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import umap.umap_ as umap
from itertools import chain
from collections import Counter
import matplotlib.colors as mcolors
import numpy as np
plt.style.use('seaborn-v0_8-whitegrid')

class Visualizer:
    def __init__(self, result_path, df):
        self.counter = 0
        self.result_path = result_path
        self.df = df
        self.result_path = self.make_necessary_folders(result_path)

    @staticmethod
    def make_necessary_folders(data_path):
        filepath = data_path.split(".")[0]
        filepath_0 = '/'.join(filepath.split("/")[1:])
        folder_name = f"results/{filepath_0}"
        if not os.path.exists(f"{folder_name}/plots_general"):
            os.makedirs(f"{folder_name}/plots_general")
        if not os.path.exists(f"{folder_name}/plots_outliers"):
            os.makedirs(f"{folder_name}/plots_outliers")
        return folder_name

    def plot_categories_count(self, categorical_columns, horizontal=False):
        for categorical_column in categorical_columns:
            vals = self.df[categorical_column].value_counts()
            plt.figure()
            if horizontal:
                vals = vals.sort_values(ascending=True)
                plt.barh(vals.index, vals)
            else:
                plt.bar(vals.index, vals)
                plt.xticks(rotation=30, ha='right')
            plt.title(categorical_column)
            plt.tight_layout()
            if '/' in categorical_column:
                categorical_column = categorical_column.replace('/', '_or_')
            plt.savefig(f"{self.result_path}/plots_general/{categorical_column}_count.png")
        return

    def plot_histograms(self, numeric_columns):
        for numeric_column in numeric_columns:
            plt.figure()
            vals = self.df[numeric_column].values
            plt.hist(vals)
            plt.title(f"{numeric_column} - histogram")
            plt.savefig(f"{self.result_path}/plots_general/{numeric_column}_hist.png")
        return

    def plot_time_series(self, date_column, numeric_columns):
        for numeric_column in numeric_columns:
            plt.figure()
            vals = self.df[numeric_column].values
            plt.plot(self.df[date_column], vals)
            plt.title(f"{numeric_column}")
            plt.savefig(f"{self.result_path}/plots_general/{numeric_column}_plot.png")

    def plot_outliers_pca(self, df_preprocessed, outliers, algorithm_name=""):
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
        reducer = umap.UMAP(n_neighbors=30)
        res = reducer.fit_transform(df_preprocessed)
        figsize = (12, 7)
        plt.figure(figsize=figsize)
        plt.title(f"{algorithm_name} - UMAP")
        plt.scatter(res[:, 0], res[:, 1], c='blue', s=40, label="normal points")
        plt.scatter(res[outliers, 0], res[outliers, 1], c='red',
                    s=40, edgecolor="red", label="predicted outliers")
        plt.legend(loc="upper right")
        plt.savefig(f"{self.result_path}/plots_outliers/{algorithm_name}_umap.png")

    def plot_outliers_pca_all(self, df_preprocessed, outliers_dict):
        pca = PCA(2)
        pca.fit(df_preprocessed)
        res = pd.DataFrame(pca.transform(df_preprocessed))
        figsize = (12, 7)
        all_values_flat = list(chain.from_iterable(outliers_dict.values()))
        all_outliers_count = Counter(all_values_flat)
        df = pd.DataFrame(all_outliers_count.items(), columns=['Outlier', 'Outlier_count'])
        colors = ["#ffcccc", "#ff6666", "#ff0000", "#990000"]  # Light red to dark red
        # colors = ["#f88379", "#ff6347", "#dc143c", "#800020"]
        custom_cmap = mcolors.ListedColormap(colors)
        plt.figure(figsize=figsize)
        plt.title(f"Outliers summary - PCA")
        plt.scatter(res[0], res[1], c='blue', s=40, label="normal points")
        for i in range(4):
            outlier_idx = list(df.loc[df['Outlier_count'] == i+1, 'Outlier'])
            plt.scatter(res.iloc[outlier_idx, 0], res.iloc[outlier_idx, 1], c=colors[i],
                        s=40, edgecolor="none", label=f"outlier predicted by {i+1} algorithm(s)")
        plt.legend()
        plt.savefig(f"{self.result_path}/plots_outliers/outliers_pca.png")
        return

    def plot_outliers_umap_all(self, df_preprocessed, outliers_dict):
        reducer = umap.UMAP(n_neighbors=30)
        res = reducer.fit_transform(df_preprocessed)
        all_values_flat = list(chain.from_iterable(outliers_dict.values()))
        all_outliers_count = Counter(all_values_flat)
        df = pd.DataFrame(all_outliers_count.items(), columns=['Outlier', 'Outlier_count'])
        colors = ["#ffcccc", "#ff6666", "#ff0000", "#990000"]  # Light red to dark red
        colors = ["#f88379", "#ff6347", "#dc143c", "#800020"]
        custom_cmap = mcolors.ListedColormap(colors)
        figsize = (12, 7)
        plt.figure(figsize=figsize)
        plt.title(f"Outliers summary - UMAP")
        plt.scatter(res[:, 0], res[:, 1], c='blue', s=40, label="normal points")
        for i in range(4):
            outlier_idx = list(df.loc[df['Outlier_count'] == i+1, 'Outlier'])
            plt.scatter(res[outlier_idx, 0], res[outlier_idx, 1], c=colors[i],
                        s=40, edgecolor="none", label=f"outlier predicted by {i+1} algorithm(s)")
        plt.legend()
        plt.savefig(f"{self.result_path}/plots_outliers/outliers_umap.png")

    def plot_k_distance(self, distances):
        print(distances)
        distances_sorted = np.sort(distances)
        plt.plot(distances_sorted)
        plt.title('k-distance plot')
        plt.xlabel('Data Points sorted by distance')
        plt.ylabel('k-distance')
        plt.show()

import sys
import time

import numpy as np
import pandas as pd
from outlier_detection import OutlierDetector



if __name__ == "__main__":
  print(sys.argv[0])
  if_parameters_dict = {"n_estimators": 100, "max_samples": 'auto', "contamination": 0.01,
                        "random_state": np.random.RandomState(42)}
  knn_parameters_dict = {"n_neighbors": 5}
  lof_parameters_dict = {"n_neighbors": 20, "contamination": 0.01}
  dbscan_parameters_dict = {"eps": 1.5, "min_samples": 4}

  res = []
  for i in range(10):
    print(i)
    N = 20_000 * (i+1)
    col1 = np.random.randint(1, 1_000_000_000, size = N)
    col2 = np.random.normal(size = N)

    additional_cols = {
      f"int{i}": np.random.randint(1, 1_000_000_000, size=N) for i in range(3, 7)
    }
    additional_cols.update({
      f"float{i}": np.random.normal(size=N) for i in range(7, 11)
    })

    df = pd.DataFrame({
      "int1": col1,
      "int2": col2,
      **additional_cols
    })

    data_path = r"data_uci/test/data.csv"
    od = OutlierDetector(df, data_path)
    z_time = od.calcualate_z_score()
    knn_time = od.run_knn(**knn_parameters_dict, outlier_ratio=.01)
    if_time = od.run_isolation_forest(**if_parameters_dict)
    lof_time= od.run_local_outlier_factor(**lof_parameters_dict)
    print("Nr od samples", N)
    # outliers_sklearn, time_sklearn = od.run_dbscan(implementation='sklearn', **dbscan_parameters_dict)
    outliers_dbscan, time_dbscan = od.run_dbscan(implementation='dbscan', **dbscan_parameters_dict)
    # print("Times: ", time_sklearn, time_dbscan)
    res.append([z_time, knn_time, if_time, lof_time, time_dbscan])
  res = pd.DataFrame(res, columns=["z_time", "knn_time", "if_time", "lof_time", "dbscan_time"])
  res.to_csv("outliers_time_10_20k-200k.csv")


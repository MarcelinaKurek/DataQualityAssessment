import sys
import time

import numpy as np
import pandas as pd

def pd_describe(df):
  t1 = time.time()
  res = df.describe()
  t2 = time.time()
  return t2 - t1

def np_partition(col1, col2):
  t1 = time.time()
  len1 = len(col1)
  len2 = len(col2)
  mean1 = np.mean(col1)
  std_dev1 = np.std(col1)
  mean2 = np.mean(col2)
  std_dev2 = np.std(col2)

  min1, max1 = np.min(col1), np.max(col1)
  min2, max2 = np.min(col2), np.max(col2)

  tmp1 = np.partition(col1, N // 2)
  col1_quartile50 = tmp1[N // 2]  # median
  col1_quartile25 = np.partition(tmp1[: N // 2], N // 4)[N // 4]
  col1_quartile75 = np.partition(tmp1[N // 2:], N // 4)[N // 4]

  tmp2 = np.partition(col2, N // 2)
  col2_quartile50 = tmp2[N // 2]  # median
  col2_quartile25 = np.partition(tmp2[: N // 2], N // 4)[N // 4]
  col2_quartile75 = np.partition(tmp2[N // 2:], N // 4)[N // 4]

  t2 = time.time()
  return t2 - t1

def np_percentile(col1, col2):
  t1 = time.time()
  len1 = len(col1)
  len2 = len(col2)
  mean1 = np.mean(col1)
  std_dev1 = np.std(col1)
  mean2 = np.mean(col2)
  std_dev2 = np.std(col2)

  min1, max1 = np.min(col1), np.max(col1)
  min2, max2 = np.min(col2), np.max(col2)

  col1_quartile25 = np.percentile(col1, [25, 50, 75])
  col2_quartile25 = np.percentile(col2, [25, 50, 75])

  # print(f"{len1}, {mean1:.6e},  {std_dev1:.6e},  {min1:.6e},  {col1_quartile25:.6e},  {col1_quartile50:.6e},  {col1_quartile75:.6e},  {max1:.6e}")
  # print(f"{len2}, {mean2:.6e},  {std_dev2:.6e},  {min2:.6e},  {col2_quartile25:.6e},  {col2_quartile50:.6e},  {col2_quartile75:.6e},  {max2:.6e}")
  t2 = time.time()
  return t2 - t1

def np_partition_axis(df):
  t1 = time.time()
  lengths = df.shape[0]
  means = np.mean(df, axis=0)
  std_devs = np.std(df, axis=0)
  mins = np.min(df, axis=0)
  maxs = np.max(df, axis=0)

  tmp1 = np.partition(df, N // 2, axis=0)
  col1_quartile50 = tmp1[N // 2,:]  # median
  col1_quartile25 = np.partition(tmp1[: N // 2, :], N // 4, axis=0)[N // 4, :]
  col1_quartile75 = np.partition(tmp1[N // 2:, :], N // 4, axis=0)[N // 4, :]
  t2 = time.time()
  return t2 - t1

def np_percentile_axis(df):
  t1 = time.time()
  lengths = df.shape[0]
  means = np.mean(df, axis=0)
  std_devs = np.std(df, axis=0)
  mins = np.min(df, axis=0)
  maxs = np.max(df, axis=0)

  col1_quartile25 = np.percentile(col1, [25, 50, 75], axis=0)
  col2_quartile25 = np.percentile(col2, [25, 50, 75], axis=0)
  t2 = time.time()
  return t2 - t1

def np_basic1(df):
  t1 = time.time()
  lengths = df.shape[0]  # Length of each column
  means = np.mean(df, axis=0)
  std_devs = np.std(df, axis=0)
  mins = np.min(df, axis=0)
  maxs = np.max(df, axis=0)
  print(means)
  t2 = time.time()
  return t2 - t1

def np_basic2(col1, col2):
  t1 = time.time()
  len1 = len(col1)
  len2 = len(col2)
  mean1 = np.mean(col1)
  std_dev1 = np.std(col1)
  mean2 = np.mean(col2)
  std_dev2 = np.std(col2)

  min1, max1 = np.min(col1), np.max(col1)
  min2, max2 = np.min(col2), np.max(col2)
  t2 = time.time()
  return t2 - t1

if __name__ == "__main__":
  print(sys.argv[0])

  result = []
  for i in range(10):
    print(i)
    N = 10_000_000 * (i+1)
    col1 = np.random.randint(1, 1_000_000_000, size = N)
    col2 = np.random.normal(size = N)
    df = pd.DataFrame({
      "int1": col1,
      "int2": col2
    })

    res = []
    for j in range(10):
      elapsed_pd = pd_describe(df)
      elapsed_np_part = np_partition(col1, col2)
      elapsed_np_perc = np_percentile(col1, col2)
      elapsed_np_part_ax = np_partition_axis(df)
      elapsed_np_perc_ax = np_percentile_axis(df)
      elapsed_conv = np_partition_axis(np.array(df))
      elapsed_perc_conv = np_percentile_axis(np.array(df))
      res.append([elapsed_pd, elapsed_np_part, elapsed_np_perc, elapsed_np_part_ax, elapsed_np_perc_ax, elapsed_conv, elapsed_perc_conv])
    res = np.mean(np.array(res), axis=0)
    result.append(res)
  result = pd.DataFrame(result, columns=['pd_describe', 'np_partition', 'np_percentile', 'np_partition_axis',
                                         'np_percentile_axis', 'np_partition_array', 'np_percentile_array'])
  result.to_csv('time_pandas_numpy_v3.csv')
  print(result)

  # print(f"elapsed time Pandas describe = {elapsed_pd:.2f} sec\n")
  # print(f"elapsed time Numpy partition = {elapsed_np_part} sec\n")
  # print(f"elapsed time Numpy percentile = {elapsed_np_perc} sec\n")
  # print(f"elapsed time Numpy partition axis = {elapsed_np_part_ax} sec\n")
  # print(f"elapsed time Numpy partition axis converted to array = {elapsed_conv} sec\n")
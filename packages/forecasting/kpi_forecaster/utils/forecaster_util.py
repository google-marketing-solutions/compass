# python3
# coding=utf-8
# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for the forecaster."""

import warnings

from matplotlib import axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')


def plot_dev_and_test_feature_correlations_with_label(
    dev_data: pd.DataFrame,
    oot_test_data: pd.DataFrame,
    label_column: str) -> axes.Axes:
  """Compute and plot correlations between dev and test features and the label.

  Args:
    dev_data: Development ML dataset with features and label.
    oot_test_data: Out of time test ML dataset with features and label.
    label_column: Name of the label column.

  Returns:
    Plot of correlation between features and the label.
  """
  corr_dev = dev_data.corr()
  corr_dev_label = corr_dev[[label_column]].reset_index()
  corr_dev_label.columns = ['var', 'dev_corr']
  corr_test = oot_test_data.corr()
  corr_test_label = corr_test[[label_column]].reset_index()
  corr_test_label.columns = ['var', 'test_corr']

  corr = corr_dev_label.merge(corr_test_label)
  corr = corr[corr['var'] != label_column].sort_values(
      'dev_corr', ascending=False)
  plot = corr.plot.bar(
      x='var',
      figsize=(30, 10),
      title='Feature-Label Correlation: Development vs OOT Testing.')
  plot.set_xlabel('Feature')
  plot.set_ylabel('Correlation')
  return plot


def calculate_mean_absolute_percentage_error(
    true_label: np.array, pred_label: np.array) -> float:
  """Calculates mean absolute percentage error (MAPE).

  Args:
    true_label: An array of true label.
    pred_label: An array of predicted label.

  Returns:
    Mean absolute percentage error
  """
  return np.mean(np.abs((true_label - pred_label) / true_label)) * 100


def plot_predictions_against_labels(
    df: pd.DataFrame,
    true_label_column: str,
    predicted_label_column: str) -> plt:
  """Generates a plot of the actual and predicted label values.

  Args:
    df: scoring ML dataset with true and predicted labels.
    true_label_column: column name of the true label.
    predicted_label_column: column name of the predicted label.

  Returns:
    Plot of the true and predicted label values.
  """
  df = df.sort_values('ts')
  plt.figure(figsize=(20, 3))
  plt.plot(
      df['ts'], df[predicted_label_column],
      'k-', color='green', label='predicted')
  plt.plot(
      df['ts'], df[true_label_column], 'k-', color='red', label='true')
  plt.legend(loc='upper left')

  # Compute metrics for the title
  true_label = df[true_label_column].values
  pred_label = df[predicted_label_column].values
  mae = round(mean_absolute_error(true_label, pred_label), 2)
  rmse = round(np.sqrt(mean_squared_error(true_label, pred_label)), 2)
  mape = round(
      calculate_mean_absolute_percentage_error(true_label, pred_label), 2)
  plt.title(f'rmse={rmse}, mae={mae}, mape={mape}%')
  return plt


def calculate_performance_by_grouping(
    scored_predictions: pd.DataFrame,
    true_label_column: str,
    predicted_label_column: str,
    grouping_column_name: str) -> pd.DataFrame:
  """Generate performance metrics for each day of the week (Sunday to Saturday).

  Args:
    scored_predictions: Dataset with true and predicted labels.
    true_label_column: column name of the true label.
    predicted_label_column: column name of the predicted label.
    grouping_column_name: Name of the column to group by (e.g. hournum, weekday
      or weeknum).

  Returns:
    Overall performance metrics on weekdays, excluding weekends.
  """
  res = scored_predictions.copy()
  res['abs_diff'] = (abs(res[true_label_column] - res[predicted_label_column]))
  res['relative_diff'] = res['abs_diff'] / res[true_label_column] * 100
  res = res.groupby(grouping_column_name)['abs_diff', 'relative_diff'].mean()
  res.columns = ['mae', 'mape (%)']
  res = res.sort_values(grouping_column_name)
  return res

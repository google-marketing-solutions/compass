# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to run CausalImpact analysis using pycausalimpact package."""

import dataclasses
from typing import List, Dict, Tuple, Sequence
from matplotlib import axes
import numpy as np
import pandas as pd
import pycausalimpact
from sklearn import metrics


@dataclasses.dataclass
class CausalImpactInput:
  """Container class for the inputs for run_ci_analysis() function.

  df: Dataset of time series data used in the analysis. Should contain column
    names defined in the parameters date_col, test_col and control_cols.
  date_col: Name of column containing date information in YYYY-MM-DD format and
    pd.to_datetime type.
  test_col: Name of column containing KPI for the test geo used in the analysis.
  control_cols: List of column names containing KPI for the contol geos used in
    the analysis.
  pre_period: List of start and end dates of the pre period used in the analysis
    in YYYY-MM-DD format.
  post_period: List of start and end dates of the post period used in the
    analysis in YYYY-MM-DD format.
  corr_threshold: Control time series having absolute values of their Pearson's
    correlations with the test time series (in the pre period) greater than or
    equal to this value used in the analysis. Should be a value between 0 and
    1.
  confidence_level: Statistical confidence level used in the analysis. Should
    be a value between 0 and 1, for example, 0.95 for the 95% level.
  model_args_standardize: Argument for the CausalImpact function indicating
    whether to standardize the time series data before building the model.
  model_args_nseasons: Argument for the CausalImpact function specifying the
    duration of the period of the seasonal component; if input data is
    specified in terms of days, then choosing nseasons=7 adds a weekly seasonal
    effect.
  model_args_n_sims: Number of model simulations to use at the inference stage.
  """
  df: pd.DataFrame
  date_col: str
  test_col: str
  control_cols: List[str]
  pre_period: List[pd.to_datetime]
  post_period: List[pd.to_datetime]
  corr_threshold: float = 0.7
  confidence_level: float = 0.95
  model_args_standardize: bool = True
  model_args_nseasons: int = 7
  model_args_n_sims: int = 1000


@dataclasses.dataclass
class CausalImpactResults:
  """Container class for the results from run_ci_analysis() function.

  ci_results: The outputs from the CausalImpact() function.
  test_col: Test column name used in the analysis.
  selected_control_cols: Selected control column names used in the analysis.
  test_control_corr: Correlations between selected control time series and test
     time series.
  ts_plot: Plot of the selected time series for the analysis.
  diag_metrics: Disgnostics metrics of the time series model.
  """
  ci_results: pycausalimpact.CausalImpact
  test_col: str
  selected_control_cols: List[str]
  test_control_corr: pd.DataFrame
  ts_plot: axes.Axes
  diag_metrics: Dict[str, float]


def _select_data(
    df: pd.DataFrame,
    date_col: str,
    test_col: str,
    control_cols: Sequence[str],
    pre_period: List[pd.to_datetime],
    post_period: List[pd.to_datetime],
    corr_threshold: float = 0.7
    ) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
  """Selects data for the CausalImpact() analysis.

  Selects the control time series having absolute values of their Pearson's
  correlations with the test time series (in the pre period) greater than or
  equal to corr_threshold.

  Args:
    df: Dataset of time series data used in the analysis. Should contain column
      names defined in the parameters date_col, test_col and control_cols.
    date_col: Name of column containing date information in YYYY-MM-DD format
      and pd.to_datetime type.
    test_col: Name of column containing KPI for the test geo used in the
      analysis.
    control_cols: List of column names containing KPI for the contol geos used
      in the analysis.
    pre_period: List of start and end dates of the pre period used in the
      analysis in YYYY-MM-DD format.
    post_period: List of start and end dates of the post period used in the
      analysis in YYYY-MM-DD format.
    corr_threshold: Correlation threshold to select control time series. Should
      be a value between 0 and 1.

  Returns:
    A list containing the following items:
      df_selected_ci: Dataset selected for the CausalImpact() analysis.
      selected_control_cols: List of control names selected.
      test_control_corr: Correlation between test and control time series.
  """
  # Select the columns and rows for the analysis
  selected_col_names = [date_col, test_col] + list(control_cols)
  df_selected = df.loc[((df[date_col] >= pre_period[0]) &
                        (df[date_col] <= post_period[1])), selected_col_names]

  # Calculate Pearson's correlations between time series for the pre period
  pre_corr = df_selected[df_selected[date_col] <= pre_period[1]].corr()
  # Separate the correlations of the test time series with control time series
  pre_corr_test = pre_corr[pre_corr.index == test_col].melt()
  # Select the control time series where the correlation >= corr_threshold
  selected_control_cols = list(
      pre_corr_test[pre_corr_test['value'] >= corr_threshold]['variable'])
  selected_control_cols.remove(test_col)

  # Select column names for the CausalImpact analysis
  selected_col_names_new = [date_col, test_col] + selected_control_cols
  # Select data for the CausalImpact analysis
  df_selected_ci = df_selected[selected_col_names_new]

  # Correlations between test and control time series
  test_control_corr = pre_corr[pre_corr.index == test_col].round(2)

  return df_selected_ci, selected_control_cols, test_control_corr


def _diagnose_model(ci_results: pycausalimpact.CausalImpact,
                    ci_data: pd.DataFrame,
                    date_col: str, test_col: str,
                    pre_end_date: pd.to_datetime) -> Dict[str, float]:
  """Runs diagnostics on the trained structutal time series model.

  Args:
    ci_results: Output returns by the CausalImpact() function.
    ci_data: Input data used for the CausalImpact() function.
    date_col: Name of column containing date information.
    test_col: Name of column containing KPI for the test geo used in the
      analysis.
    pre_end_date: End date of the pre period used in the analysis in YYYY-MM-DD
      format.

  Returns:
    Dictionary with the following performance metrics on the model training
      dataset: r-squared, mean absolute percentage error (mape) and root mean
      squared error (rmse).
  """
  # Extract actual and predicted values for the model training period
  inf = ci_results.inferences[['preds']].reset_index()
  inf.columns = [date_col, 'preds']
  pred = inf[inf[date_col] <= pre_end_date]
  actual = ci_data.reset_index()
  actual = actual[[date_col, test_col]]
  actual.columns = [date_col, 'actuals']
  pred_actual = actual.merge(pred, on=date_col)

  # Calculates performance metrics
  r2 = round(
      metrics.r2_score(pred_actual['actuals'].values,
                       pred_actual['preds'].values), 2)
  mape = round(
      metrics.mean_absolute_percentage_error(
          pred_actual['actuals'].values, pred_actual['preds'].values) * 100, 2)
  rmse = round(
      np.sqrt(
          metrics.mean_squared_error(pred_actual['actuals'].values,
                                     pred_actual['preds'].values)), 2)

  return {'r-squared': r2, 'mape': mape, 'rmse': rmse}


def run_ci_analysis(input_params: CausalImpactInput) -> CausalImpactResults:
  """Runs CausalImpact analysis by using pycausalimpact package.

  Args:
    input_params: Input paranmeters for the analysis.

  Returns:
    An instance of CausalImpactResults.
  """
  # Select data for the analysis
  df_selected_ci, selected_control_cols, test_control_corr = _select_data(
      input_params.df,
      input_params.date_col,
      input_params.test_col,
      input_params.control_cols,
      input_params.pre_period,
      input_params.post_period,
      input_params.corr_threshold)

  # Plot the selected time series data
  ts_plot = df_selected_ci.plot(figsize=(10, 3), x=input_params.date_col)

  # Prepare data for the CausalImpact analysis
  df_selected_ci = df_selected_ci.set_index(input_params.date_col)

  # Run Causal Impact analysis
  ci_results = pycausalimpact.CausalImpact(
      data=df_selected_ci,
      pre_period=input_params.pre_period,
      post_period=input_params.post_period,
      standardize=input_params.model_args_standardize,
      nseasons=[{
          'period': input_params.model_args_nseasons
      }],
      n_sims=input_params.model_args_n_sims,
      alpha=1 - input_params.confidence_level)

  # Run model diagnostics
  diag_metrics = _diagnose_model(ci_results,
                                 df_selected_ci,
                                 input_params.date_col,
                                 input_params.test_col,
                                 input_params.pre_period[1])

  return CausalImpactResults(ci_results, input_params.test_col,
                             selected_control_cols,
                             test_control_corr, ts_plot, diag_metrics)

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
"""Propensity model custom helper functions."""

import json
import logging
import pathlib
import sys
import types
from typing import List, Union

from matplotlib.backends import backend_pdf
import numpy as np
import pandas as pd
import yaml


logging.basicConfig(
    format='%(levelname)s: %(message)s', level=logging.INFO, stream=sys.stdout)

_VISUALIZATION_OUTPUT_DIR = 'visualization_outputs'

_DEFAULT_CONFIGS = 'configs.yaml'


def get_configs(filename: str) -> types.SimpleNamespace:
  """Gets configuration details from yaml file.

  Args:
    filename: Relative or full path to config file. If not provided, default
      config.yaml is used.

  Returns:
    A namespace object holding configuration.
  """
  if not filename:
    filename = _DEFAULT_CONFIGS
    logging.info('Config file is not provided. Using default configs file.')
  with open(filename, 'r') as f:
    contents = yaml.safe_load(f)
  config = json.loads(
      json.dumps(contents),
      object_hook=lambda content: types.SimpleNamespace(**content))
  return config


def create_folder(folder_name: str) -> pathlib.Path:
  path = pathlib.Path(folder_name)
  try:
    path.mkdir(parents=True, exist_ok=False)
  except FileExistsError:
    logging.warning('Folder "%s" already exists', folder_name)
  else:
    logging.info('Created "%s".', folder_name)
  return path.absolute()


def save_to_pdf(filename: str, plots: Union[np.ndarray,
                                            List[np.ndarray]]) -> None:
  """Saves pyplot axes into a PDF file.

  Args:
    filename: PDF filename to save plots.
    plots: List of pyplot axes to save to PDF.

  Raises:
    TypeError if plots are not in List[np.ndarray] or np.ndarray type.
  """
  abs_path = create_folder(_VISUALIZATION_OUTPUT_DIR)
  pdf_path = pathlib.Path(abs_path, filename)
  logging.info('Creating PDF file in "%s"', pdf_path)
  pdf = backend_pdf.PdfPages(pdf_path)
  if all(isinstance(plot, np.ndarray) for plot in plots):
    for plot in plots:
      pdf.savefig(plot[0].get_figure())
  elif isinstance(plots, np.ndarray):
    pdf.savefig(plots[0].get_figure())
  else:
    raise TypeError('Plots list is not supported. Provide either '
                    'List[np.ndarray] or np.ndarray containing pyplot axes.')
  pdf.close()


def generate_date_range_stats(timeseries: pd.Series) -> pd.DataFrame:
  """Generates statistics about the date range and missing days.

  Takes the pandas Series with date values, converts it to datetime,
  reindexes it using full range of dates and calculates statistics
  like first date, last date, date range length, missing days.

  Args:
    timeseries: Column with date values.

  Returns:
   df_summary: Summary table with date range statistics.
  """
  timeseries = pd.to_datetime(timeseries.values)
  first_day = timeseries.min()
  last_day = timeseries.max()
  df_summary = pd.DataFrame({
      'first_day': [first_day],
      'last_day': [last_day]
  })
  date_range = pd.date_range(start=first_day, end=last_day, freq='D')
  df_summary['date_range_len'] = len(date_range)
  df_summary['dates_count'] = len(timeseries.unique())
  df_summary['date_range'] = df_summary.apply(
      lambda x: date_range.strftime('%Y-%m-%d').tolist(), axis=1)
  df_summary['missing_days'] = df_summary.apply(
      lambda x: (date_range).difference(timeseries).to_list(), axis=1)
  df_summary['number_missing_days'] = df_summary['date_range_len'] - df_summary[
      'dates_count']
  df_summary = df_summary.T
  df_summary.columns = ['value']
  return df_summary

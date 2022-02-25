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
"""Library for running the Exploratory Data Analysis (EDA) in BigQuery.

At this moment it supports GA360 data format stored in BigQuery.
We are planning to enable extending it to firebase data format
and other sources by modifying SQL templates.
"""
import logging
import os
from typing import Dict, Tuple, Union

from IPython.display import display
import pandas as pd

from gps_building_blocks.cloud.utils import bigquery as bigquery_utils
from gps_building_blocks.ml import utils

_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_TEMPLATES_DIR = os.path.join(_UTILS_DIR, 'templates')

_ParamsType = Dict[str, Union[int, float, str]]


class Analysis:
  """Interacts with BigQuery to analyse data in the context of building ML models.

  Attributes:
    tables: A pandas DataFrame with stats of tables in the BigQuery dataset.
    params: Dict with configuration for analysis.
    bq_utils: BigQueryUtils class with methods to manage BigQuery.
  """

  def __init__(self, bq_utils: bigquery_utils.BigQueryUtils,
               params: _ParamsType):
    self.params = params
    self.bq_utils = bq_utils
    self.tables = None

  def _update_params_with_defaults(self, params: _ParamsType):
    """Populates parameters not specified by the user with default values.

    Takes dictionary containing parameters for running the eda process and
    sets missing keys with specified values in order to simplify the usage.

    Args:
      params: Dictonary with parameters populated by user.
    """
    params.setdefault('verbose', True)
    params.setdefault('dataset_description_sql', 'dataset_description.sql')
    params.setdefault('table_stats_sql', 'table_stats')

  def get_ds_description(self) -> Tuple[pd.DataFrame, str]:
    """Queries information schema in BigQuery to get metadata of the dataset.

    Returns:
      df_table_options: Dataframe with table metadata.
      description: Content of the description field (if populated).
    """
    self._update_params_with_defaults(self.params)
    sql = utils.render_jinja_sql(
        template_dir=_TEMPLATES_DIR,
        template_name='dataset_description',
        **self.params)
    df_table_options = self.bq_utils.run_query(sql).to_dataframe()
    description = df_table_options['option_value'][0]
    if self.params['verbose']:
      # Will display outputs only if verbose is True
      # otherwise will just return objects, which can be displayed separately.
      display(df_table_options)
      print(f'description: {description}')
    return (df_table_options, description)

  def get_tables_stats(self) -> pd.DataFrame:
    """Queries tables metadata to check structure of the data.

    Returns:
      self.tables: Dataframe with tables statistics (size, count etc).
    """
    self._update_params_with_defaults(self.params)
    sql = utils.render_jinja_sql(
        template_dir=_TEMPLATES_DIR,
        template_name=self.params['table_stats_sql'],
        **self.params)
    self.tables = self.bq_utils.run_query(sql).to_dataframe()
    # Create derived fields to use for filtering on the table type
    self.tables['is_intraday'] = self.tables['table_id'].str.contains(
        '_intraday_')
    self.tables['table_type'] = self.tables['table_id'].apply(
        lambda x: '_'.join(x.split('_')[:-1]))
    self.tables['table_id_split'] = self.tables['table_id'].str.split('_')
    self.tables['last_suffix'] = self.tables['table_id_split'].apply(
        lambda x: x[-1])
    return self.tables

  def get_table_types(self) -> pd.DataFrame:
    """Aggregates metrics on tables attributes.

    Returns:
      table_types: Aggregated stats per table type.
    """
    segments = ['table_type', 'is_intraday']
    try:
      table_types = self.tables.groupby(segments).agg({
          'table_id': ['count'],
          'size_gb': ['sum'],
          'last_suffix': ['min', 'max']
      })
      return table_types
    except (AttributeError, KeyError):
      logging.error('Provide DataFrame containing %s', segments)

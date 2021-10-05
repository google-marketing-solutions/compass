# Copyright 2021 Google LLC
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

from typing import Dict, Tuple, Union

from IPython.display import display
import pandas as pd

from gps_building_blocks.cloud.utils import bigquery as bigquery_utils
from gps_building_blocks.ml import utils

_TEMPLATES_DIR = 'templates'

_ParamsType = Dict[str, Union[int, float, str]]


class Analysis:
  """Interacts with BigQuery to analyse data in the context of building ML models."""

  def __init__(self, bq_utils: bigquery_utils.BigQueryUtils,
               params: _ParamsType):
    self.params = params
    self.bq_utils = bq_utils

  def _update_params_with_defaults(self, params: _ParamsType):
    """Populates parameters not specified by the user with default values.

    Takes dictionary containing parameters for running the eda process and
    sets missing keys with specified values in order to simplify the usage.

    Args:
      params: Dictonary with parameters populated by user.
    """
    params.setdefault('verbose', True)
    params.setdefault('dataset_description_sql', 'dataset_description.sql')

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

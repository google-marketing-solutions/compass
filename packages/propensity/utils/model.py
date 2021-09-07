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
"""Propensity model templated module(s) based on BigQeury ML."""

import enum
import logging
import sys
from typing import Dict, List, Mapping, Optional, Union
from IPython.core import display
import pandas as pd

from gps_building_blocks.cloud.utils import bigquery as bigquery_utils
from gps_building_blocks.ml import utils

logging.basicConfig(
    format='%(levelname)s: %(message)s', level=logging.INFO, stream=sys.stdout)

_TEMPLATES_DIR = 'templates'
_CLOUD_CONSOLE_URL = 'https://console.cloud.google.com/bigquery?project='

_ParamsType = Mapping[str, Union[str, List[str], Mapping[str, str]]]


class ModelTypes(enum.Enum):
  """Supported model types for PropensityModel."""
  LOGISTIC_REG = 'LOGISTIC_REG'
  AUTOML_CLASSIFIER = 'AUTOML_CLASSIFIER'
  BOOSTED_TREE_CLASSIFIER = 'BOOSTED_TREE_CLASSIFIER'
  DNN_CLASSIFIER = 'DNN_CLASSIFIER'


class PropensityModel:
  """Interacts with BigQuery ML to create and evaluate propensity models."""

  def __init__(self, bq_client: bigquery_utils.BigQueryUtils,
               params: _ParamsType):
    self.params = params
    self.bq_client = bq_client

  def _validate_columns(self, column: Union[str, List[str]]) -> None:
    """Validates feature and target columns required for training.

    Args:
      column: BigQuery column name for features or target value.

    Raises:
      ValueError: If empty features column list or target column provided.
    """
    if isinstance(column, List) and not column:
      raise ValueError('No features provided for training the model. '
                       'Provide at least one feature column.')
    elif isinstance(column, str) and not column:
      raise ValueError('No target column specified. Provide correct target '
                       'column to train the model.')

  def _validate_model_type(self, model_type: str) -> None:
    """Validates propensity model options.

    Args:
      model_type: BigQuery ML model type as defined here
      https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create

    Raises:
      ValueError: If model type is not supported.
    """
    supported_model_types = [model.value for model in ModelTypes]
    if model_type.upper() not in supported_model_types:
      raise ValueError(
          f'"{model_type}" is not supported. Use one of {supported_model_types}.'
      )

  def _validate_training_params(self, params: _ParamsType) -> None:
    """Validates all required parameters for initializing PropensityModel.

    Args:
      params: GCP and BQML model option configuration parameters.
    """
    # TODO(): Add validators for other BQML model options.
    for column in [params['feature_columns'], params['target_column']]:
      self._validate_columns(column)
    self._validate_model_type(params['model_type'])

  def _display_model_url(self) -> None:
    """Displays Cloud Console URL for BigQuery trained model."""
    model_path = str(self.params['model_path'])
    # Extract project id, dataset and model name from model path.
    project_id, dataset, model_name = model_path.split('.')
    url = (f'{_CLOUD_CONSOLE_URL}{project_id}&ws=&p={project_id}'
           f'&d={dataset}&m={model_name}&page=model')
    html_tag = f'<a href={url}>BigQuery model link</a>'

    logging.info('Finished training. Model can be access via following URL:')
    # Make sure to write html tag only in Jupyter notebook environment.
    if 'ipykernel' in sys.modules:
      display.display(display.HTML(html_tag))
    else:
      logging.info(url)

  def train(self, verbose: bool = False) -> pd.DataFrame:
    """Trains propensity model in BigQuery ML.

    Args:
      verbose: If set true, prints parsed SQL content.

    Returns:
      Training BigQuery job results as Pandas DataFrame.
    """
    self._validate_training_params(self.params)
    # Flatten nested dict params for parser to apply params to sql template.
    train_params = {key: value for key, value in self.params.items()}
    sql = utils.render_jinja_sql(
        template_dir=_TEMPLATES_DIR,
        template_name='classification',
        verbose=verbose,
        **train_params)
    training_job = self.bq_client.run_query(sql)
    self._display_model_url()
    return training_job.result().to_dataframe()

  def get_feature_info(self,
                       model_path: Optional[str] = None,
                       verbose: Optional[bool] = False) -> pd.DataFrame:
    """Retreives dataset feature information from BigQuery training dataset.

    Args:
      model_path: Full path to BigQuery model. Ex: project.dataset.model. This
        is optional, if not provided, default model path specified in params is
        used to extract feature inforamation.
      verbose: If set true, prints parsed SQL content.

    Returns:
      Pandas DataFrame containing features' statistics.
    """
    if model_path is None:
      params = {'model_path': self.params['model_path']}
    else:
      params = {'model_path': model_path}

    sql = utils.render_jinja_sql(
        template_dir=_TEMPLATES_DIR,
        template_name='features_info',
        verbose=verbose,
        **params)
    return self.bq_client.run_query(sql).result().to_dataframe()

  def evaluate(self,
               params: Optional[Mapping[str, Union[str, float]]] = None,
               verbose: bool = False) -> List[pd.DataFrame]:
    """Evaluates BigQuery ML trained model.

    Args:
      params: Additional evaluation parameters containing evaluation dataset
        name and threshold value.
      verbose: If set true, prints parsed SQL content.

    Returns:
      List of DataFrames containing model's evaluation, confusion matrix and ROC
      curve metrics.
    """
    eval_params = {'model_path': self.params['model_path']}
    if eval_params:
      eval_params.update(params)

    eval_dataframes = []
    for template in ['evaluation', 'confustion_matrix', 'roc_curve']:
      sql = utils.render_jinja_sql(
          template_dir=_TEMPLATES_DIR,
          template_name=template,
          verbose=verbose,
          **eval_params)
      dataframe = self.bq_client.run_query(sql).result().to_dataframe()
      eval_dataframes.append(dataframe)
    return eval_dataframes

  def predict(self,
              params: Dict[str, Union[str, float]],
              overwrite_table: Optional[bool] = True,
              verbose: Optional[bool] = False) -> None:
    """Executes prediction job using BigQuery ML trained model.

    Args:
      params: Prediction parameters.
      overwrite_table: If true, prediction results table is overwritten each
        time prediction is made.
      verbose: If set true, prints parsed SQL content.

    Returns:
      Prediction BigQuery job results as Pandas DataFrame.
    """
    predict_params = params
    if overwrite_table:
      predict_params.update({'overwrite_table': overwrite_table})
    sql = utils.render_jinja_sql(
        template_dir=_TEMPLATES_DIR,
        template_name='prediction',
        verbose=verbose,
        **predict_params)
    # TODO(): Add a prediction table link post prediction job.
    self.bq_client.run_query(sql)

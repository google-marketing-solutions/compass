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
"""Classification and Regression model templated module(s) based on BigQeury ML."""

import abc
import enum
import logging
import os
import sys
from typing import Dict, List, Mapping, Optional, Union
from IPython.core import display
import pandas as pd
from gps_building_blocks.cloud.utils import bigquery as bigquery_utils
from gps_building_blocks.ml import utils

logging.basicConfig(
    format='%(levelname)s: %(message)s', level=logging.INFO, stream=sys.stdout)

_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_TEMPLATES_DIR = os.path.join(_UTILS_DIR, 'templates')
_ParamsType = Mapping[str, Union[str, List[str], Mapping[str, str]]]


class ModelTypes(enum.Enum):
  """Supported model types."""
  REGRESSION = 'REGRESSION'
  CLASSIFICATION = 'CLASSIFICATION'

  def validate_model_type(self, model_type: str) -> None:
    """Validates supported model types.

    Args:
      model_type: Type of the model.

    Raises:
      ValueError: If model type is not supported.
    """
    if self == ModelTypes.REGRESSION:
      model_types = RegressionModelTypes
    elif self == ModelTypes.CLASSIFICATION:
      model_types = ClassificationModelTypes
    else:
      raise ValueError(f'{model_type} is not supported.')

    supported_model_types = model_types.supported_types()
    if model_type.upper() not in supported_model_types:
      raise ValueError(
          f'"{model_type}" is not supported. Use one of {supported_model_types}.'
      )


class ClassificationModelTypes(enum.Enum):
  """Supported model types for PropensityModel."""
  LOGISTIC_REG = 'LOGISTIC_REG'
  AUTOML_CLASSIFIER = 'AUTOML_CLASSIFIER'
  BOOSTED_TREE_CLASSIFIER = 'BOOSTED_TREE_CLASSIFIER'
  DNN_CLASSIFIER = 'DNN_CLASSIFIER'

  @staticmethod
  def supported_types() -> List[str]:
    return [model.value for model in ClassificationModelTypes]


class RegressionModelTypes(enum.Enum):
  """Supported model types for LTVModel."""
  LINEAR_REG = 'LINEAR_REG'
  AUTOML_REGRESSOR = 'AUTOML_REGRESSOR'
  BOOSTED_TREE_REGRESSOR = 'BOOSTED_TREE_REGRESSOR'
  DNN_REGRESSOR = 'DNN_REGRESSOR'

  @staticmethod
  def supported_types() -> List[str]:
    return [model.value for model in RegressionModelTypes]


class Model(abc.ABC):
  """This is an abstract class for Compass Package models."""

  def __init__(self, bq_utils: bigquery_utils.BigQueryUtils,
               params: _ParamsType):
    self.params = {k.lower(): v for k, v in params.items()}
    self.bq_utils = bq_utils

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

  def _validate_training_params(self, model_type: ModelTypes) -> None:
    """Validates all required parameters for initializing BQML model.

    Args:
      model_type: Type of the model.
    """
    # TODO(): Add validators for other BQML model options.
    for column in [
        self.params['feature_columns'], self.params['target_column']
    ]:
      self._validate_columns(column)
    model_type.validate_model_type(str(self.params['model_type']))

  def _display_url(self, url_part: str, text: str = 'Access URL.') -> None:
    """Displays newly created BigQuery Dataset, Table or Model URLs.

    Args:
      url_part: Newly generated URL part concatenated to cloud console url.
      text: URL text to show to user.
    """
    cloud_console_url = 'https://console.cloud.google.com/bigquery?project='
    # Makes sure to write html tag only in Jupyter notebook environment.
    url = f'{cloud_console_url}{url_part}'
    if 'ipykernel' in sys.modules:
      html_tag = f'<a href={url}>{text}</a>'
      display.display(display.HTML(html_tag))
    else:
      logging.info(url)

  def _display_model_url(self) -> None:
    project_id, dataset, model_name = str(self.params['model_path']).split('.')
    url = (f'{project_id}&ws=&p={project_id}&d={dataset}'
           f'&m={model_name}&page=model')

    logging.info('Finished training. Model can be access via following URL:')
    self._display_url(url, 'BigQuery Model.')

  def _display_table_url(self,
                         table_path: str,
                         text: str = 'BigQuery Table.') -> None:
    project_id, dataset, table = table_path.split('.')
    url = f'&d={dataset}&p={project_id}&t={table}&page=table'

    self._display_url(url, text)

  def _get_train_params(self) -> _ParamsType:
    """Extracts train params to be used in sql template.

    Returns:
      _ParamsType with flattened key value pair.
    """
    return {key: value for key, value in self.params.items()}

  @abc.abstractmethod
  def train(self, verbose: bool = False) -> pd.DataFrame:
    """Trains BigQuery ML.

    Args:
      verbose: If set true, prints parsed SQL content.

    Returns:
      Training BigQuery job results as Pandas DataFrame.
    """

  @abc.abstractmethod
  def evaluate(self, verbose: bool = False) -> List[pd.DataFrame]:
    """Evaluates pretrained BigQuery ML model.

    Args:
      verbose: If set true, prints parsed SQL content.

    Returns:
      List of DataFrames containing model's evaluation metrics.
    """

  @abc.abstractmethod
  def predict(self, verbose: bool = False) -> pd.DataFrame:
    """Predicts based on pretrained BigQuery ML model.

    Args:
      verbose: If set true, prints parsed SQL content.

    Returns:
      Prediction BigQuery job results as Pandas DataFrame.
    """

  def get_feature_info(self,
                       model_path: Optional[str] = None,
                       verbose: Optional[bool] = False) -> pd.DataFrame:
    """Retreives dataset feature information from BigQuery training dataset.

    Args:
      model_path: Full path to BigQuery model. Ex: project.dataset.model.
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
    return self.bq_utils.run_query(sql).to_dataframe()


class LTVModel(Model):
  """Interacts with BigQuery ML to create and evaluate LTV models."""

  def train(self, verbose: bool = False) -> None:
    """Trains regression model in BigQuery ML.

    Args:
      verbose: If set true, prints parsed SQL content.

    Returns:
      Training BigQuery job results as Pandas DataFrame.
    """
    self._validate_training_params(model_type=ModelTypes.REGRESSION)
    train_params = self._get_train_params()
    sql = utils.render_jinja_sql(
        template_dir=_TEMPLATES_DIR,
        template_name='regression',
        verbose=verbose,
        **train_params)
    self.bq_utils.run_query(sql)
    self._display_model_url()

  def evaluate(self,# overriding-default-value-checks
               params: Optional[Mapping[str, Union[str, float]]] = None,
               verbose: bool = False) -> pd.DataFrame:
    """Evaluates BigQuery ML trained regression model.

    Args:
      params: Additional evaluation parameters containing evaluation dataset
        name and threshold value.
      verbose: If set true, prints parsed SQL content.

    Returns:
      DataFrame containing model's mean_absolute_error, mean_squared_error,
      mean_squared_log_error, median_absolute_error, r2_score and
      explained_variance metrics.
    """
    eval_params = {'model_path': self.params['model_path']}
    if eval_params:
      eval_params.update(params)

    sql = utils.render_jinja_sql(
        template_dir=_TEMPLATES_DIR,
        template_name='regression_eval',
        verbose=verbose,
        **eval_params)
    return self.bq_utils.run_query(sql).to_dataframe()

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
        template_name='regression_predict',
        verbose=verbose,
        **predict_params)
    self.bq_utils.run_query(sql)
    logging.info('Finished scoring.')
    self._display_table_url(
        table_path=params['output_table_path'], text='Prediction Table.')


class PropensityModel(Model):
  """Interacts with BigQuery ML to train, evaluate and predict."""

  def train(self, verbose: bool = False) -> None:
    """Trains propensity model in BigQuery ML.

    Args:
      verbose: If set true, prints parsed SQL content.

    Returns:
      Training BigQuery job results as Pandas DataFrame.
    """
    self._validate_training_params(model_type=ModelTypes.CLASSIFICATION)
    train_params = self._get_train_params()
    sql = utils.render_jinja_sql(
        template_dir=_TEMPLATES_DIR,
        template_name='classification',
        verbose=verbose,
        **train_params)
    self.bq_utils.run_query(sql)
    self._display_model_url()

  def evaluate(self,# overriding-default-value-checks
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
    for template in ['evaluation', 'confusion_matrix', 'roc_curve']:
      sql = utils.render_jinja_sql(
          template_dir=_TEMPLATES_DIR,
          template_name=template,
          verbose=verbose,
          **eval_params)
      dataframe = self.bq_utils.run_query(sql).to_dataframe()
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
    self.bq_utils.run_query(sql)
    logging.info('Finished scoring.')
    self._display_table_url(
        table_path=params['output_table_path'], text='Scoring Table.')

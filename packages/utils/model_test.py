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
"""Tests for model."""
import inspect
from unittest import mock

from absl.testing import absltest
from IPython.core import display
import pandas as pd

from gps_building_blocks.cloud.utils import bigquery as bigquery_utils
from compass.packages.utils import model

_MODEL_PARAMS = {
    'model_path': 'project.dataset.model',
    'features_table_path': 'project.dataset.table',
    'model_type': 'LOGISTIC_REG',
    'data_split_method': 'AUTO_SPLIT',
    'auto_class_weights': 'TRUE',
    'feature_columns': ['feature1', 'feature2'],
    'target_column': 'label'
}


class ModelTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.addCleanup(mock.patch.stopall)
    self.mock_bigquery_utils = mock.patch.object(
        bigquery_utils, 'BigQueryUtils', autospec=True).start()
    self.mock_display = mock.patch.object(
        display, 'display', autospec=True).start()
    self.model = model.PropensityModel(self.mock_bigquery_utils, _MODEL_PARAMS)
    self.mock_query = self.model.bq_utils.run_query
    self.mock_df = self.mock_query.return_value.to_dataframe

  def test_train_returns_job_result_as_dataframe(self):
    train_sql_template = """
      CREATE OR REPLACE MODEL
        `{{ model_path }}`
      OPTIONS(
        MODEL_TYPE='{{ model_type }}',
        INPUT_LABEL_COLS=['{{ target_column }}'],
        AUTO_CLASS_WEIGHTS={{ auto_class_weights }}
      ) AS
      SELECT
        {{ feature_columns |join(', ') }}, {{ target_column }}
      FROM
        `{{ table_path }}`;
    """
    expected_result = pd.DataFrame([{'iteration': 1, 'loss': 0.4}])
    self.mock_df.return_value = expected_result

    with mock.patch('builtins.open',
                    mock.mock_open(read_data=train_sql_template)):
      actual_result = self.model.train()
    pd.testing.assert_frame_equal(actual_result, expected_result)

  def test_get_feature_info_returns_job_result_as_dataframe(self):
    feature_info_sql_template = """
      SELECT
        *
      FROM ML.FEATURE_INFO(MODEL `{{ model_path }}`);
    """
    expected_result = pd.DataFrame([{
        'input': 'column1',
        'min': 0.1,
        'max': 0.0,
        'mean': 0.1,
        'median': 0.2,
        'stddev': 0.3,
        'category_count': 'null',
        'null_count': 0.4
    }])
    self.mock_df.return_value = expected_result

    with mock.patch('builtins.open',
                    mock.mock_open(read_data=feature_info_sql_template)):
      actual_result = self.model.get_feature_info()
    pd.testing.assert_frame_equal(actual_result, expected_result)

  def test_evaluate_returns_job_list_of_eval_metrics_dataframes(self):
    sql_template = """
      SELECT
        *
      FROM ML.EVALUATE(MODEL `{{ model_path }}`);
    """

    params = {'eval_table_path': 'project.dataset.model'}
    evaluation_metrics = pd.DataFrame([{
        'precision': {
            0: 0.02
        },
        'recall': {
            0: 0.5
        },
        'accuracy': {
            0: 0.56
        },
        'f1_score': {
            0: 0.01
        },
        'log_loss': {
            0: 0.69
        },
        'roc_auc': {
            0: 0.61
        }
    }])
    confusion_matrix = pd.DataFrame([{
        'expected_label': 'false',
        'FALSE': '5',
        'TRUE': '4'
    }, {
        'expected_label': 'true',
        'FALSE': '0',
        'TRUE': '2'
    }])
    roc_curve = pd.DataFrame([{
        'threshold': '0.5',
        'recall': '0.1',
        'false_positive_rate': '0.01',
        'true_positives': '0.5',
        'false_positives': '1',
        'true_negatives': '9',
        'false_negatives': '1',
        'precision': '0.3'
    }])
    expected_dataframes = [evaluation_metrics, confusion_matrix, roc_curve]
    self.mock_df.side_effect = expected_dataframes

    with mock.patch('builtins.open', mock.mock_open(read_data=sql_template)):
      actual_dataframes = self.model.evaluate(params=params)
      for actual, expected in zip(actual_dataframes, expected_dataframes):
        pd.testing.assert_frame_equal(actual, expected)
    self.assertEqual(len(actual_dataframes), len(expected_dataframes))

  def test_predict_returns_returns_empty_row_iterator(self):
    predict_sql_template = inspect.cleandoc(
        """CREATE TABLE `{{ output_table_path }}` AS
        SELECT *
        FROM ML.PREDICT(MODEL `{{ model_path }}`,
          (SELECT {{ feature_columns |join(', ') }}
          FROM `{{ features_table_path }}`));""")
    params = {
        'model_path': 'project.dataset.model',
        'features_table_path': 'project.dataset.features',
        'output_table_path': 'project.dataset.scored',
        'feature_columns': ['feature1', 'feature2'],
        'overwrite_table': False
    }
    expected_query = inspect.cleandoc(
        """CREATE TABLE `project.dataset.scored` AS
        SELECT *
        FROM ML.PREDICT(MODEL `project.dataset.model`,
          (SELECT feature1, feature2
          FROM `project.dataset.features`));""")

    with mock.patch('builtins.open',
                    mock.mock_open(read_data=predict_sql_template)):
      self.model.predict(params=params)
      self.mock_bigquery_utils.run_query.assert_called_once_with(expected_query)


if __name__ == '__main__':
  absltest.main()

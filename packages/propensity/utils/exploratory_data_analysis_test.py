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
"""Tests for exploratory data analysis."""
from unittest import mock

from absl.testing import absltest
from IPython.core import display
import pandas as pd

from gps_building_blocks.cloud.utils import bigquery as bigquery_utils
from compass.packages.propensity.utils import exploratory_data_analysis

_EDA_PARAMS = {
    'dataset': 'project.dataset.table',
    'verbose': False
}


class EdaTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.addCleanup(mock.patch.stopall)
    self.mock_bigquery_utils = mock.patch.object(
        bigquery_utils, 'BigQueryUtils', autospec=True).start()
    self.mock_display = mock.patch.object(
        display, 'display', autospec=True).start()
    self.exploratory_data_analysis = exploratory_data_analysis.Analysis(
        self.mock_bigquery_utils, _EDA_PARAMS)
    self.mock_query_value = self.exploratory_data_analysis.bq_utils.run_query.return_value
    self.mock_df = self.mock_query_value.to_dataframe

  def test_get_ds_description_returns_job_result_as_dataframe(self):
    get_description_template = """
    SELECT
          *
    FROM
    `{{dataset_path}}`.INFORMATION_SCHEMA.TABLE_OPTIONS;

    """
    table_options = {
        'table_catalog': {
            0: 'project'
        },
        'table_schema': {
            0: 'dataset'
        },
        'table_name': {
            0: 'table_20170801'
        },
        'option_name': {
            0: 'description'
        },
        'option_type': {
            0: 'STRING'
        },
        'option_value': {
            0:
                '"Example BigQuery dataset description line 1. '
                'Description line 2."'
        }
    }
    description = pd.DataFrame(table_options)['option_value'][0]

    expected_df = pd.DataFrame(table_options)
    expected_str = description
    self.mock_df.return_value = expected_df

    with mock.patch('builtins.open',
                    mock.mock_open(read_data=get_description_template)):
      actual_df, actual_str = self.exploratory_data_analysis.get_ds_description(
      )
    pd.testing.assert_frame_equal(actual_df, expected_df)
    self.assertEqual(actual_str, expected_str)


if __name__ == '__main__':
  absltest.main()

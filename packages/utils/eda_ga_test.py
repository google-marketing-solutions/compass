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
"""Tests for exploratory data analysis."""
from unittest import mock

from absl.testing import absltest
from IPython.core import display
import pandas as pd

from gps_building_blocks.cloud.utils import bigquery as bigquery_utils
from compass.packages.utils import eda_ga

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
    self.eda_ga = eda_ga.Analysis(self.mock_bigquery_utils, _EDA_PARAMS)
    self.mock_query_value = self.eda_ga.bq_utils.run_query.return_value
    self.mock_df = self.mock_query_value.to_dataframe
    self.tables = {
        'project_id': {
            0: 'bigquery-public-data',
            1: 'bigquery-public-data'
        },
        'dataset_id': {
            0: 'google_analytics_sample',
            1: 'google_analytics_sample'
        },
        'table_id': {
            0: 'ga_sessions_20160801',
            1: 'ga_sessions_20160802'
        },
        'creation_time': {
            0: 1522253706433,
            1: 1522253739710
        },
        'last_modified_time': {
            0: 1540300347286,
            1: 1540300362205
        },
        'row_count': {
            0: 1711,
            1: 2140
        },
        'size_bytes': {
            0: 19920125,
            1: 23308851
        },
        'type': {
            0: 1,
            1: 1
        },
        'size_mb': {
            0: 19.920125,
            1: 23.308851
        },
        'size_gb': {
            0: 0.019920125,
            1: 0.023308851
        },
        'is_intraday': {
            0: False,
            1: False
        },
        'table_type': {
            0: 'ga_sessions',
            1: 'ga_sessions'
        },
        'table_id_split': {
            0: ['ga', 'sessions', '20160801'],
            1: ['ga', 'sessions', '20160802']
        },
        'last_suffix': {
            0: '20160801',
            1: '20160802'
        }
    }

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
      actual_df, actual_str = self.eda_ga.get_ds_description()
    pd.testing.assert_frame_equal(actual_df, expected_df)
    self.assertEqual(actual_str, expected_str)

  def test_get_tables_stats_returns_job_result_as_dataframe(self):
    get_tables_template = """
    SELECT
      *,
      SAFE_DIVIDE(size_bytes,
        POW(10,6)) AS size_mb,
      SAFE_DIVIDE(size_bytes, POW(10,9)) AS size_gb
    FROM
    `{{dataset_path}}`.__TABLES__
    ORDER BY table_id;
    """
    expected_df = pd.DataFrame(self.tables)
    self.mock_df.return_value = expected_df

    with mock.patch('builtins.open',
                    mock.mock_open(read_data=get_tables_template)):
      actual_df = self.eda_ga.get_tables_stats()
    pd.testing.assert_frame_equal(actual_df, expected_df)

  def test_get_table_types_returns_correct_results(self):
    expected = pd.DataFrame({
        ('table_id', 'count'): {
            ('ga_sessions', False): 2
        },
        ('size_gb', 'sum'): {
            ('ga_sessions', False): 0.043228976
        },
        ('last_suffix', 'min'): {
            ('ga_sessions', False): '20160801'
        },
        ('last_suffix', 'max'): {
            ('ga_sessions', False): '20160802'
        }
    })
    self.eda_ga.tables = pd.DataFrame(self.tables)
    actual = self.eda_ga.get_table_types()
    # Added check_like to ignore undeterministic order of the index
    # due to using dict in the data set up.
    pd.testing.assert_frame_equal(actual, expected, check_like=True)


if __name__ == '__main__':
  absltest.main()

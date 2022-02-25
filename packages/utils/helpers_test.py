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
"""Tests for utils."""

import pathlib
import types
from unittest import mock

from absl.testing import absltest
from matplotlib import pyplot
from matplotlib.backends import backend_pdf
import pandas as pd

from compass.packages.utils import helpers


class UtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.addCleanup(mock.patch.stopall)
    self.mock_mkdir = mock.patch.object(
        pathlib.Path, 'mkdir', autospec=True).start()
    self.mock_pdfpages = mock.patch.object(
        backend_pdf, 'PdfPages', autospec=True).start()

  def test_get_configs_returns_source_and_dest_configs(self):
    test_configs = """
      source:
        project_id: 'source_project_id'
        dataset_name: 'source_dataset_name'
        table_name: 'source_table_name'
      destination:
        project_id: 'destination_project_id'
        dataset_name: 'destination_dataset_name'
        table_name: 'destination_table_name'
    """
    mock_open = mock.mock_open(read_data=test_configs)
    with mock.patch('builtins.open', mock_open, create=True):
      configs = helpers.get_configs('configs.yaml')

    self.assertIsInstance(configs, types.SimpleNamespace)
    source_config, dest_config = configs.source, configs.destination

    for config in [source_config, dest_config]:
      for attr in ['project_id', 'dataset_name', 'table_name']:
        self.assertTrue(hasattr(config, attr))

  def test_create_folder_returns_absolute_path(self):
    folder_name = 'test_folder'
    actual_path = helpers.create_folder(folder_name)
    expected_path = pathlib.Path(pathlib.Path.cwd(), folder_name)
    self.assertEqual(actual_path, expected_path)

  def test_save_pdf_saves_pyplot_axes_to_pdf_file(self):
    _, plots = pyplot.subplots(nrows=1, ncols=2)
    filename = 'test_plots.pdf'

    helpers.save_to_pdf(filename, plots)

    self.mock_pdfpages.return_value.savefig.assert_called_once_with(
        plots[0].get_figure())

  def test_save_pdf_raises_value_error_on_incorrect_plots_instance(self):
    filename = 'test_plots.pdf'
    plots = [[1, 2, 3]]

    with self.assertRaises(TypeError):
      helpers.save_to_pdf(filename, plots)

  def test_generate_date_range_stats_raises_error_if_empty_series(self):
    timeseries = pd.Series([])
    with self.assertRaises(ValueError):
      helpers.generate_date_range_stats(timeseries)

  def test_generate_date_range_stats_raises_error_if_str_passed(self):
    timeseries = '20160801'
    with self.assertRaises(AttributeError):
      helpers.generate_date_range_stats(timeseries)

  def test_generate_date_range_stats_returns_correct_summary_all_days(self):
    expected = pd.DataFrame({
        'value': {
            'first_day': pd.Timestamp('2016-08-01 00:00:00'),
            'last_day': pd.Timestamp('2016-08-05 00:00:00'),
            'date_range_len': 5,
            'dates_count': 5,
            'date_range': [
                '2016-08-01', '2016-08-02', '2016-08-03', '2016-08-04',
                '2016-08-05'
            ],
            'missing_days': [],
            'number_missing_days': 0
        }
    })
    timeseries = pd.Series(
        ['20160801', '20160802', '20160803', '20160804', '20160805'])
    actual = helpers.generate_date_range_stats(timeseries)
    # Added check_like to ignore undeterministic order of the index
    # due to using dict in the data set up.
    pd.testing.assert_frame_equal(actual, expected, check_like=True)

  def test_generate_date_range_stats_returns_correct_summary_missing_days(self):
    expected = pd.DataFrame({
        'value': {
            'first_day': pd.Timestamp('2016-08-01 00:00:00'),
            'last_day': pd.Timestamp('2016-08-05 00:00:00'),
            'date_range_len': 5,
            'dates_count': 4,
            'date_range': [
                '2016-08-01', '2016-08-02', '2016-08-03', '2016-08-04',
                '2016-08-05'
            ],
            'missing_days': [pd.Timestamp('2016-08-02 00:00:00')],
            'number_missing_days': 1
        }
    })
    timeseries = pd.Series(['20160801', '20160803', '20160804', '20160805'])
    actual = helpers.generate_date_range_stats(timeseries)
    # Added check_like to ignore undeterministic order of the index
    # due to using dict in the data set up.
    pd.testing.assert_frame_equal(actual, expected, check_like=True)


if __name__ == '__main__':
  absltest.main()

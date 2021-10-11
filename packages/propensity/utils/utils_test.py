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
"""Tests for utils."""

import pathlib
from unittest import mock
from absl.testing import absltest

from matplotlib import pyplot
from matplotlib.backends import backend_pdf

from compass.packages.propensity.utils import utils


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
      source_configs, dest_configs = utils.get_configs('configs.yaml')

    for config in [source_configs, dest_configs]:
      self.assertIsInstance(config, utils.Configs)
      for attr in ['project_id', 'dataset_name', 'table_name']:
        self.assertTrue(hasattr(config, attr))

  def test_get_configs_returns_dest_configs_only(self):
    test_configs = """
      destination:
        project_id: 'destination_project_id'
        dataset_name: 'destination_dataset_name'
        table_name: 'destination_table_name'
    """
    mock_open = mock.mock_open(read_data=test_configs)
    with mock.patch('builtins.open', mock_open, create=True):
      dest_configs = utils.get_configs('configs.yaml')

    self.assertIsInstance(dest_configs, utils.Configs)
    for attr in ['project_id', 'dataset_name', 'table_name']:
      self.assertTrue(hasattr(dest_configs, attr))

  def test_get_configs_raises_error_if_dest_values_not_provided(self):
    test_configs = """
      source:
        project_id: 'source_project_id'
        dataset_name: 'source_dataset_name'
        table_name: 'source_table_name'
      destination:
        project_id: ''
        dataset_name: ''
        table_name: ''
    """
    mock_open = mock.mock_open(read_data=test_configs)

    with mock.patch('builtins.open', mock_open, create=True):
      with self.assertRaises(ValueError):
        utils.get_configs('configs.yaml')

  def test_create_folder_returns_absolute_path(self):
    folder_name = 'test_folder'
    actual_path = utils.create_folder(folder_name)
    expected_path = pathlib.Path(pathlib.Path.cwd(), folder_name)
    self.assertEqual(actual_path, expected_path)

  def test_save_pdf_saves_pyplot_axes_to_pdf_file(self):
    _, plots = pyplot.subplots(nrows=1, ncols=2)
    filename = 'test_plots.pdf'

    utils.save_to_pdf(filename, plots)

    self.mock_pdfpages.return_value.savefig.assert_called_once_with(
        plots[0].get_figure())

  def test_save_pdf_raises_value_error_on_incorrect_plots_instance(self):
    filename = 'test_plots.pdf'
    plots = [[1, 2, 3]]

    with self.assertRaises(TypeError):
      utils.save_to_pdf(filename, plots)


if __name__ == '__main__':
  absltest.main()

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
"""Tests for run_ci_analysis."""

from absl.testing import absltest
from matplotlib import axes
import pandas as pd
import pycausalimpact
from compass.packages.geo_experiments.utils import run_ci

_DATA = pd.DataFrame({
    'Date': pd.to_datetime([
        '2018-07-01', '2018-07-02', '2018-07-03', '2018-07-04', '2018-07-05',
        '2018-07-06', '2018-07-07', '2018-07-08', '2018-07-09', '2018-07-10',
        '2018-07-11', '2018-07-12', '2018-07-13', '2018-07-14', '2018-07-15',
        '2018-07-16', '2018-07-17', '2018-07-18', '2018-07-19', '2018-07-20',
        '2018-07-21', '2018-07-22', '2018-07-23', '2018-07-24', '2018-07-25',
        '2018-07-26', '2018-07-27', '2018-07-28', '2018-07-29', '2018-07-30',
        '2018-07-31', '2018-08-01', '2018-08-02', '2018-08-03', '2018-08-04',
        '2018-08-05', '2018-08-06', '2018-08-07', '2018-08-08', '2018-08-09'
    ]),
    'Geo_1': [
        22347.0, 19708.0, 18688.0, 19054.0, 21678.0, 21487.0, 21497.0, 24233.0,
        21382.0, 20300.0, 19839.0, 19160.0, 17952.0, 18815.0, 20606.0, 19970.0,
        18371.0, 17819.0, 18069.0, 18884.0, 21292.0, 23065.0, 20067.0, 19784.0,
        19619.0, 18248.0, 17888.0, 16798.0, 21487.0, 19627.0, 18514.0, 17457.0,
        16801.0, 16802.0, 18234.0, 21560.0, 19416.0, 18987.0, 18883.0, 18790.0
    ],
    'Geo_2': [
        10594.0, 9218.0, 9427.0, 8993.0, 10391.0, 10255.0, 11512.0, 12049.0,
        10685.0, 9112.0, 10241.0, 9800.0, 9009.0, 9444.0, 9883.0, 9946.0,
        9079.0, 8752.0, 9553.0, 8711.0, 9692.0, 11987.0, 9432.0, 9895.0, 9727.0,
        9756.0, 9154.0, 7895.0, 10304.0, 9751.0, 9735.0, 9330.0, 7942.0, 8972.0,
        9661.0, 9901.0, 8885.0, 9064.0, 9711.0, 9879.0
    ],
    'Geo_3': [
        6518.0, 6697.0, 6556.0, 6372.0, 8218.0, 6868.0, 6944.0, 7488.0, 7634.0,
        7059.0, 5717.0, 7088.0, 5947.0, 6848.0, 7278.0, 6695.0, 6269.0, 5722.0,
        6193.0, 6220.0, 6553.0, 7036.0, 6985.0, 5794.0, 6860.0, 6922.0, 5919.0,
        5206.0, 6746.0, 6477.0, 6732.0, 5248.0, 5660.0, 5169.0, 5693.0, 7588.0,
        5062.0, 7010.0, 5471.0, 6552.0
    ]
})


class RunCIAanalysisTest(absltest.TestCase):

  def setUp(self):
    super(RunCIAanalysisTest, self).setUp()

    ci_input_params = run_ci.CausalImpactInput(
        df=_DATA,
        date_col='Date',
        test_col='Geo_1',
        control_cols=['Geo_2', 'Geo_3'],
        pre_period=[pd.to_datetime('2018-07-01'), pd.to_datetime('2018-07-31')],
        post_period=[pd.to_datetime('2018-08-01'),
                     pd.to_datetime('2018-08-09')],
        corr_threshold=0.7,
        confidence_level=0.95)
    self.results = run_ci.run_ci_analysis(ci_input_params)

  def test_run_ci_analysis_returns_causalimpact_instance(self):
    self.assertIsInstance(self.results.ci_results, pycausalimpact.CausalImpact)

  def test_run_ci_analysis_returns_test_col(self):
    self.assertEqual(self.results.test_col, 'Geo_1')

  def test_run_ci_analysis_returns_control_cols(self):
    self.assertListEqual(self.results.selected_control_cols, ['Geo_2'])

  def test_run_ci_analysis_returns_correlations(self):
    # test correlations between Test and Control geos
    self.assertAlmostEqual(
        list(self.results.test_control_corr.iloc[0, :]), [1.0, 0.86, 0.65])

  def test_run_ci_analysis_returns_time_series_plot(self):
    # test the plot instance type
    self.assertIsInstance(self.results.ts_plot, axes.Axes)

  def test_run_ci_analysis_returns_disgnostic_metrics(self):
    # test model dosgnostics metrics
    self.assertListEqual(list(self.results.diag_metrics.keys()),
                         ['r-squared', 'mape', 'rmse'])


if __name__ == '__main__':
  absltest.main()

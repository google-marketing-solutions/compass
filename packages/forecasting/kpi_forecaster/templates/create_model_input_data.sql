# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SQL script for creating the features table, which is the input to model training.
# Args:
#   project_id: GCP project ID.
#   dataset_id: Dataset to write the output features table.
#   features_table: Name of the table to write the features table.
#   features_template: SQL template for generating features. See features.sql for reference.
#   label: Name of the BigQuery column containing the numeric key business
#          objective that the model will predict.
#   start_date: If specified, restrict feature data to after this date.
#   max_lookback: Maximum number of days in a lookback window.
#   end_date: If specified, restrict feature data to before this date.
#   window_size: Size of a window (HOUR, DAY, WEEK).
#   num_prediction_periods: Make predictions for this many windows in the future.

{% include features_template %}

CREATE OR REPLACE TABLE `{{project_id}}.{{dataset_id}}.{{features_table}}`
AS (
  SELECT *
  FROM Features
  WHERE
    label_{{label}} IS NOT NULL
    AND `ts` BETWEEN (
      {% if start_ts %}
      SELECT TIMESTAMP_ADD('{{start_ts}}', INTERVAL {{max_lookback}} {{window_size}})
      {% else %}
      SELECT
        TIMESTAMP_ADD(MIN(`ts`), INTERVAL {{max_lookback}} {{window_size}})
      FROM DataTable
      {% endif %}
    ) AND (
      {% if end_ts %}
      SELECT TIMESTAMP_SUB('{{end_ts}}', INTERVAL {{num_prediction_periods}} {{window_size}})
      {% else %}
      SELECT
        TIMESTAMP_SUB(MAX(`ts`), INTERVAL {{num_prediction_periods}} {{window_size}})
      FROM DataTable
      {% endif %}
    )
  ORDER BY ts
);

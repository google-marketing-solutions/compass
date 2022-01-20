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
# SQL script for running model prediction.
# Args:
#   project_id: GCP project ID.
#   dataset_id: Dataset to write the output prediction table.
#   prediction_table: Name of the table to write the prediction table.
#   prediction_ts: Timestamp for the first prediction in YYYY-MM-DD:HH format.
#                  Predictions are made for prediction_periods in the future.

{% include features_template %}

CREATE OR REPLACE TABLE `{{project_id}}.{{dataset_id}}.{{prediction_table}}` AS (
  SELECT *
  FROM Features
  WHERE ts = PARSE_TIMESTAMP('%Y-%m-%d:%H', '{{prediction_ts}}')
);

CREATE OR REPLACE TABLE `{{project_id}}.{{dataset_id}}.{{prediction_table}}` AS (
  {% set input_features_table = prediction_table %}
  {% include prediction_template %}
);

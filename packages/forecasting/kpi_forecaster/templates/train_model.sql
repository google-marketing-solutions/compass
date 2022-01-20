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
# SQL script to train model.
# Args:
#   project_id: GCP project ID.
#   dataset_id: Dataset location of the model.
#   model_table: Name of the table to write the model.
#   input_features_table: Name of the input features table.
#   label: Name of the BigQuery column containing the numeric key business
#          objective that the model will predict.


CREATE OR REPLACE MODEL `{{project_id}}.{{dataset_id}}.{{model_table}}`
  OPTIONS(
    MODEL_TYPE='AUTOML_REGRESSOR',
    INPUT_LABEL_COLS=['label_{{label}}'],
    OPTIMIZATION_OBJECTIVE='MINIMIZE_MAE',
    BUDGET_HOURS=1.0)
AS (
  SELECT * EXCEPT(ts)
  FROM `{{project_id}}.{{dataset_id}}.{{input_features_table}}`
);

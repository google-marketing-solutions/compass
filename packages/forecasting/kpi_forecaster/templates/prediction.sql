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
# SQL script to run prediction with the given model over the given input features.
# Args:
#  project_id: GCP project ID.
#  dataset_id: Dataset location of the model.
#  model_table: Name of the model table.
#  input_features_table: Name of the input features table.


SELECT *
FROM ML.PREDICT(MODEL `{{project_id}}.{{dataset_id}}.{{model_table}}`,
                TABLE `{{project_id}}.{{dataset_id}}.{{input_features_table}}`)
ORDER BY ts, prediction_period

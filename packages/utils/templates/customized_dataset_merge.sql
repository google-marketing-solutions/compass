-- Copyright 2022 Google LLC
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- Merge the dataset generated from MLWP feature data and CRM data.

-- Query expects following parameters:
-- @param {merged_dataset_table}: Full path to BigQuery table where the output merged feature
--   dataset (MLWP features and user features) is restored.
-- @param {mlwp_feature_table}: Full path to BigQuery table where the pre-generated feature dataset
--   is restored.
-- @param {user_dataset_table}: Full path to BigQuery table where the pre-generated user (CRM)
--   dataset for modeling is restored. Ex: project.dataset.dataset.
-- @param {crm_data_date_start}: Start date of CRM data for building dataset. Ex: 2018-03-15.
-- @param {crm_data_date_end}: End date of CRM data for building dataset. Ex: 2021-09-15.
-- @param {crm_user_id}: Column name of user_id to join with window features. Ex: user_id.
-- @param {crm_snapshot_ts}: Column name of snapshot_ts to join with window features. Ex: snapshot_ts.

CREATE OR REPLACE TABLE `{merged_dataset_table}`
AS (
  SELECT
    WindowFeatures.*,
    CrmFeatures.* EXCEPT ({crm_user_id}, {crm_snapshot_ts}),
  FROM
    `{mlwp_feature_table}` AS WindowFeatures
  LEFT JOIN
    `{user_dataset_table}` AS CrmFeatures
    ON
      WindowFeatures.user_id = CrmFeatures.{crm_user_id}
      AND WindowFeatures.snapshot_ts = TIMESTAMP(CrmFeatures.{crm_snapshot_ts})
  WHERE CrmFeatures.{crm_snapshot_ts} BETWEEN {crm_data_date_start} AND {crm_data_date_end}
);

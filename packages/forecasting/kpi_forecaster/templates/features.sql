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
# SQL Script to generate features to train an ML model from the input data_query.
# Args:
#   data_query: SQL query to extract the raw user-level data.
#   training_mode: True to generate a training label from the data_query.
#                  False for prediction_mode, when no label is known.
#   label: Name of the BigQuery column containing the numeric key business
#          objective that the model will predict.
#   numeric_features: List of BigQuery column names in data_query to make
#                     numeric features from.
#   aggregate_functions: List of BigQuery aggregation functions to apply to the
#                        historical windows.
#   lookback_windows: List of windows. Numeric features are constructed over historical
#                     lookback window periods. Each window is specified with a pair
#                     (window_start, window_end), which corresponds to the range
#                     (today - window_start day) to (today + window_end days) inclusive.
#  window_size: Size of a window (HOUR, DAY, WEEK).
#  num_prediction_periods: Make predictions for this many windows in the future.
#  micros_per_window: Microseconds per window.

CREATE TEMPORARY TABLE DataTable AS (
  {{data_query}}
);

CREATE TEMPORARY TABLE BaseFeatures AS (
  SELECT
    ts,
    {% for feature in numeric_features %}
        {% for aggregate_function in aggregate_functions %}
          {% for (window_start, window_end) in lookback_windows %}
            {% if not (aggregate_function == 'AVG' and window_start == window_end) %}
      {{aggregate_function}}({{feature}}) OVER (
        ORDER BY UNIX_MICROS(ts) / {{micros_per_window}}
        RANGE BETWEEN
          {% if window_start == 0 %}
          CURRENT ROW
          {% else %}
          {{window_start}} PRECEDING
          {% endif %}
          {% if window_end == 0 %}
          AND CURRENT ROW
          {% else %}
          AND {{window_end}} PRECEDING
          {% endif %}
      ) AS {{feature}}_{{window_start}}_to_{{window_end}}_{{window_size}}sago_{{aggregate_function}},
            {% endif %}
          {% endfor %}
        {% endfor %}
      {% endfor %}
    FROM DataTable
);

{% set batch_size = 20 %}
{% for batch in range((num_prediction_periods // batch_size) +1) %}
CREATE TEMPORARY TABLE PredictionDayFeaturesBatch{{batch}} AS (
  {% for i in range(batch_size) %}
    {% if batch * batch_size + i <= num_prediction_periods %}
    {% set prediction_period = batch * batch_size + i %}
  SELECT
      ts,
      {{prediction_period}} AS prediction_period,
      {% if training_mode %}
      SUM({{label}}) OVER (
        ORDER BY UNIX_MICROS(ts) / {{micros_per_window}}
        RANGE BETWEEN
          {{prediction_period}} FOLLOWING
          AND {{prediction_period}} FOLLOWING
      ) AS label_{{label}},
      {% endif %}
      CONCAT(
        LPAD(
            EXTRACT(HOUR FROM TIMESTAMP_ADD(ts, INTERVAL {{prediction_period}} {{window_size}})),
            2, '0'
        ), 'H'
      ) AS hournum,
      CONCAT(
        LPAD(
            EXTRACT(
                DAYOFWEEK FROM TIMESTAMP_ADD(ts, INTERVAL {{prediction_period}} {{window_size}})),
            2, '0'
        ), 'D'
      ) AS weekday,
      CONCAT(
        LPAD(
          EXTRACT(WEEK FROM TIMESTAMP_ADD(ts, INTERVAL {{prediction_period}} {{window_size}})),
          2, '0'
        ), 'W'
      ) AS weeknum,
      CONCAT(
        LPAD(
            EXTRACT(MONTH FROM TIMESTAMP_ADD(ts, INTERVAL {{prediction_period}} {{window_size}})),
            2, '0'
        ), 'M'
      ) AS month,
    FROM DataTable
    {% if not loop.last and batch * batch_size + i < num_prediction_periods %}
    UNION ALL
    {% endif %}
  {% endif %}
{% endfor %}
);
{% endfor %}

CREATE TEMPORARY TABLE PredictionDayFeatures AS (
{% for batch in range((num_prediction_periods // batch_size) +1) %}
    SELECT * FROM PredictionDayFeaturesBatch{{batch}}
{% if not loop.last %}
    UNION ALL
{% endif %}
{% endfor %}
);

CREATE TEMPORARY TABLE Features AS (
  SELECT
    PredictionDayFeatures.*,
    BaseFeatures.* EXCEPT(ts)
  FROM PredictionDayFeatures JOIN BaseFeatures USING (ts)
  ORDER BY ts, prediction_period
);

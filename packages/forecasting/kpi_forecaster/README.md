# KPI Forecaster

The KPI Forecaster can predict the future value of key performance indicators
(KPIs), like sales, store visits, signups, leads, quotes etc. These forecasts
can help businesses manage inventory, workforce scheduling and
marketing budgets.

The KPI Forecaster runs on the Google Cloud Platform. It consists of two
[Jupyter Notebooks](https://jupyter.org/): one for building the forecaster
model, and one for using the model making predictions. Setting up the Notebooks
requires some knowledge of SQL to extract the historical data. After that,
the Notebooks can be run by anyone.

Under the hood, the KPI Forecaster is a <em>multivariate time series</em>em>
forecaster. This means it makes predictions based on multiple data sources, such
as historical website and/or marketing analytics, and historical values of the
target KPI etc. In contrast, a <em>univariate time series</em> forecaster only
uses the historical values of the target KPI.

Also, KPI Forecaster uses [AutoML](https://cloud.google.com/automl), which means
it can produce high-quality machine learning models without requiring a machine
learning expert.


## Example: Training an Forecaster

Here is an example of how to use the forecaster using publically-available obfuscated Google Analytics 360 [data](https://support.google.com/analytics/answer/7586738) from the Google
Merchandise Store. We will build a simple forecaster using the notebook
`model_training.ipynb` to predict the number of transactions per day.

The first step is to write a SQL `data_query` to extract the forecaster training data:

    SELECT
      TIMESTAMP_TRUNC(TIMESTAMP_SECONDS(visitStartTime), DAY) AS ts,
      IFNULL(SUM(totals.visits), 0) AS visits,
      IFNULL(SUM(totals.transactions), 0) AS transactions
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
    GROUP BY ts

Each row of the training data contains a `TIMESTAMP` `ts`, which is required,
and a collection of numeric features (`INT64` or `FLOAT64`) aggregated by
`TIMESTAMP` at either the `HOUR`, `DAY` or `WEEK` level. The query must also
extract the target label (i.e. the KPI to forecast). Here the label is `transactions`, though it can be named anything.

There are separate parameters to specify the column containing the `label`, and
all the `numeric_features` to include in the model. Make sure to include the
`label` in the numeric features, as historical values for the label are powerful
predictors of future values. Also, specify the `window_size` as either `HOUR`,
`DAY` or `WEEK` to match the `data_query` and to inform the forecaster over what
time period to make forecasts. Finally, use the corresponding `num_*_prediction_windows` parameter to specify how many time periods to
forecast in the future. For example, if `num_day_prediction_windows` is set to
28, then the forecaster will make predictions for every day up to 28 days in the
future.

## Example: Making Forecasts

Once a model has been trained, use the notebook `model_prediction.ipynb` to
make forecasts. Many of the parameters for this notebook are the same as for the
`model_training.ipynb` notebook. However, the `model_prediction.ipynb` notebook
needs the name of the model table, which by default is `model_{run_date}`, where
`{run_date}` is the `YYMMDD` the model was trained. By default, this is stored
in the `{project_id}.{dataset_id}` BigQuery dataset specified during training.
The `model_prediction.ipynb` also need the `prediction_ts`, which is the
`TIMESTAMP` for the first forecast. There must be a row in the `data_query` with
this exact `TIMESTAMP`, as the forecaster needs the data to make the forecast.

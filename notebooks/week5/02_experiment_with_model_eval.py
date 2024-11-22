# Databricks notebook source
# MAGIC %pip install ../power_consumption-0.0.1-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from databricks import feature_engineering
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from datetime import datetime
import mlflow
from pyspark.sql import DataFrame
from power_consumption.config import Config
from databricks.feature_engineering import FeatureFunction

# COMMAND ----------

config = Config.from_yaml("../../configs/project_configs.yml")

spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# COMMAND ----------

num_features = config.processed_features.num_features
cat_features = config.processed_features.cat_features
target = config.target.target

catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

serving_endpoint_name = "power-consumption-model-serving-fe"
serving_endpoint = workspace.serving_endpoints.get(serving_endpoint_name)
model_name = serving_endpoint.config.served_models[0].model_name
model_version = serving_endpoint.config.served_models[0].model_version
previous_model_uri = f"models:/{model_name}/{model_version}"

# COMMAND ----------

test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

# COMMAND ----------

test_set = test_set.withColumn("Temperature", test_set["Temperature"].cast("double"))
test_set = test_set.withColumn("Humidity", test_set["Humidity"].cast("double"))
test_set = test_set.withColumn("Wind_Speed", test_set["Wind_Speed"].cast("double"))

# COMMAND ----------

function_name = f"{catalog_name}.{schema_name}.calculate_weather_interaction"

testing_set = fe.create_training_set(
    df=test_set,
    label=target,
    feature_lookups=[
        FeatureFunction(
            udf_name=function_name,
            output_name="weather_interaction",
            input_bindings={"temperature": "Temperature", "humidity": "Humidity", "wind_speed": "Wind_Speed"},
        ),
    ],
    exclude_columns=["update_timestamp_utc"],
)

# COMMAND ----------

testing_df = testing_set.load_df()

# COMMAND ----------

X_test_spark = testing_df.select(["DateTime"] + num_features + cat_features + ["weather_interaction", "id"])
y_test_spark = testing_df.select(["id"] + target)

# COMMAND ----------

display(X_test_spark.limit(5))

# COMMAND ----------

new_model_uri =model_uri = 'runs:/2ef77d3a642e4b7ab50b759a40ae0425/lightgbm-pipeline-model-fe'

# COMMAND ----------

predictions_previous = fe.score_batch(model_uri=previous_model_uri, df=X_test_spark)
predictions_new = fe.score_batch(model_uri=new_model_uri, df=X_test_spark)

# COMMAND ----------

predictions_new = predictions_new.withColumnRenamed("prediction", "prediction_new")
predictions_old = predictions_previous.withColumnRenamed("prediction", "prediction_old")
test_set = test_set.select(["id"] + target)

# COMMAND ----------

df = test_set \
    .join(predictions_new, on="id") \
    .join(predictions_old, on="id")

# COMMAND ----------



# Calculate the absolute error for each model
df = df.withColumn("error_new", F.abs(df["Zone_1_Power_Consumption"] - df["prediction_new"]))
df = df.withColumn("error_old", F.abs(df["Zone_1_Power_Consumption"] - df["prediction_old"]))

# Calculate the absolute error for each model
df = df.withColumn("error_new", F.abs(df["Zone_1_Power_Consumption"] - df["prediction_new"]))
df = df.withColumn("error_old", F.abs(df["Zone_1_Power_Consumption"] - df["prediction_old"]))

# Calculate the Mean Absolute Error (MAE) for each model
mae_new = df.agg(F.mean("error_new")).collect()[0][0]
mae_old = df.agg(F.mean("error_old")).collect()[0][0]

# COMMAND ----------

evaluator = RegressionEvaluator(labelCol="Zone_1_Power_Consumption", predictionCol="prediction_new", metricName="rmse")
rmse_new = evaluator.evaluate(df)

evaluator.setPredictionCol("prediction_old")
rmse_old = evaluator.evaluate(df)

# COMMAND ----------

print(f"MAE for New Model: {mae_new}")
print(f"MAE for Old Model: {mae_old}")

# COMMAND ----------

if mae_new < mae_old:
    print("New model is better based on MAE.")
    model_version = mlflow.register_model(
        model_uri=new_model_uri,
        name=f"{catalog_name}.{schema_name}.house_prices_model_fe",
        tags={"git_sha": f"{git_sha}",
            "job_run_id": job_run_id})

    print("New model registered with version:", model_version.version)
    dbutils.jobs.taskValues.set(key="model_version", value=model_version.version)
    dbutils.jobs.taskValues.set(key="model_update", value=1)
else:
    print("Old model is better based on MAE.")
    dbutils.jobs.taskValues.set(key="model_update", value=0)

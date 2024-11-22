"""Evaluate the new model and register it if it's better than the old model."""

import argparse

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction
from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from power_consumption.config import Config
from power_consumption.utils import get_dbutils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path",
    action="store",
    default=None,
    type=str,
    required=True,
)
parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)
parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)
parser.add_argument(
    "--new_model_uri",
    action="store",
    default=None,
    type=str,
    required=True,
)


args = parser.parse_args()
config_path = args.config_path
git_sha = args.git_sha
job_run_id = args.job_run_id
new_model_uri = args.new_model_uri


spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()


mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

dbutils = get_dbutils(spark)


config = Config.from_yaml(config_path)
num_features = config.processed_features.num_features
cat_features = config.processed_features.cat_features
target = config.target.target

catalog_name = config.catalog_name
schema_name = config.schema_name

serving_endpoint_name = "power-consumption-model-serving-fe"
serving_endpoint = workspace.serving_endpoints.get(serving_endpoint_name)
model_name = serving_endpoint.config.served_models[0].model_name
model_version = serving_endpoint.config.served_models[0].model_version
previous_model_uri = f"models:/{model_name}/{model_version}"


test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")


test_set = test_set.withColumn("Temperature", test_set["Temperature"].cast("double"))
test_set = test_set.withColumn("Humidity", test_set["Humidity"].cast("double"))
test_set = test_set.withColumn("Wind_Speed", test_set["Wind_Speed"].cast("double"))


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


testing_df = testing_set.load_df()


X_test_spark = testing_df.select(["DateTime"] + num_features + cat_features + ["weather_interaction", "id"])
y_test_spark = testing_df.select(["id"] + target)


new_model_uri = model_uri = "runs:/2ef77d3a642e4b7ab50b759a40ae0425/lightgbm-pipeline-model-fe"


predictions_previous = fe.score_batch(model_uri=previous_model_uri, df=X_test_spark)
predictions_new = fe.score_batch(model_uri=new_model_uri, df=X_test_spark)


predictions_new = predictions_new.withColumnRenamed("prediction", "prediction_new")
predictions_old = predictions_previous.withColumnRenamed("prediction", "prediction_old")
test_set = test_set.select(["id"] + target)


df = test_set.join(predictions_new, on="id").join(predictions_old, on="id")


# Calculate the absolute error for each model
df = df.withColumn("error_new", F.abs(df["Zone_1_Power_Consumption"] - df["prediction_new"]))
df = df.withColumn("error_old", F.abs(df["Zone_1_Power_Consumption"] - df["prediction_old"]))

# Calculate the absolute error for each model
df = df.withColumn("error_new", F.abs(df["Zone_1_Power_Consumption"] - df["prediction_new"]))
df = df.withColumn("error_old", F.abs(df["Zone_1_Power_Consumption"] - df["prediction_old"]))

# Calculate the Mean Absolute Error (MAE) for each model
mae_new = df.agg(F.mean("error_new")).collect()[0][0]
mae_old = df.agg(F.mean("error_old")).collect()[0][0]


evaluator = RegressionEvaluator(labelCol="Zone_1_Power_Consumption", predictionCol="prediction_new", metricName="rmse")
rmse_new = evaluator.evaluate(df)

evaluator.setPredictionCol("prediction_old")
rmse_old = evaluator.evaluate(df)


logger.info(f"MAE for New Model: {mae_new}")
logger.info(f"MAE for Old Model: {mae_old}")


if mae_new < mae_old:
    logger.info("New model is better based on MAE.")
    model_version = mlflow.register_model(
        model_uri=new_model_uri,
        name=f"{catalog_name}.{schema_name}.power_consumption_model_fe",
        tags={"git_sha": f"{git_sha}", "job_run_id": job_run_id},
    )

    logger.info("New model registered with version:", model_version.version)
    dbutils.jobs.taskValues.set(key="model_version", value=model_version.version)
    dbutils.jobs.taskValues.set(key="model_update", value=1)
else:
    logger.info("Old model is better based on MAE.")
    dbutils.jobs.taskValues.set(key="model_update", value=0)

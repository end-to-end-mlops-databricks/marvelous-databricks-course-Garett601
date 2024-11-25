"""Train the model for the power consumption project."""

import argparse

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMRegressor
from loguru import logger
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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

args = parser.parse_args()
config_path = args.config_path
git_sha = args.git_sha
job_run_id = args.job_run_id

spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()


mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")


config = Config.from_yaml(config_path)

num_features = config.processed_features.num_features
cat_features = config.processed_features.cat_features
target = config.target.target
parameters = config.hyperparameters.__dict__

catalog_name = config.catalog_name
schema_name = config.schema_name


feature_table_name = f"{catalog_name}.{schema_name}.power_consumption_features"
function_name = f"{catalog_name}.{schema_name}.calculate_weather_interaction"


train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").drop("general_diffuse_flows", "diffuse_flows")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

train_set = train_set.withColumn("Temperature", train_set["Temperature"].cast("double"))
train_set = train_set.withColumn("Humidity", train_set["Humidity"].cast("double"))
train_set = train_set.withColumn("Wind_Speed", train_set["Wind_Speed"].cast("double"))

test_set = test_set.withColumn("Temperature", test_set["Temperature"].cast("double"))
test_set = test_set.withColumn("Humidity", test_set["Humidity"].cast("double"))
test_set = test_set.withColumn("Wind_Speed", test_set["Wind_Speed"].cast("double"))


training_set = fe.create_training_set(
    df=train_set,
    label=target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["general_diffuse_flows", "diffuse_flows"],
            lookup_key="id",
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="weather_interaction",
            input_bindings={"temperature": "Temperature", "humidity": "Humidity", "wind_speed": "Wind_Speed"},
        ),
    ],
    exclude_columns=["update_timestamp_utc"],
)


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


training_df = training_set.load_df().toPandas()
testing_df = testing_set.load_df().toPandas()


X_train = training_df[num_features + cat_features + ["weather_interaction"]]
y_train = training_df[target]

X_test = testing_df[num_features + cat_features + ["weather_interaction"]]
y_test = testing_df[target]


preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)], remainder="passthrough"
)

pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", MultiOutputRegressor(LGBMRegressor(**parameters)))]
)


mlflow.set_experiment(experiment_name="/Shared/power-consumption-fe")

with mlflow.start_run(tags={"git_sha": f"{git_sha}", "job_run_id": job_run_id, "branch": "feature/week5"}) as run:
    run_id = run.info.run_id

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate the model performance
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Mean Squared Error: {mse}")
    logger.info(f"Mean Absolute Error: {mae}")
    logger.info(f"R2 Score: {r2}")

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # Log model with feature engineering
    fe.log_model(
        model=pipeline,
        flavor=mlflow.sklearn,
        artifact_path="lightgbm-pipeline-model-fe",
        training_set=training_set,
        signature=signature,
    )

model_uri = f"runs:/{run_id}/lightgbm-pipeline-model-fe"
dbutils = get_dbutils(spark)
dbutils.jobs.taskValues.set(key="new_model_uri", value=model_uri)

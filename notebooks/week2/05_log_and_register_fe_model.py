# Databricks notebook source
# MAGIC %pip install ../power_consumption-0.0.1-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import yaml
import mlflow
from mlflow.models import infer_signature

from datetime import datetime

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder

from lightgbm import LGBMRegressor

from power_consumption.config import Config

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

config = Config.from_yaml("../../configs/project_configs.yml")

num_features = config.processed_features.num_features
cat_features = config.processed_features.cat_features
target = config.target.target
parameters = config.hyperparameters.__dict__

catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

# Define table names and function name
feature_table_name = f"{catalog_name}.{schema_name}.power_consumption_features"
function_name = f"{catalog_name}.{schema_name}.calculate_weather_interaction"

# COMMAND ----------

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {feature_table_name} (
    DateTime TIMESTAMP NOT NULL,
    Temperature DOUBLE,
    Humidity DOUBLE,
    Wind_Speed DOUBLE,
    general_diffuse_flows DOUBLE,
    diffuse_flows DOUBLE,
    DayOfWeek_1 INT,
    DayOfWeek_2 INT,
    DayOfWeek_3 INT,
    DayOfWeek_4 INT,
    DayOfWeek_5 INT,
    DayOfWeek_6 INT,
    IsWeekend_1 INT
);
""")

# Add primary key constraint
spark.sql(f"""
    ALTER TABLE {feature_table_name}
    ADD CONSTRAINT power_consumption_pk
    PRIMARY KEY(DateTime)
""")

# Enable Change Data Feed
spark.sql(f"""
    ALTER TABLE {feature_table_name}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# COMMAND ----------

spark.sql(f"""
INSERT INTO {catalog_name}.{schema_name}.power_consumption_features
SELECT
    DateTime,
    Temperature,
    Humidity,
    Wind_Speed,
    general_diffuse_flows,
    diffuse_flows,
    DayOfWeek_1,
    DayOfWeek_2,
    DayOfWeek_3,
    DayOfWeek_4,
    DayOfWeek_5,
    DayOfWeek_6,
    IsWeekend_1
FROM {catalog_name}.{schema_name}.train_set
""")

# Insert data from test_set
spark.sql(f"""
INSERT INTO {catalog_name}.{schema_name}.power_consumption_features
SELECT
    DateTime,
    Temperature,
    Humidity,
    Wind_Speed,
    general_diffuse_flows,
    diffuse_flows,
    DayOfWeek_1,
    DayOfWeek_2,
    DayOfWeek_3,
    DayOfWeek_4,
    DayOfWeek_5,
    DayOfWeek_6,
    IsWeekend_1
FROM {catalog_name}.{schema_name}.test_set
""")

# COMMAND ----------

# Define a function to calculate the weather interaction
spark.sql(f"""
CREATE OR REPLACE FUNCTION {function_name}(temperature DOUBLE, humidity DOUBLE, wind_speed DOUBLE)
RETURNS DOUBLE
LANGUAGE PYTHON AS
$$
return temperature * humidity * wind_speed
$$
""")

# COMMAND ----------

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").drop("general_diffuse_flows", "diffuse_flows")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

# COMMAND ----------

# Cast relevant columns to double for the function input
train_set = train_set.withColumn("Temperature", train_set["Temperature"].cast("double"))
train_set = train_set.withColumn("Humidity", train_set["Humidity"].cast("double"))
train_set = train_set.withColumn("Wind_Speed", train_set["Wind_Speed"].cast("double"))

test_set = test_set.withColumn("Temperature", test_set["Temperature"].cast("double"))
test_set = test_set.withColumn("Humidity", test_set["Humidity"].cast("double"))
test_set = test_set.withColumn("Wind_Speed", test_set["Wind_Speed"].cast("double"))

# COMMAND ----------

training_set = fe.create_training_set(
    df=train_set,
    label=target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["general_diffuse_flows", "diffuse_flows"],
            lookup_key="DateTime",
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="weather_interaction",
            input_bindings={
                "temperature": "Temperature",
                "humidity": "Humidity",
                "wind_speed": "Wind_Speed"
            },
        ),
    ],
    exclude_columns=["update_timestamp_utc"]
)

# COMMAND ----------

testing_set = fe.create_training_set(
    df=test_set,
    label=target,
    feature_lookups=[
        FeatureFunction(
            udf_name=function_name,
            output_name="weather_interaction",
            input_bindings={
                "temperature": "Temperature",
                "humidity": "Humidity",
                "wind_speed": "Wind_Speed"
            },
        ),
    ],
    exclude_columns=["update_timestamp_utc"]
)

# COMMAND ----------

training_df = training_set.load_df().toPandas()
testing_df = testing_set.load_df().toPandas()

# COMMAND ----------

X_train = training_df[num_features + cat_features + ["weather_interaction"]]
y_train = training_df[target]

X_test= testing_df[num_features + cat_features + ["weather_interaction"]]
y_test = testing_df[target]

# COMMAND ----------

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)],
    remainder='passthrough'
)

# Create the pipeline with preprocessing and the multi-output LightGBM regressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MultiOutputRegressor(LGBMRegressor(**parameters)))
])

# COMMAND ----------

mlflow.set_experiment(experiment_name='/Shared/power-consumption-fe')
git_sha = "30d57afb2efca70cede3061d00f2a553c2b4779b"

with mlflow.start_run(
    tags={"git_sha": f"{git_sha}",
        "branch": "feature/week2"},
) as run:
    run_id = run.info.run_id

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate the model performance
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")

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

mlflow.register_model(
model_uri=f'runs:/{run_id}/lightgbm-pipeline-model-fe',
name=f"{catalog_name}.{schema_name}.power_consumption_model_fe")

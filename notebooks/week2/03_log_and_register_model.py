# Databricks notebook source
import mlflow
from mlflow.models import infer_signature

from pyspark.sql import SparkSession
from power_consumption.config import Config

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
# COMMAND ----------
config = Config.from_yaml("../../configs/project_configs.yml")
# COMMAND ----------
num_features = config.processed_features.num_features
cat_features = config.processed_features.cat_features
target = config.target.target
parameters = config.hyperparameters.__dict__

catalog_name = config.catalog_name
schema_name = config.schema_name
# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
# COMMAND ----------
X_train = train_set[num_features + cat_features]
y_train = train_set[target]

X_test = test_set[num_features + cat_features]
y_test = test_set[target]
# COMMAND ----------
# Define the preprocessor for categorical features
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
mlflow.set_experiment(experiment_name='/Shared/power-consumption')
git_sha = "30d57afb2efca70cede3061d00f2a553c2b4779b"

# Start an MLflow run to track the training process
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

    dataset = mlflow.data.from_spark(
    train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set",
    version="0")
    mlflow.log_input(dataset, context="training")

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="lightgbm-pipeline-model",
        signature=signature
    )
# COMMAND ----------
model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/lightgbm-pipeline-model',
    name=f"{catalog_name}.{schema_name}.power_consumption_model",
    tags={"git_sha": f"{git_sha}"})

# COMMAND ----------
run = mlflow.get_run(run_id)
dataset_info = run.inputs.dataset_inputs[0].dataset
dataset_source = mlflow.data.get_source(dataset_info)
dataset_source.load()

# COMMAND ----------

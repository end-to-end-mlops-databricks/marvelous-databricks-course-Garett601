# Databricks notebook source
# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

# MAGIC %pip install ../power_consumption-0.0.1-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,imports
import time

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import hashlib
import requests

from power_consumption.config import Config

# COMMAND ----------

# DBTITLE 1,initialisations
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

client = MlflowClient()

spark = SparkSession.builder.getOrCreate()

config = Config.from_yaml("../../configs/project_configs.yml")

catalog_name = config.catalog_name
schema_name = config.schema_name

num_features = config.processed_features.num_features
cat_features = config.processed_features.cat_features
target = config.target.target
ab_test_params = config.ab_test_hyperparameters.__dict__

# COMMAND ----------

# DBTITLE 1,get a/b parameters
parameters_a = {
    "learning_rate": ab_test_params["learning_rate_a"],
    "n_estimators": ab_test_params["n_estimators"],
    "max_depth": ab_test_params["max_depth_a"],
}

parameters_b = {
    "learning_rate": ab_test_params["learning_rate_b"],
    "n_estimators": ab_test_params["n_estimators"],
    "max_depth": ab_test_params["max_depth_b"],
}

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Loading

# COMMAND ----------

train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
train_set = train_set_spark.toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# COMMAND ----------

# DBTITLE 1,define features and target
X_train = train_set[num_features + cat_features]
y_train = train_set[target]
X_test = test_set[num_features + cat_features]
y_test = test_set[target]

# COMMAND ----------

# MAGIC %md
# MAGIC # Train and Log Models

# COMMAND ----------

# DBTITLE 1,create pipelines
# Define the preprocessor for categorical features
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)],
    remainder='passthrough'
)

# Create the pipeline with preprocessing and the multi-output LightGBM regressor
pipeline_a = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MultiOutputRegressor(LGBMRegressor(**parameters_a)))
])

pipeline_b = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MultiOutputRegressor(LGBMRegressor(**parameters_b)))
])

# Set the MLflow experiment to track this A/B testing project
mlflow.set_experiment(experiment_name="/Shared/power-consumption-ab")
model_name = f"{catalog_name}.{schema_name}.power_consumption_model_ab"

git_sha = "30d57afb2efca70cede3061d00f2a553c2b4779b"

# COMMAND ----------

# DBTITLE 1,Train Model A and Log with MLflow
# Start MLflow run to track training of Model A
with mlflow.start_run(tags={"model_class": "A", "git_sha": git_sha}) as run:
    run_id = run.info.run_id

    # Train the model
    pipeline_a.fit(X_train, y_train)
    y_pred = pipeline_a.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log model parameters, metrics, and other artifacts in MLflow
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters_a)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # Log the input dataset for tracking reproducibility
    dataset = mlflow.data.from_spark(train_set_spark,
                                     table_name=f"{catalog_name}.{schema_name}.train_set",
                                     version="0")
    mlflow.log_input(dataset, context="training")

    # Log the pipeline model in MLflow with a unique artifact path
    mlflow.sklearn.log_model(sk_model=pipeline_a, artifact_path="lightgbm-pipeline-model", signature=signature)

model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/lightgbm-pipeline-model", name=model_name, tags={"git_sha": f"{git_sha}"}
)

# COMMAND ----------

# DBTITLE 1,Register Model A and Assign Alias
model_version_alias = "model_A"

client.set_registered_model_alias(model_name, model_version_alias, f"{model_version.version}")
model_uri = f"models:/{model_name}@{model_version_alias}"
model_A = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

# DBTITLE 1,Train Model B and Log with MLflow
# Start MLflow run to track training of Model B
with mlflow.start_run(tags={"model_class": "B", "git_sha": git_sha}) as run:
    run_id = run.info.run_id

    # Train the model
    pipeline_b.fit(X_train, y_train)
    y_pred = pipeline_b.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log model parameters, metrics, and other artifacts in MLflow
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters_b)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # Log the input dataset for tracking reproducibility
    dataset = mlflow.data.from_spark(train_set_spark,
                                     table_name=f"{catalog_name}.{schema_name}.train_set",
                                     version="0")
    mlflow.log_input(dataset, context="training")

    # Log the pipeline model in MLflow with a unique artifact path
    mlflow.sklearn.log_model(sk_model=pipeline_b, artifact_path="lightgbm-pipeline-model", signature=signature)

model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/lightgbm-pipeline-model", name=model_name, tags={"git_sha": f"{git_sha}"}
)

# COMMAND ----------

# DBTITLE 1,Register Model B and Assign Alias
model_version_alias = "model_B"

client.set_registered_model_alias(model_name, model_version_alias, f"{model_version.version}")
model_uri = f"models:/{model_name}@{model_version_alias}"
model_B = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC # Define Custom A/B Test Model

# COMMAND ----------

class PowerConsumptionModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, models):
        self.models = models
        self.model_a = models[0]
        self.model_b = models[1]

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            time_id = str(model_input["id"].values[0])
            hashed_id = hashlib.md5(time_id.encode(encoding="UTF-8")).hexdigest()
            # convert a hexadecimal (base-16) string into an integer
            if int(hashed_id, 16) % 2:
                predictions = self.model_a.predict(model_input.drop(["id"], axis=1))
                return {"Prediction": predictions[0], "model": "Model A"}
            else:
                predictions = self.model_b.predict(model_input.drop(["id"], axis=1))
                return {"Prediction": predictions[0], "model": "Model B"}
        else:
            raise ValueError("Input must be a pandas DataFrame.")

# COMMAND ----------

X_train = train_set[num_features + cat_features + ["id"]]
X_test = test_set[num_features + cat_features + ["id"]]

# COMMAND ----------

models = [model_A, model_B]
wrapped_model = PowerConsumptionModelWrapper(models)
example_input_1 = X_test.iloc[0:1]
example_prediction_1 = wrapped_model.predict(
    context=None,
    model_input=example_input_1)
example_input_2 = X_test.iloc[1:2]
example_prediction_2 = wrapped_model.predict(
    context=None,
    model_input=example_input_2)
print("Example Prediction:", example_prediction_1)
print("Example Prediction:", example_prediction_2)

# COMMAND ----------

# MAGIC %md
# MAGIC # Log, Register and Use Models

# COMMAND ----------

# DBTITLE 1,log and register model
mlflow.set_experiment(experiment_name="/Shared/power-consumption-ab-testing")
model_name = f"{catalog_name}.{schema_name}.power_consumption_model_pyfunc_ab_test"

with mlflow.start_run() as run:
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train,
                                model_output={"Prediction": 1234.5,
                                              "model": "Model B"})
    dataset = mlflow.data.from_spark(train_set_spark,
                                     table_name=f"{catalog_name}.{schema_name}.train_set",
                                     version="0")
    mlflow.log_input(dataset, context="training")
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="pyfunc-power-consumption-model-ab",
        signature=signature
    )
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/pyfunc-power-consumption-model-ab",
    name=model_name,
    tags={"git_sha": f"{git_sha}"}
)

# COMMAND ----------

# DBTITLE 1,load and predict
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version.version}")

predictions = model.predict(X_test.iloc[0:1])

predictions

# COMMAND ----------

# MAGIC %md
# MAGIC # Create serving endpoint

# COMMAND ----------

workspace = WorkspaceClient()

workspace.serving_endpoints.create(
    name="power-consumption-model-serving-ab-test",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog_name}.{schema_name}.power_consumption_model_pyfunc_ab_test",
                scale_to_zero_enabled=True,
                workload_size="Small",
                entity_version=model_version.version,
            )
        ]
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Call the endpoint

# COMMAND ----------

# DBTITLE 1,set token and host
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

# DBTITLE 1,create sample request body
required_columns = config.processed_features.num_features + config.processed_features.cat_features + ["id"]

sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")

dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

start_time = time.time()

model_serving_endpoint = (
    f"https://{host}/serving-endpoints/power-consumption-model-serving-ab-test/invocations"
)

response = requests.post(
    f"{model_serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": dataframe_records[0]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")

# COMMAND ----------

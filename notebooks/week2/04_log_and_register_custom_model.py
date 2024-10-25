# Databricks notebook source
import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from mlflow.models import infer_signature
from power_consumption.config import Config
import json
from mlflow import MlflowClient
from mlflow.utils.environment import _mlflow_conda_env

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
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
# COMMAND ----------
run_id = mlflow.search_runs(
    experiment_names=["/Shared/power-consumption"],
    filter_string="tags.branch='feature/week2'",
).run_id[0]

model = mlflow.sklearn.load_model(f'runs:/{run_id}/lightgbm-pipeline-model')
# COMMAND ----------
print(run_id)

# COMMAND ----------
def adjust_predictions(predictions, scale_factor=1.3):
    return predictions * scale_factor

class PowerConsumptionModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            predictions = self.model.predict(model_input)
            predictions = {"Prediction": adjust_predictions(
                predictions[0])}
            return predictions
        else:
            raise ValueError("Input must be a pandas DataFrame.")
# COMMAND ----------
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

X_train = train_set[num_features + cat_features].toPandas()
y_train = train_set[[target]].toPandas()

X_test = test_set[num_features + cat_features].toPandas()
y_test = test_set[[target]].toPandas()
# COMMAND ----------
wrapped_model = PowerConsumptionModelWrapper(model) # we pass the loaded model to the wrapper
example_input = X_test.iloc[0:1]  # Select the first row for prediction as example
example_prediction = wrapped_model.predict(context=None, model_input=example_input)
print("Example Prediction:", example_prediction)
# COMMAND ----------
# COMMAND ----------
# IMPORTANT
# this is a trick with custom packages
# https://docs.databricks.com/en/machine-learning/model-serving/private-libraries-model-serving.html
# but does not work with pyspark, so we have a better option :-)

mlflow.set_experiment(experiment_name="/Shared/power-consumption-pyfunc")
git_sha = "30d57afb2efca70cede3061d00f2a553c2b4779b"

with mlflow.start_run(tags={"branch": "feature/week2",
                            "git_sha": f"{git_sha}"}) as run:

    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train, model_output={'Prediction': example_prediction})
    dataset = mlflow.data.from_spark(
        train_set, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["code/power_consumption-0.0.1-py3-none-any.whl",
                            ],
        additional_conda_channels=None,
    )
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="pyfunc-power-consumption-model",
        code_paths = ["../../dist/power_consumption-0.0.1-py3-none-any.whl"],
        signature=signature
    )
# COMMAND ----------
print(f"runs:/{run_id}/pyfunc-power-consumption-model")
# COMMAND ----------
loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/pyfunc-power-consumption-model")
loaded_model.unwrap_python_model()

# COMMAND ----------
model_name = f"{catalog_name}.{schema_name}.power_consumption_model_pyfunc"

model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/pyfunc-power-consumption-model',
    name=model_name,
    tags={"git_sha": f"{git_sha}"})
# COMMAND ----------
with open("model_version.json", "w") as json_file:
    json.dump(model_version.__dict__, json_file, indent=4)
# COMMAND ----------
model_version_alias = "the_bestest_model"
client.set_registered_model_alias(model_name, model_version_alias, "1")

model_uri = f"models:/{model_name}@{model_version_alias}"
model = mlflow.pyfunc.load_model(model_uri)
# COMMAND ----------
client.get_model_version_by_alias(model_name, model_version_alias)
# COMMAND ----------
model
# COMMAND ----------

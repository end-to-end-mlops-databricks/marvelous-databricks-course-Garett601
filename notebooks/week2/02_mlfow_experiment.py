# Databricks notebook source
import json

import mlflow

mlflow.set_tracking_uri("databricks")

mlflow.set_experiment(experiment_name="/Shared/power-consumption")
mlflow.set_experiment_tags({"repository_name": "power-consumption"})

# COMMAND ----------
experiments = mlflow.search_experiments(
    filter_string="tags.repository_name='power-consumption'"
)

print(experiments)

# COMMAND ----------
with open("mlflow_experiment.json", "w") as json_file:
    json.dump(experiments[0].__dict__, json_file, indent=4)

# COMMAND ----------
with mlflow.start_run(
    run_name="test-run",
    tags={
        "git_sha": "30d57afb2efca70cede3061d00f2a553c2b4779b"
    }
) as run:
    mlflow.log_params({"type": "demo"})
    mlflow.log_metrics(
        {
            "metric_1": 1.0,
            "metric_2": 2.0
        }
    )
# COMMAND ----------
run_id = mlflow.search_runs(
    experiment_names=["/Shared/power-consumption"],
    filter_string="tags.git_sha='30d57afb2efca70cede3061d00f2a553c2b4779b'",
).run_id[0]
run_info = mlflow.get_run(run_id=f"{run_id}").to_dictionary()
print(run_info)

# COMMAND ----------
with open("run_info.json", "w") as json_file:
    json.dump(run_info, json_file, indent=4)

# COMMAND ----------
print(run_info["data"]["metrics"])

# COMMAND ----------
print(run_info["data"]["params"])
# COMMAND ----------

# Databricks notebook source
from power_consumption.preprocessing.data_preprocessor import DataProcessor
from power_consumption.model.rf_model import ConsumptionModel
from power_consumption.utils import visualise_results, plot_actual_vs_predicted, plot_feature_importance
from power_consumption.config import Config
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()


# COMMAND ----------

config = Config.from_yaml("../../configs/project_configs.yml")

# COMMAND ----------
catalog_name = config.catalog_name
schema_name = config.schema_name
raw_data_table = config.dataset.raw_data_table
# COMMAND ----------
data_spark = spark.table(f"{catalog_name}.{schema_name}.{raw_data_table}")
# COMMAND ----------
data_pandas = data_spark.toPandas()
# COMMAND ----------
data_processor = DataProcessor(config, data_pandas)
# COMMAND ----------
data_processor.preprocess_data()
# COMMAND ----------
train_set, test_set = data_processor.split_data()

# COMMAND ----------
target_columns = config.target.target
feature_columns = config.processed_features.num_features + config.processed_features.cat_features

X_train = train_set[feature_columns]
y_train = train_set[target_columns]
X_test = test_set[feature_columns]
y_test = test_set[target_columns]

# COMMAND ----------
model = ConsumptionModel(config)
model.train(X_train, y_train)

# COMMAND ----------

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
mse, r2 = model.evaluate(X_test, y_test)

# COMMAND ----------

# Visualize results as time series
visualise_results(y_test, y_pred, target_columns)

# COMMAND ----------

# Get feature importance
feature_importance, feature_names = model.get_feature_importance()
# COMMAND ----------

# Plot actual vs predicted values
plot_actual_vs_predicted(y_test.values, y_pred, target_columns)

# COMMAND ----------

# Plot feature importance
plot_feature_importance(feature_importance, feature_names, top_n=15)

# COMMAND ----------

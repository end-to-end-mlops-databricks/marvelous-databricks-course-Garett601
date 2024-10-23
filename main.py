"""Main script for the power consumption project."""

from loguru import logger

from power_consumption.config import Config
from power_consumption.model import ConsumptionModel
from power_consumption.preprocessing import DataProcessor
from power_consumption.utils import plot_actual_vs_predicted, plot_feature_importance, visualise_results

# Load configuration
config = Config.from_yaml("./configs/project_configs.yml")

# Initialise data processor and preprocess data
processor = DataProcessor(config=config)
processor.preprocess_data()

# Split data into train and test sets
X_train, X_test, y_train, y_test = processor.split_data()

# Transform features
X_train_transformed, X_test_transformed = processor.fit_transform_features(X_train, X_test)

# Initialise and train the model
model = ConsumptionModel(processor.preprocessor, config)
model.train(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse, r2 = model.evaluate(X_test, y_test)

# Log Mean Squared Error for each target
logger.info("Mean Squared Error for each target:")
for i, target in enumerate(config.target.target):
    logger.info(f"{target}: {mse[i]:.4f}")

# Log R-squared for each target
logger.info("\nR-squared for each target:")
for i, target in enumerate(config.target.target):
    logger.info(f"{target}: {r2[i]:.4f}")

# Visualise results
visualise_results(y_test, y_pred, config.target.target)
plot_actual_vs_predicted(y_test.values, y_pred, config.target.target)

# Get and plot feature importance
feature_importance, feature_names = model.get_feature_importance()
plot_feature_importance(feature_importance, feature_names)

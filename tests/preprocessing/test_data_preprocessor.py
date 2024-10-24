import pandas as pd
import numpy as np
import pytest
from power_consumption.preprocessing.data_preprocessor import DataProcessor
from power_consumption.schemas.processed_data import ProcessedPowerConsumptionSchema


@pytest.fixture
def sample_data():
    """
    Create a sample DataFrame for testing.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing sample data for testing.
    """
    date_range = pd.date_range(start="2023-01-01", end="2023-01-10", freq="h")
    data = {
        "DateTime": date_range,
        "Temperature": np.random.uniform(0, 30, len(date_range)),
        "Humidity": np.random.uniform(30, 90, len(date_range)),
        "Wind_Speed": np.random.uniform(0, 20, len(date_range)),
        "general_diffuse_flows": np.random.uniform(0, 100, len(date_range)),
        "diffuse_flows": np.random.uniform(0, 100, len(date_range)),
        "Zone_1_Power_Consumption": np.random.uniform(100, 500, len(date_range)),
        "Zone_2_Power_Consumption": np.random.uniform(100, 500, len(date_range)),
        "Zone_3_Power_Consumption": np.random.uniform(100, 500, len(date_range)),
    }
    return pd.DataFrame(data)

@pytest.fixture
def data_processor(project_config, sample_data):
    """
    Create a DataProcessor instance with sample data for testing.

    Parameters
    ----------
    project_config : Config
        The project configuration.
    sample_data : pd.DataFrame
        The sample data for testing.

    Returns
    -------
    DataProcessor
        An instance of DataProcessor initialized with sample data.
    """
    return DataProcessor(config=project_config, data=sample_data)

def test_clean_column_names(data_processor):
    """
    Test the _clean_column_names method of DataProcessor.

    Parameters
    ----------
    data_processor : DataProcessor
        The DataProcessor instance to test.
    """
    data_processor._clean_column_names()
    assert "Zone_2_Power_Consumption" in data_processor.data.columns
    assert "Zone_3_Power_Consumption" in data_processor.data.columns

def test_create_preprocessor(data_processor):
    """
    Test the _create_preprocessor method of DataProcessor.

    Parameters
    ----------
    data_processor : DataProcessor
        The DataProcessor instance to test.
    """
    preprocessor = data_processor._create_preprocessor()
    assert preprocessor is not None
    assert len(preprocessor.transformers) == 2

def test_preprocess_data(data_processor):
    data_processor.preprocess_data()

    # Check that the index is DateTime
    assert data_processor.data.index.name == "DateTime"

    # Check for the presence of all columns defined in the schema
    expected_columns = [
        "Temperature", "Humidity", "Wind_Speed", "Hour", "Day", "Month",
        "general_diffuse_flows", "diffuse_flows",
        "DayOfWeek_1", "DayOfWeek_2", "DayOfWeek_3", "DayOfWeek_4", "DayOfWeek_5", "DayOfWeek_6",
        "IsWeekend_1",
        "Zone_1_Power_Consumption", "Zone_2_Power_Consumption", "Zone_3_Power_Consumption"
    ]

    for column in expected_columns:
        assert column in data_processor.data.columns, f"Column {column} is missing from the processed data"

    # Validate the entire DataFrame against the schema
    try:
        ProcessedPowerConsumptionSchema.validate(data_processor.data)
    except Exception as e:
        pytest.fail(f"Schema validation failed: {str(e)}")


def test_split_data(data_processor):
    data_processor.preprocess_data()
    train_set, test_set = data_processor.split_data(test_size=0.2)

    assert len(train_set) > len(test_set)
    assert train_set.shape[1] == test_set.shape[1]
    assert train_set.index[0] < test_set.index[0]  # Check temporal order
    assert train_set.index[-1] < test_set.index[0]  # Ensure no overlap

    # Validate train and test sets using the ProcessedPowerConsumptionSchema
    try:
        ProcessedPowerConsumptionSchema.validate(train_set)
        ProcessedPowerConsumptionSchema.validate(test_set)
    except Exception as e:
        pytest.fail(f"Schema validation failed: {str(e)}")

import pandas as pd
import numpy as np
import pytest
from power_consumption.preprocessing.data_preprocessor import DataProcessor

@pytest.fixture
def sample_data():
    date_range = pd.date_range(start='2023-01-01', end='2023-01-10', freq='H')
    data = {
        'DateTime': date_range,
        'Temperature': np.random.uniform(0, 30, len(date_range)),
        'Humidity': np.random.uniform(30, 90, len(date_range)),
        'Wind_Speed': np.random.uniform(0, 20, len(date_range)),
        'general_diffuse_flows': np.random.uniform(0, 100, len(date_range)),
        'diffuse_flows': np.random.uniform(0, 100, len(date_range)),
        'Zone_1_Power_Consumption': np.random.uniform(100, 500, len(date_range)),
        'Zone_2_Power_Consumption': np.random.uniform(100, 500, len(date_range)),
        'Zone_3_Power_Consumption': np.random.uniform(100, 500, len(date_range)),
    }
    return pd.DataFrame(data)

@pytest.fixture
def data_processor(project_config, mocker):
    mocker.patch('power_consumption.preprocessing.data_preprocessor.DataProcessor.load_data')
    processor = DataProcessor(dataset_id=849, config=project_config)
    return processor

@pytest.fixture
def data_processor_with_data(data_processor, sample_data):
    data_processor.data = sample_data
    data_processor._clean_column_names()
    return data_processor

def test_load_data(project_config, mocker):
    mock_fetch = mocker.patch('power_consumption.preprocessing.data_preprocessor.fetch_ucirepo')
    mock_dataset = mocker.MagicMock()
    mock_dataset.data.original = pd.DataFrame({'A': [1, 2, 3]})
    mock_dataset.metadata.name = 'Test Dataset'
    mock_fetch.return_value = mock_dataset

    processor = DataProcessor(dataset_id=849, config=project_config)

    mock_fetch.assert_called_once_with(id=849)
    assert processor.data.equals(pd.DataFrame({'A': [1, 2, 3]}))

def test_load_data_fallback(project_config, mocker):
    mock_fetch = mocker.patch('power_consumption.preprocessing.data_preprocessor.fetch_ucirepo', side_effect=Exception('UCI fetch failed'))
    mock_read_csv = mocker.patch('pandas.read_csv')
    mock_read_csv.return_value = pd.DataFrame({'B': [4, 5, 6]})

    processor = DataProcessor(dataset_id=1, config=project_config)

    mock_fetch.assert_called_once_with(id=1)
    mock_read_csv.assert_called_once_with('../data/Tetuan City power consumption.csv')
    assert processor.data.equals(pd.DataFrame({'B': [4, 5, 6]}))

def test_clean_column_names(data_processor_with_data):
    assert 'Zone_2_Power_Consumption' in data_processor_with_data.data.columns
    assert 'Zone_3_Power_Consumption' in data_processor_with_data.data.columns

def test_create_preprocessor(data_processor):
    preprocessor = data_processor.create_preprocessor()
    assert preprocessor is not None
    assert len(preprocessor.transformers) == 2

def test_preprocess_data(data_processor_with_data):
    data_processor_with_data.preprocess_data()
    assert 'Hour' in data_processor_with_data.data.columns
    assert 'Day' in data_processor_with_data.data.columns
    assert 'Month' in data_processor_with_data.data.columns
    assert 'DayOfWeek' in data_processor_with_data.data.columns
    assert 'IsWeekend' in data_processor_with_data.data.columns
    assert data_processor_with_data.data.index.name == 'DateTime'

def test_split_data(data_processor_with_data):
    data_processor_with_data.preprocess_data()
    X_train, X_test, y_train, y_test = data_processor_with_data.split_data(test_size=0.2)

    assert len(X_train) > len(X_test)
    assert len(y_train) > len(y_test)
    assert X_train.shape[1] == len(data_processor_with_data.config['num_features']) + len(data_processor_with_data.config['cat_features'])
    assert y_train.shape[1] == len(data_processor_with_data.config['target'])

def test_fit_transform_features(data_processor_with_data):
    data_processor_with_data.preprocess_data()
    X_train, X_test, _, _ = data_processor_with_data.split_data(test_size=0.2)
    X_train_transformed, X_test_transformed = data_processor_with_data.fit_transform_features(X_train, X_test)

    assert isinstance(X_train_transformed, np.ndarray)
    assert isinstance(X_test_transformed, np.ndarray)
    assert X_train_transformed.shape[0] == X_train.shape[0]
    assert X_test_transformed.shape[0] == X_test.shape[0]

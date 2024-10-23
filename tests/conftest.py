import pytest
from pathlib import Path
from power_consumption.config import Config

@pytest.fixture
def project_config():
    """
    Fixture to provide the project configuration.

    Returns
    -------
    Config
        An instance of the Config class populated with data from the YAML file.
    """
    config_path = Path(__file__).parent / "conftest.yml"
    return Config.from_yaml(str(config_path))

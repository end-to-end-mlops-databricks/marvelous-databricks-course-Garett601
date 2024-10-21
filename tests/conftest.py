import pytest
import yaml
from pathlib import Path

@pytest.fixture
def project_config():
    config_path = Path(__file__).parent / "conftest.yml"
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

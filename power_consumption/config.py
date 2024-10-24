from typing import List

import yaml
from pydantic import BaseModel


class Hyperparameters(BaseModel):
    learning_rate: float
    n_estimators: int
    max_depth: int


class Features(BaseModel):
    num_features: List[str]
    cat_features: List[str]


class Target(BaseModel):
    target: List[str]


class Dataset(BaseModel):
    id: int


class Config(BaseModel):
    catalog_name: str
    schema_name: str
    hyperparameters: Hyperparameters
    features: Features
    target: Target
    dataset: Dataset

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """
        Load config from yaml file.

        Parameters
        ----------
        yaml_path : str
            Path to the YAML configuration file.

        Returns
        -------
        Config
            An instance of the Config class populated with data from the YAML file.
        """
        with open(yaml_path, "r") as f:
            yaml_dict = yaml.safe_load(f)
        return cls(**yaml_dict)

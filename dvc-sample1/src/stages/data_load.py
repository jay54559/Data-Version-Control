"""
This is the module for data loading
Stage 1 in pipeline
"""

from typing import Text
import pandas as pd
from sklearn.datasets import load_iris
import yaml
import argparse

def data_load(config_path: Text) -> None:
    """ Load Raw Data
    Args:
        config_path {Text}: Path to config file
    """
    print("Loading raw data...")
    with open(config_path) as config_path:
        config = yaml.safe_load(config_path)

    data = load_iris(as_frame=True)
    dataset = data.frame
    dataset.columns = [colname.strip(' (cm)').replace(' ', '_') for colname in dataset.columns.tolist()]
    dataset.to_csv(config['data']['dataset_csv'], index=False)

    print("Data loading done. \n")

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(description="Process the config file")
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_load(config_path=args.config)
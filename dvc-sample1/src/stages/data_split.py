"""
This is the module for data splitting
Stage 3 in pipeline
"""

import pandas as pd
import yaml
from typing import Text
from sklearn.model_selection import train_test_split
import argparse

def data_split(config_path: Text) -> None:
    """ Split the data into train and test sets
    Args:
        config_path {Text}: Path to config file
    """
    print("Splitting the featurized dataset into train and test sets...")
    with open(config_path) as config_path:
        config = yaml.safe_load(config_path)

    featurized_dataset = pd.read_csv(config['data']['features_path'])

    train_dataset, test_dataset = train_test_split(featurized_dataset, test_size=config['data']['test_size'], random_state=config['base']['random_state'])

    train_dataset.to_csv(config['data']['trainset_path'])
    test_dataset.to_csv(config['data']['testset_path'])

    print(f"Featurized dataset split into train and test sets with {(1-config['data']['test_size'])*100}%-{(config['data']['test_size'])*100}% split.\n")

if __name__=="__main__":
    args_parser = argparse.ArgumentParser(description="Process the config file")
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_split(config_path=args.config)
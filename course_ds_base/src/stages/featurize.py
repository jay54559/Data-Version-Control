"""
This is the module for data featurizing
Stage 2 in pipeline
"""

import argparse
import pandas as pd
import yaml
from typing import Text

def featurize(config_path: Text) -> None:
    """ Featurizes the raw data
    Args:
        config_path {Text}: Path to config file
    """
    print("Featurizing raw data...")
    with open(config_path) as config_path:
        config = yaml.safe_load(config_path)
    
    dataset = pd.read_csv(config['data']['dataset_csv'])
    dataset['sepal_length_to_sepal_width'] = dataset['sepal_length'] / dataset['sepal_width']
    dataset['petal_length_to_petal_width'] = dataset['petal_length'] / dataset['petal_width']
    dataset = dataset[[
    'sepal_length', 'sepal_width', 'petal_length', 'petal_width',
    'sepal_length_to_sepal_width', 'petal_length_to_petal_width',
    'target'
    ]]
    dataset.to_csv(config['data']['features_path'], index=False)

    print("Data featurization done.\n")

if __name__=='__main__':
    args_parser = argparse.ArgumentParser(description="Process the config file")
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    featurize(config_path=args.config)


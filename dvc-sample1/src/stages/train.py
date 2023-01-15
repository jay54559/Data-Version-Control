"""
This is the module for training
Stage 4 in pipeline
"""

import pandas as pd
import yaml
from typing import Text
from sklearn.model_selection import train_test_split
import argparse
import joblib
from src.train.train import train

def train_model(config_path: Text) -> None:
    """ Train the model
    Args:
         config_path {Text}: Path to config file
    """
    print("Starting model training...")
    with open(config_path) as config_path:
        config = yaml.safe_load(config_path)
    
    train_dataset = pd.read_csv(config['data']['trainset_path'])
    estimator_name = config['train']['estimator_name']
    model = train(df=train_dataset,
                  target_column=config['featurize']['target_column'],
                  estimator_name=estimator_name,
                  param_grid=config['train']['estimators'][estimator_name]['param_grid'],
                  cv=config['train']['cv']

    )
    
    joblib.dump(model, config['train']['model_path'])

    print("Model training done. Model saved.\n")


if __name__=="__main__":
    args_parser = argparse.ArgumentParser(description="Process the config file")
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train_model(config_path=args.config)
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
from sklearn.linear_model import LogisticRegression


def train(config_path: Text) -> None:
    """ Train the model
    Args:
         config_path {Text}: Path to config file
    """
    print("Starting model training...")
    with open(config_path) as config_path:
        config = yaml.safe_load(config_path)
    
    train_dataset = pd.read_csv(config['data']['trainset_path'])

    X_train = train_dataset.drop('target', axis=1).values.astype('float32')
    y_train = train_dataset.loc[:, 'target'].values.astype('int32')
    
    logreg = LogisticRegression(**config['train']['clf_params'], random_state=config['base']['random_state'])
    logreg.fit(X_train, y_train)

    joblib.dump(logreg, config['train']['model_path'])

    print("Model training done.\n")


if __name__=="__main__":
    args_parser = argparse.ArgumentParser(description="Process the config file")
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train(config_path=args.config)
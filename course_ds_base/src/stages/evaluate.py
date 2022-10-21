"""
This is the module for evaluation
Stage 5 in pipeline
"""

import pandas as pd
import yaml
from typing import Text
from sklearn.model_selection import train_test_split
import argparse
from sklearn.datasets import load_iris
import joblib
import json

from sklearn.metrics import f1_score, confusion_matrix
from src.report.visualize import plot_confusion_matrix

def evaluate_model(config_path: Text) -> None:
    """ Evaluate the model
    Args:
        config_path {Text}: Path to config file
    """
    print("Evaluating the model...")
    with open(config_path) as config_path:
        config = yaml.safe_load(config_path)
    
    model = joblib.load(config['train']['model_path'])

    test_dataset = pd.read_csv(config['data']['testset_path'])
    target_column = config['featurize']['target_column']

    X_test = test_dataset.drop(target_column, axis=1).values.astype('float32')
    y_test = test_dataset.loc[:, target_column].values.astype('int32')
    
    test_prediction = model.predict(X_test)
    cm = confusion_matrix(test_prediction, y_test)
    f1 = f1_score(y_true = y_test, y_pred = test_prediction, average='macro')

    metrics = {
    'f1': f1
    }

    with open(config['reports']['metrics_file'], 'w') as mf:
        json.dump(
            obj=metrics,
            fp=mf,
            indent=4
        )

    data = load_iris(as_frame=True)
    cm_plot = plot_confusion_matrix(cm, data.target_names, normalize=False)
    cm_plot.savefig(config['reports']['confusion_matrix_image'])

    print("Model evaluated. Metric and confusion matrix saved.\n")

if __name__=="__main__":
    args_parser = argparse.ArgumentParser(description="Process the config file")
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate_model(config_path=args.config)
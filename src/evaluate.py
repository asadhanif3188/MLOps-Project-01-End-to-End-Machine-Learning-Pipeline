"""Model evaluation stage: loads trained model and logs accuracy metrics."""
import os
import pickle

import mlflow
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score


def evaluate(data_path, model_path):
    """Load trained model and evaluate accuracy on dataset.

    Args:
        data_path: Path to CSV dataset.
        model_path: Path to pickled model file.
    """
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    predictions = model.predict(X)
    model_accuracy_score = accuracy_score(y, predictions)

    mlflow.log_metric("accuracy", model_accuracy_score)
    print("INFO: Evaluated Model Accuracy:", model_accuracy_score)


if __name__ == "__main__":
    load_dotenv()

    with open('params.yaml') as f:
        params = yaml.safe_load(f)['test']

    evaluate(
        params['data'],
        params['model'],
    )

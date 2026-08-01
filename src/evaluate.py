"""Model evaluation stage: loads trained model and logs accuracy metrics."""
import os
import pickle

import mlflow
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score

from logging_config import configure_logging, get_logger

logger = get_logger("evaluate")


def evaluate(data_path, model_path):
    """Load trained model and evaluate accuracy on dataset.

    Args:
        data_path: Path to CSV dataset.
        model_path: Path to pickled model file.
    """
    logger.info("Evaluate stage started (data=%s, model=%s)", data_path, model_path)

    data = pd.read_csv(data_path)
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    predictions = model.predict(X)
    model_accuracy_score = accuracy_score(y, predictions)

    mlflow.log_metric("accuracy", model_accuracy_score)
    logger.info("Evaluate stage completed; model accuracy: %.4f", model_accuracy_score)


if __name__ == "__main__":
    load_dotenv()
    configure_logging()

    with open('params.yaml') as f:
        params = yaml.safe_load(f)['test']

    evaluate(
        params['data'],
        params['model'],
    )

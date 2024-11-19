import pandas as pd
import yaml
import os
import pickle
import mlflow
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
from urllib.parse import urlparse

# -----------------------------------------------------
def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    # Load the model from the models folder
    model = pickle.load(open(model_path, 'rb'))

    predictions = model.predict(X)
    model_accuracy_score = accuracy_score(y, predictions)

    mlflow.log_metric("accuracy", model_accuracy_score)
    print("INFO: Evaualted Model Accuracy:", model_accuracy_score)
    
# -----------------------------------------------------

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    # Load parameters from param.yaml
    params = yaml.safe_load(open('params.yaml'))['test']

    evaluate(
        params['data'],
        params['model'],
    )

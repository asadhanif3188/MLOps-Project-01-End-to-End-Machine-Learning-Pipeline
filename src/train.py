import pandas as pd
import yaml
import pickle
import os
import mlflow
from dotenv import load_dotenv

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

from mlflow.models import infer_signature

from urllib.parse import urlparse
# ----------------------------------------------------------------
def hyperparameter_tuning(X_train, y_train, param_grid):
    # Define the model and the hyperparameters to tune
    rf_model = RandomForestClassifier()

    # Perform grid search
    grid_search = GridSearchCV(estimator=rf_model, 
                               param_grid=param_grid, 
                               cv=3,
                               n_jobs=-1,
                               verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

# ----------------------------------------------------------------
def train(data_path, 
          model_path, 
          random_state, 
          n_estimators, 
          max_depth):
    # print("**** Received following items ****")
    # print("-> data_path:", data_path)
    # print("-> model_path:", model_path)
    # print("-> random_state:", random_state)
    # print("-> n_estimators:", n_estimators)
    # print("-> max_depth:", max_depth)
    # print("-> MLFLOW_TRACKING_URI:", os.environ['MLFLOW_TRACKING_URI'])
    # print("-> MLFLOW_TRACKING_USERNAME:", os.environ['MLFLOW_TRACKING_USERNAME'])
    # print("-> MLFLOW_TRACKING_PASSWORD:", os.environ['MLFLOW_TRACKING_PASSWORD'])
    # print("**********************************")

    data = pd.read_csv(data_path)
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    # start the mlflow run 
    with mlflow.start_run():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        signature = infer_signature(X_train, y_train)

        # Define hyperparameters grid
        param_grid = {
            'n_estimators':         [100, 200],
            'max_depth':            [5, 10, None],
            'min_samples_split':    [2, 5],
            'min_samples_leaf':     [1, 2],
        }

        # Perform hyperparameter tuning
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)

        # Get the best model
        best_model = grid_search.best_estimator_

        # Predict and evaluate the best model
        y_pred = best_model.predict(X_test)
        model_accuracy_score = accuracy_score(y_test, y_pred)
        print("Best Model Accuracy:", model_accuracy_score)


# ----------------------------------------------------------------
if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    # Load parameters from param.yaml
    params = yaml.safe_load(open('params.yaml'))['train']

    train(
        params['input'],
        params['output'],
        params['random_state'],
        params['n_estimators'],
        params['max_depth'],
    )
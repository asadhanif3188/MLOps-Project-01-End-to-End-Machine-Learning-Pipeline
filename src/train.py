"""Model training stage: hyperparameter tuning, training, and MLflow tracking."""
import os
import pickle
from urllib.parse import urlparse

import mlflow
import pandas as pd
import yaml
from dotenv import load_dotenv
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

from logging_config import configure_logging, get_logger

logger = get_logger("train")


def hyperparameter_tuning(X_train, y_train, param_grid):
    """Perform grid search over hyperparameter space using 3-fold cross-validation.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        param_grid: Dictionary of hyperparameters and their candidate values.

    Returns:
        GridSearchCV object fitted with best parameters.
    """
    rf_model = RandomForestClassifier()

    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    logger.info("Hyperparameter tuning started")
    grid_search.fit(X_train, y_train)
    logger.info("Hyperparameter tuning completed; best params: %s", grid_search.best_params_)

    return grid_search


def train(data_path, model_path, random_state, n_estimators, max_depth):
    """Train a Random Forest model, tune hyperparameters, log to MLflow.

    Args:
        data_path: Path to raw CSV dataset.
        model_path: Path to save pickled model.
        random_state: Random seed for reproducibility.
        n_estimators: Baseline number of estimators (used in grid search).
        max_depth: Baseline max depth (used in grid search).
    """
    logger.info("Train stage started (data=%s, model=%s)", data_path, model_path)

    data = pd.read_csv(data_path)
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        signature = infer_signature(X_train, y_train)

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
        }

        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        model_accuracy_score = accuracy_score(y_test, y_pred)
        logger.info("Best model accuracy: %.4f", model_accuracy_score)

        mlflow.log_metric("accuracy", model_accuracy_score)
        mlflow.log_param("best_n_estimators", grid_search.best_params_['n_estimators'])
        mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
        mlflow.log_param("best_samples_split", grid_search.best_params_['min_samples_split'])
        mlflow.log_param("best_samples_leaf", grid_search.best_params_['min_samples_leaf'])

        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(str(cr), "classification_report.txt")

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                best_model,
                "model",
                registered_model_name="Best Random Forest Classifier",
                signature=signature,
            )
        else:
            mlflow.sklearn.log_model(best_model, "model", signature=signature)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)

        logger.info("Model saved to %s", model_path)
        logger.info("Train stage completed")


if __name__ == "__main__":
    configure_logging()
    load_dotenv()

    with open('params.yaml') as f:
        params = yaml.safe_load(f)['train']

    train(
        params['input'],
        params['output'],
        params['random_state'],
        params['n_estimators'],
        params['max_depth'],
    )
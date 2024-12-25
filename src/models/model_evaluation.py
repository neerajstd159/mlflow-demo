import numpy as np
import pandas as pd
import logging
import pickle
import json
import os
import mlflow
import dagshub
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "neerajstd159"
repo_name = "mlflow-demo"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# setup logger
logger = logging.Logger('model evaluation')
logger.setLevel(logging.DEBUG)

# console handler
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG)

# file handler
fileHandler = logging.FileHandler('errors.log')
fileHandler.setLevel(logging.ERROR)

# formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consoleHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

# add logger
logger.addHandler(consoleHandler)
logger.addHandler(fileHandler)


def load_model(model_path: str):
    """Load the trained model from given path"""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug(f"Model loaded from {model_path}")
        return model
    except FileNotFoundError:
        logger.error(f"File not found {model_path}")
        raise
    except Exception as e:
        logger.error(f"unexpected error occured while loading the model: {e}")
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from given csv file"""
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Data loaded from {file_path}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse csv file: {e}")
        raise
    except Exception as e:
        logger.error(f"unexpected error occured while loading data: {e}")
        raise


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation matrix"""
    try:
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:,1]

        accuracy = accuracy_score(y_pred, y_test)
        precision = precision_score(y_pred, y_test)
        recall = recall_score(y_pred, y_test)
        f1 = f1_score(y_pred, y_test)
        roc = roc_auc_score(y_test, y_pred_prob)

        metric = {
            "accuracy" : accuracy,
            "precision" : precision,
            "recall" : recall,
            "f1_score" : f1,
            "roc_score" : roc
        }
        logger.debug("Model evaluation matrix calculated")
        return metric
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise


def save_matrics(metrics: dict, file_path: str) -> None:
    """save the evaluation metrics in json file"""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug(f"metrics saved to {file_path}")
    except Exception as e:
        logger.error(f"Error occured while saving metrics to {file_path}")
        raise


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """save the model/experiment run_id and path to its json file"""
    try:
        model_info = {
            "run_id" : run_id,
            "model_path" : model_path
        }
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug(f"model info saved to {file_path}")
    except Exception as e:
        logger.error(f"Error occured while saving metrics to {file_path}")
        raise


def main():
    mlflow.set_experiment("dvc-pipeline")
    with mlflow.start_run() as run:
        try:
            # load model
            clf = load_model('./models/model.pkl')

            # load data
            test_data = load_data('./data/processed/test_bow.csv')
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            # evaluate metrics
            metrics = evaluate_model(clf, X_test, y_test)

            # save metric
            save_matrics(metrics, 'reports/metrics.json')

            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, value in params.items():
                    mlflow.log_param(param_name, value)

            # log model
            mlflow.sklearn.log_model(clf, "model")

            # save model info
            save_model_info(run.info.run_id, "models", 'reports/experiment_info.json')

            # Log the metrics file to MLflow
            mlflow.log_artifact('reports/metrics.json')

            # Log the model info file to MLflow
            mlflow.log_artifact('reports/experiment_info.json')

            # Log the evaluation errors log file to MLflow
            mlflow.log_artifact('model_evaluation_errors.log')
        except Exception as e:
            logger.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")


if __name__ == '__main__':
    main()
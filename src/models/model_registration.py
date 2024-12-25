import mlflow
import dagshub
import json
import logging
import os

# setup logging
logger = logging.Logger('model_registry')
logger.setLevel(logging.DEBUG)

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler('errors.log')
fileHandler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fileHandler.setFormatter(formatter)
consoleHandler.setFormatter(formatter)

logger.addHandler(fileHandler)
logger.addHandler(consoleHandler)


# setup dagshub
dagshub_token = os.getenv('DAGSHUB_PAT')
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_token
os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "neerajstd159"
repo_name = "mlflow-demo"

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

# load model info from json file
def load_model_info(path: str) -> dict:
    """load the model info from given json file"""
    try:
        with open(path, 'r') as file:
            info = json.load(file)
        logger.debug(f"model info loaded from {path}")
        return info
    except FileNotFoundError:
        logger.error(f"model info file not found: {path}")
        raise
    except Exception as e:
        logger.error(f"unexpected error occured: {e}")
        raise


def register_model(model_name: str, model_info: dict):
    """register the model to the mlflow model registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logger.debug(f"Attempting to register model with URI: {model_uri}")
        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logging.debug(f"Model {model_name} version {model_version.version} registered and transitioned to staging")
    except Exception as e:
        logger.error(f"error during model registration: {e}")
        raise


def main():
    try:
        model_info_path = "reports/experiment_info.json"
        model_info = load_model_info(model_info_path)

        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")



if __name__ == '__main__':
    main()
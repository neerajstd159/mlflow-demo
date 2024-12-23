import numpy as np
import pandas as pd
import logging
import yaml
import os
from sklearn.model_selection import train_test_split

# logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug("parameters retrieved from %s", params_path)
    except FileNotFoundError:
        logger.error("file not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("yaml error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from given csv file"""
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data loaded from %s", data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse csv file: %s", e)
        raise
    except FileNotFoundError:
        logger.error("file not found: %s", data_url)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data"""
    try:
        df = df.drop(columns=['tweet_id'])
        df = df[df['sentiment'].isin(['happiness', 'sadness', 'love'])]
        df['sentiment'] = df['sentiment'].map({'happiness':0, 'love':0, 'sadness':1})
        logger.debug("Data preprocess completed")
        return df
    except KeyError as e:
        logger.error(f"Missing column from dataset: {e}")
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, data_path: str) -> None:
    """Save the train and test dataset"""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_df.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)
        logger.debug(f"Train and Test dataset saved to {raw_data_path}")
    except Exception as e:
        logger.error(f"Unexpected error occured while saving the data: {e}")
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        df = load_data(data_url='https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=1)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error(f"Failed to complete the data ingestion process: {e}")
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
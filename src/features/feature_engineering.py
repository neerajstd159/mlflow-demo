import numpy as np
import pandas as pd
import os
import yaml
import logging
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# logging configuration
logger = logging.getLogger('feature engineering')
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
    """Load parameters from given yaml file"""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f"parameters retrived from {params_path}")
        return params
    except FileNotFoundError:
        logger.error(f"File not found: {params_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"yaml error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

def load_data(data_path: str) -> pd.DataFrame:
    """Load data from a csv file"""
    try:
        df = pd.read_csv(data_path)
        df.fillna('', inplace=True)
        logger.debug(f"data loaded from {data_path}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"failed to parse the csv file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


def apply_bow(train_df: pd.DataFrame, test_df: pd.DataFrame, max_features: int) -> tuple:
    """Apply bow vectorizer to data"""
    try:
        vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1,2))
        X_train = train_df['content'].values
        y_train = train_df['sentiment'].values
        X_test = test_df['content'].values
        y_test = test_df['sentiment'].values

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train
        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        with open('models/vectorizer.pkl', 'wb') as file:
            pickle.dump(vectorizer, file)
        
        logger.debug("bow applied and data transformed")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, data_path: str) -> None:
    """Save the train and test dataset"""
    try:
        processed_data_path = os.path.join(data_path, 'processed')
        os.makedirs(processed_data_path, exist_ok=True)
        train_df.to_csv(os.path.join(processed_data_path, 'train_bow.csv'), index=False)
        test_df.to_csv(os.path.join(processed_data_path, 'test_bow.csv'), index=False)
        logger.debug(f"Train and Test dataset saved to {processed_data_path}")
    except Exception as e:
        logger.error(f"Unexpected error occured while saving the data: {e}")
        raise


def main():
    try:
        params = load_params(params_path='params.yaml')
        max_features = params['feature_engineering']['max_features']

        # load data
        train_df = load_data(data_path='./data/interim/train_processed.csv')
        test_df = load_data(data_path='./data/interim/test_processed.csv')

        # apply bow
        train_df, test_df = apply_bow(train_df, test_df, max_features)

        # save data
        save_data(train_df, test_df, data_path='./data')
    except Exception as e:
        logger.error(f"failed to complete feature engineering process: {e}")
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
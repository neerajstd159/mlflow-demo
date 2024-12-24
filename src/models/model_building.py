import numpy as np
import pandas as pd
import logging
import yaml
import pickle
from sklearn.ensemble import RandomForestClassifier

# logging configuration
logger = logging.getLogger('model building')
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


def load_data(data_path: str) -> pd.DataFrame:
    """load data from a csv file"""
    try:
        df = pd.read_csv(data_path)
        logger.debug(f"dataset loaded from {data_path}")
        return df
    except pd.errors.ParserError:
        logger.error(f"failed to parse the csv file")
        raise
    except Exception as e:
        logger.error(f"unexpected error: {e}")
        raise


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """train the random  forest model"""
    try:
        lr = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, min_samples_split=5)
        lr.fit(X_train, y_train)
        logger.debug("model training complete")
        return lr
    except Exception as e:
        logger.error(f"error while model trainig: {e}")
        raise


def save_model(model, save_path: str) -> None:
    """save the trained model to a pkl file"""
    try:
        with open(save_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug(f"model saved to {save_path}")
    except Exception as e:
        logger.error(f"error while model saving: {e}")
        raise


def main():
    try:
        # load data
        train_data = load_data('./data/processed/train_bow.csv')
        X_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values

        # train model
        lr = train_model(X_train, y_train)

        # save model
        save_model(lr, 'models/model.pkl')
    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
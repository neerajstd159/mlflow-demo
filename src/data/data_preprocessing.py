import numpy as np
import pandas as pd
import logging
import os
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# logging configuration
logger = logging.getLogger('data_preprocessing')
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

def lower_case(text):
    """Change text to lower case"""
    text = [word.lower() for word in text.split()]
    return " ".join(text)


def replace_pattern(pattern, text):
    """match and remove words/letters that matches pattern"""
    text = re.sub(pattern, ' ', text)
    return text


def replace_pattern_with_pattern(pattern1, pattern2, text):
    """replace pattern1 with pattern2"""
    text = re.sub(pattern1, pattern2, text)
    return text

def remove_stopwords(stopword, text):
    """remove stopwords from given text"""
    new_text = [word for word in text.split() if word not in stopword]
    return " ".join(new_text)

def apply_lemmatization(lemmatizer, text):
    """apply lemmatization to given text"""
    new_text = [lemmatizer.lemmatize(word) for word in text.split()]
    return " ".join(new_text)


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """normalize the text data"""
    try:
        # lower case
        df['content'] = df['content'].apply(lambda x: lower_case(x))
        logger.debug("changed to lower case")
        # remove urls
        url_pattern = r"https?://[^\s]*|www\.[^\s]*"
        df['content'] = df['content'].apply(lambda x: replace_pattern(url_pattern, x))
        logger.debug("removed urls")
        # remove @username
        user_pattern = r"@[^\s]*"
        df['content'] = df['content'].apply(lambda x: replace_pattern(user_pattern, x))
        logger.debug("removed username")
        # remove non-alphanumeric letters
        pattern = r"[^a-zA-Z0-9]"
        df['content'] = df['content'].apply(lambda x: replace_pattern(pattern, x))
        logger.debug("removed non alpha-numeric letters")
        # remove letters that repeated more than 2 times
        search_pattern = r'(.)\1{2,}'
        replace_with = r'\1\1'
        df['content'] = df['content'].apply(lambda x: replace_pattern_with_pattern(search_pattern, replace_with, x))
        logger.debug("repeated charectors removed")
        # remove stopwords
        stopword = set(stopwords.words('english'))
        df['content'] = df['content'].apply(lambda x: remove_stopwords(stopword, x))
        logger.debug("removed stopwords")
        # apply lemmatization
        lemmatizer = WordNetLemmatizer()
        df['content'] = df['content'].apply(lambda x: apply_lemmatization(lemmatizer, x))
        logger.debug("applied lemmatization")
        return df
    except Exception as e:
        logger.error(f"Some error occured: {e}")
        raise

def main():
    try:
        # fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug("data loaded successfully")

        # transform data
        train_processed_df = normalize_text(train_data)
        test_processed_df = normalize_text(test_data)
        logger.debug("data transformed successfully")

        # store the data into data/interim
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_df.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_df.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        logger.debug(f"processed data saved to {data_path}")
    except Exception as e:
        logger.error(f"Failed to complete the data transformation: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
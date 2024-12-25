import numpy as np
import pandas as pd
import logging
import os
import re
import nltk
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


def normalize_text(text: str) -> str:
    """normalize the text data"""
    try:
        # lower case
        text = lower_case(text)
        logger.debug("changed to lower case")
        # remove urls
        url_pattern = r"https?://[^\s]*|www\.[^\s]*"
        text = replace_pattern(url_pattern, text)
        logger.debug("removed urls")
        # remove @username
        user_pattern = r"@[^\s]*"
        text = replace_pattern(user_pattern, text)
        logger.debug("removed username")
        # remove non-alphanumeric letters
        pattern = r"[^a-zA-Z0-9]"
        text = replace_pattern(pattern, text)
        logger.debug("removed non alpha-numeric letters")
        # remove letters that repeated more than 2 times
        search_pattern = r'(.)\1{2,}'
        replace_with = r'\1\1'
        text = replace_pattern_with_pattern(search_pattern, replace_with, text)
        logger.debug("repeated charectors removed")
        # remove stopwords
        stopword = set(stopwords.words('english'))
        text = remove_stopwords(stopword, text)
        logger.debug("removed stopwords")
        # apply lemmatization
        lemmatizer = WordNetLemmatizer()
        text = apply_lemmatization(lemmatizer, text)
        logger.debug("applied lemmatization")
        return text
    except Exception as e:
        logger.error(f"Some error occured: {e}")
        raise
# app/utils/data_preprocessing.py

import os
import pandas as pd
import logging
from typing import List


def load_passwords(data_source: str) -> List[str]:
    """
    Load passwords from the specified data source.

    :param data_source: Path to the data file (e.g., .txt, .csv)
    :return: List of password strings
    """
    if not os.path.exists(data_source):
        logging.error(f"Data source {data_source} does not exist.")
        return []

    try:
        if data_source.endswith('.txt'):
            with open(data_source, 'r', encoding='utf-8') as f:
                passwords = [line.strip() for line in f if line.strip()]
        elif data_source.endswith('.csv'):
            df = pd.read_csv(data_source)
            if 'password' in df.columns:
                passwords = df['password'].dropna().astype(str).tolist()
            else:
                logging.error("CSV file does not contain 'password' column.")
                passwords = []
        else:
            logging.error("Unsupported file format. Only .txt and .csv are supported.")
            passwords = []
        return passwords
    except Exception as e:
        logging.error(f"Error loading passwords from {data_source}: {e}")
        return []


def preprocess_passwords(passwords: List[str]) -> List[str]:
    """
    Preprocess the list of passwords.

    :param passwords: List of raw password strings
    :return: List of cleaned password strings
    """
    # Example preprocessing: remove duplicates, convert to lower case, etc.
    cleaned = list(set(password.strip() for password in passwords if password.strip()))
    return cleaned

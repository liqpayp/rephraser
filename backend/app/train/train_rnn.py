# backend/app/train/train_rnn.py

import json
import logging
import os
from ..models.rnn_model import RNNModel

logger = logging.getLogger(__name__)


def train_rnn_model(corpus_path: str, incremental: bool = False, model_path: str = "app/models/rnn_model.h5"):
    """
    Train the RNN model on the provided corpus.

    - **corpus_path**: Path to the password corpus file.
    - **incremental**: Whether to perform incremental training.
    - **model_path**: Path to save/load the trained model.
    """
    try:
        model = RNNModel()
        if incremental and os.path.exists(model_path):
            model.load(model_path)
            logger.info("Loaded existing RNN model for incremental training.")
        else:
            logger.info("Initialized new RNN model for training.")

        with open(corpus_path, 'r', encoding='utf-8') as f:
            passwords = f.read().splitlines()

        model.train(passwords)

        model.save_model(model_path)

        logger.info(f"RNN model trained and saved to {model_path}")
    except Exception as e:
        logger.error(f"Error training RNN model: {e}")
        raise e

import pickle
import logging
import os
from ..models.markov_model import ImprovedMarkovModel

logger = logging.getLogger(__name__)


def train_markov_model(corpus_path: str, incremental: bool = False, model_path: str = "app/models/markov_model.pkl"):
    """
    Train the Markov model on the provided corpus.

    - **corpus_path**: Path to the password corpus file.
    - **incremental**: Whether to perform incremental training.
    - **model_path**: Path to save/load the trained model.
    """
    try:
        if incremental and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info("Loaded existing Markov model for incremental training.")
        else:
            model = ImprovedMarkovModel()
            logger.info("Initialized new Markov model for training.")

        with open(corpus_path, 'r', encoding='utf-8') as f:
            passwords = f.read().splitlines()

        model.train(passwords)

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        logger.info(f"Markov model trained and saved to {model_path}")
    except Exception as e:
        logger.error(f"Error training Markov model: {e}")
        raise e

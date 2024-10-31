# backend/app/train/train_gan.py

import pickle
import logging
import os
from ..models.gan_model import GANModel

logger = logging.getLogger(__name__)


def train_gan_model(corpus_path: str, incremental: bool = False, model_path: str = "app/models/gan_model.pkl"):
    """
    Train the GAN model on the provided corpus.

    - **corpus_path**: Path to the password corpus file.
    - **incremental**: Whether to perform incremental training.
    - **model_path**: Path to save/load the trained model.
    """
    try:
        if incremental and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info("Loaded existing GAN model for incremental training.")
        else:
            model = GANModel()
            logger.info("Initialized new GAN model for training.")

        with open(corpus_path, 'r', encoding='utf-8') as f:
            passwords = f.read().splitlines()

        model.train(passwords)

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        logger.info(f"GAN model trained and saved to {model_path}")
    except Exception as e:
        logger.error(f"Error training GAN model: {e}")
        raise e

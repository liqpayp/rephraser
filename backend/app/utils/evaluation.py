# app/utils/evaluation.py

from typing import Dict
from ..models import AdvancedHybridModel
import logging


def evaluate_models(hybrid_model: AdvancedHybridModel) -> Dict[str, float]:
    """
    Evaluate the performance of all models.

    :param hybrid_model: Instance of HybridModel containing all models
    :return: Dictionary of model scores
    """
    scores = {}
    try:
        # Example evaluations

        # Markov Model: Diversity (number of unique passwords generated)
        markov_passwords = [hybrid_model.markov.generate_password() for _ in range(100)]
        unique_markov = len(set(markov_passwords))
        scores['markov'] = unique_markov / 100.0

        # RNN Model: Diversity
        rnn_passwords = [hybrid_model.rnn.generate_password() for _ in range(100)]
        unique_rnn = len(set(rnn_passwords))
        scores['rnn'] = unique_rnn / 100.0

        # GAN Model: Diversity
        gan_passwords = [hybrid_model.gan.generate_password() for _ in range(100)]
        unique_gan = len(set(gan_passwords))
        scores['gan'] = unique_gan / 100.0

        # Additional metrics can be added here (e.g., entropy, average length)
    except Exception as e:
        logging.error(f"Error evaluating models: {e}")
    return scores

# backend/tests/test_hybrid_model.py

import pytest
from ..models import HybridModel
from typing import List


@pytest.fixture
def sample_passwords() -> List[str]:
    return [
        "password123",
        "admin2023",
        "letmein!",
        "qwerty",
        "123456",
        "welcome",
        "monkey",
        "dragon",
        "baseball",
        "football"
    ]


@pytest.fixture
def hybrid_model() -> HybridModel:
    # Initialize with test model paths to avoid conflicts
    return HybridModel()


def test_hybrid_model_training(hybrid_model: HybridModel, sample_passwords: List[str]):
    hybrid_model.train_all(
        passwords=sample_passwords,
        epochs_rnn=1,  # Minimal epochs for testing
        epochs_gan=10,  # Minimal epochs for testing
        batch_size_rnn=2,
        batch_size_gan=2,
        save_interval_gan=5
    )
    assert hybrid_model.markov.model is not None and len(
        hybrid_model.markov.model) > 0, "Markov model should be trained."
    assert hybrid_model.rnn.model is not None, "RNN model should be trained."
    assert hybrid_model.gan.generator is not None and hybrid_model.gan.discriminator is not None, "GAN models should be trained."


def test_hybrid_model_generation(hybrid_model: HybridModel, sample_passwords: List[str]):
    hybrid_model.train_all(
        passwords=sample_passwords,
        epochs_rnn=1,
        epochs_gan=10,
        batch_size_rnn=2,
        batch_size_gan=2,
        save_interval_gan=5
    )
    generated_password = hybrid_model.generate_password(max_length=12)
    assert isinstance(generated_password, str), "Generated password should be a string."
    assert 6 <= len(generated_password) <= 12, "Generated password should respect the max_length constraint."
    print(f"Generated Password: {generated_password}")


def test_hybrid_model_persistence(hybrid_model: HybridModel, sample_passwords: List[str]):
    hybrid_model.train_all(
        passwords=sample_passwords,
        epochs_rnn=1,
        epochs_gan=10,
        batch_size_rnn=2,
        batch_size_gan=2,
        save_interval_gan=5
    )
    hybrid_model.save_all_models()

    # Create a new instance and load all models
    new_hybrid_model = HybridModel()
    new_hybrid_model.load_all_models()

    assert new_hybrid_model.markov.model == hybrid_model.markov.model, "Markov models should match after loading."
    assert new_hybrid_model.rnn.model is not None, "RNN model should be loaded correctly."
    assert new_hybrid_model.gan.generator is not None and new_hybrid_model.gan.discriminator is not None, "GAN models should be loaded correctly."

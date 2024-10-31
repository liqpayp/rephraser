# backend/tests/test_markov_model.py

import pytest
from ..models import MarkovModel
from typing import List


@pytest.fixture
def sample_passwords() -> List[str]:
    return [
        "password123",
        "admin2023",
        "letmein!",
        "qwerty",
        "123456",
        "password123",  # Duplicate to test preprocessing
        "welcome",
        "monkey",
        "dragon",
        "baseball"
    ]


@pytest.fixture
def markov_model() -> MarkovModel:
    # Use a temporary path for testing to avoid conflicts
    return MarkovModel(order=2, model_path="tests/models/test_markov_model.pkl")


def test_markov_model_training(markov_model: MarkovModel, sample_passwords: List[str]):
    markov_model.train(sample_passwords)
    assert len(markov_model.model) > 0, "Markov model should be trained with non-empty transition probabilities."


def test_markov_model_generation(markov_model: MarkovModel, sample_passwords: List[str]):
    markov_model.train(sample_passwords)
    generated_password = markov_model.generate_password(max_length=12)
    assert isinstance(generated_password, str), "Generated password should be a string."
    assert 6 <= len(generated_password) <= 12, "Generated password should respect the max_length constraint."


def test_markov_model_persistence(markov_model: MarkovModel, sample_passwords: List[str]):
    markov_model.train(sample_passwords)
    markov_model.save_model()

    # Create a new instance and load the model
    new_markov_model = MarkovModel(order=2, model_path="tests/models/test_markov_model.pkl")
    new_markov_model.load_model()

    assert new_markov_model.model == markov_model.model, "Loaded model should match the saved model."

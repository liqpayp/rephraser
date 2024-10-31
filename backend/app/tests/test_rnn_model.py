# backend/tests/test_rnn_model.py

import pytest
from ..models import RNNModel
from typing import List
import os


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
def rnn_model() -> RNNModel:
    # Use temporary paths for testing to avoid conflicts
    return RNNModel(
        sequence_length=3,
        model_path="tests/models/test_rnn_model.h5",
        tokenizer_path="tests/models/test_rnn_tokenizer.pkl"
    )


def test_rnn_model_training(rnn_model: RNNModel, sample_passwords: List[str]):
    rnn_model.train(sample_passwords, epochs=1, batch_size=2)  # Use minimal epochs for testing
    assert rnn_model.model is not None, "RNN model should be trained and not None."


def test_rnn_model_generation(rnn_model: RNNModel, sample_passwords: List[str]):
    rnn_model.train(sample_passwords, epochs=1, batch_size=2)
    generated_password = rnn_model.generate_password(max_length=12)
    assert isinstance(generated_password, str), "Generated password should be a string."
    assert 6 <= len(generated_password) <= 12, "Generated password should respect the max_length constraint."


def test_rnn_model_persistence(rnn_model: RNNModel, sample_passwords: List[str]):
    rnn_model.train(sample_passwords, epochs=1, batch_size=2)
    rnn_model.save_model()
    rnn_model.save_tokenizer()

    # Create a new instance and load the model
    new_rnn_model = RNNModel(
        sequence_length=3,
        model_path="tests/models/test_rnn_model.h5",
        tokenizer_path="tests/models/test_rnn_tokenizer.pkl"
    )
    new_rnn_model.load_model()

    assert new_rnn_model.model is not None, "Loaded RNN model should not be None."
    assert new_rnn_model.char_to_int == rnn_model.char_to_int, "Character mappings should match."
    assert new_rnn_model.int_to_char == rnn_model.int_to_char, "Inverse character mappings should match."
    assert new_rnn_model.vocab_size == rnn_model.vocab_size, "Vocabulary sizes should match."

# backend/tests/test_gan_model.py

import pytest
import os
import shutil
from ..models.gan_model import GANModel


@pytest.fixture
def sample_passwords():
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
        "football",
    ]


@pytest.fixture
def gan_model():
    # Define the directory for test models
    model_dir = os.path.join("tests", "models")

    # Remove existing model directory to ensure a fresh start
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    # Initialize GANModel without loading existing models
    return GANModel(
        sequence_length=5,  # Increased sequence_length for better context
        model_path=os.path.join(model_dir, "test_gan_model.h5"),
        tokenizer_path=os.path.join(model_dir, "test_gan_tokenizer.pkl"),
        load_existing=False,  # Do not load existing models
    )


def test_gan_model_building(gan_model: GANModel):
    generator = gan_model.build_generator()
    discriminator = gan_model.build_discriminator()
    assert generator is not None, "Generator model should be built successfully."
    assert (
            discriminator is not None
    ), "Discriminator model should be built successfully."


def test_gan_model_training(gan_model: GANModel, sample_passwords):
    # Train the GANModel with reduced epochs for testing purposes
    gan_model.train(sample_passwords, epochs=1000, batch_size=2, save_interval=500)
    assert (
            gan_model.generator is not None
    ), "GAN generator should be trained and not None."
    assert (
            gan_model.discriminator is not None
    ), "GAN discriminator should be trained and not None."
    assert (
            gan_model.gan is not None
    ), "GAN composite model should be trained and not None."


def test_gan_model_generation(gan_model: GANModel, sample_passwords):
    # Train the GANModel before generating passwords
    gan_model.train(sample_passwords, epochs=1000, batch_size=2, save_interval=500)

    # Generate a password
    generated_password = gan_model.generate_password(max_length=12)

    # Assertions to verify generated password
    assert isinstance(generated_password, str), "Generated password should be a string."
    assert (
            6 <= len(generated_password) <= 12
    ), "Generated password should respect the max_length constraint."
    print(f"Generated password: {generated_password}")


def test_gan_model_persistence(gan_model: GANModel, sample_passwords):
    # Train the GANModel
    gan_model.train(sample_passwords, epochs=1000, batch_size=2, save_interval=500)

    # Save models and tokenizer
    gan_model.save_models()
    gan_model.save_tokenizer()

    # Initialize a new GANModel instance with load_existing=True
    new_gan_model = GANModel(
        sequence_length=5,
        model_path=os.path.join("tests", "models", "test_gan_model.h5"),
        tokenizer_path=os.path.join("tests", "models", "test_gan_tokenizer.pkl"),
        load_existing=True,  # Load existing models and tokenizer
    )
    new_gan_model.load_models()

    # Assertions to verify that models and mappings are loaded correctly
    assert (
            new_gan_model.generator is not None
    ), "Loaded GAN generator should not be None."
    assert (
            new_gan_model.discriminator is not None
    ), "Loaded GAN discriminator should not be None."
    assert (
            new_gan_model.gan is not None
    ), "Loaded GAN composite model should not be None."
    assert (
            new_gan_model.char_to_int == gan_model.char_to_int
    ), "Character mappings should match."
    assert (
            new_gan_model.int_to_char == gan_model.int_to_char
    ), "Inverse character mappings should match."
    assert (
            new_gan_model.vocab_size == gan_model.vocab_size
    ), "Vocabulary sizes should match."

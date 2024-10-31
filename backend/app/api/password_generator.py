# app/api/password_generator.py

from fastapi import APIRouter, Query, HTTPException
from typing import List
import logging
from ..models.markov_model import ImprovedMarkovModel
from ..models.rnn_model import AdvancedRNNModel
from ..models.gan_model import AdvancedGANModel
import pickle
import os

password_generator_router = APIRouter()
logger = logging.getLogger(__name__)


def load_model(model_type: str):
    """Загрузка модели нужного типа"""
    try:
        if model_type == "markov":
            model_path = "app/models/markov_model.pkl"
            if not os.path.exists(model_path):
                logger.error(f"No trained Markov model found at {model_path}")
                raise HTTPException(
                    status_code=500,
                    detail="Markov model not trained or not found."
                )
            model = ImprovedMarkovModel(config={'model_path': model_path})
            logger.info(f"Markov model loaded from {model_path}")
            return model

        elif model_type == "rnn":
            model_path = "app/models/rnn_model.h5"
            if not os.path.exists(model_path):
                logger.error(f"No trained RNN model found at {model_path}")
                raise HTTPException(
                    status_code=500,
                    detail="RNN model not trained or not found."
                )
            model = AdvancedRNNModel(config={'model_path': model_path})
            logger.info(f"RNN model loaded from {model_path}")
            return model

        elif model_type == "gan":
            model_path = "app/models/gan_model.h5"
            if not os.path.exists(model_path):
                logger.error(f"No trained GAN model found at {model_path}")
                raise HTTPException(
                    status_code=500,
                    detail="GAN model not trained or not found."
                )
            model = AdvancedGANModel(config={'model_path': model_path})
            logger.info(f"GAN model loaded from {model_path}")
            return model

        else:
            logger.error(f"Invalid model type: {model_type}")
            raise HTTPException(status_code=400, detail="Invalid model selected.")

    except Exception as e:
        logger.error(f"Error loading {model_type} model: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error loading {model_type} model: {str(e)}"
        )


@password_generator_router.post("/generate-password")
async def generate_password_endpoint(
        model: str = Query(..., description="Model to use: markov, rnn, gan"),
        num_passwords: int = Query(1, ge=1),
        stream: bool = Query(False)
):
    """Генерация паролей"""
    try:
        selected_model = load_model(model)

        passwords = []
        for _ in range(num_passwords):
            password = selected_model.generate_password()
            passwords.append(password)

            if stream:
                # Здесь можно добавить streaming логику если нужно
                pass

        return {"passwords": passwords}

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Password generation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate password(s): {str(e)}"
        )

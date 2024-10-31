# app/api/models_api.py

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional

from ..models import AdvancedHybridModel
from ..utils.data_preprocessing import load_passwords
from ..utils.evaluation import evaluate_models
import logging

models_api_router = APIRouter()

hybrid_model = AdvancedHybridModel()


class TrainModelsRequest(BaseModel):
    data_source: str = Field(..., description="Path to the password data file (e.g., .txt or .csv)")
    epochs_rnn: Optional[int] = Field(
        default=50,
        ge=1,
        description="Number of epochs for RNN training"
    )
    epochs_gan: Optional[int] = Field(
        default=10000,
        ge=1,
        description="Number of epochs for GAN training"
    )
    batch_size_rnn: Optional[int] = Field(
        default=128,
        ge=1,
        description="Batch size for RNN training"
    )
    batch_size_gan: Optional[int] = Field(
        default=64,
        ge=1,
        description="Batch size for GAN training"
    )
    save_interval_gan: Optional[int] = Field(
        default=1000,
        ge=1,
        description="Save interval for GAN models"
    )


class TrainModelsResponse(BaseModel):
    task_id: str
    status: str


class ModelsStatusResponse(BaseModel):
    markov_trained: bool
    rnn_trained: bool
    gan_trained: bool


class ModelsEvaluationResponse(BaseModel):
    markov_score: float
    rnn_score: float
    gan_score: float


@models_api_router.post(
    "/train",
    response_model=TrainModelsResponse,
    summary="Train all models with provided data"
)
def train_models(request: TrainModelsRequest, background_tasks: BackgroundTasks):
    """
    Train all models using the provided password data.
    """
    try:
        # Load and preprocess data
        passwords = load_passwords(request.data_source)
        if not passwords:
            raise HTTPException(status_code=400, detail="No passwords found in the data source")

        # Generate a unique task ID (for future enhancements)
        task_id = "train_all_models"

        # Add training as a background task
        background_tasks.add_task(
            hybrid_model.train_all,
            passwords=passwords,
            epochs_rnn=request.epochs_rnn,
            epochs_gan=request.epochs_gan,
            batch_size_rnn=request.batch_size_rnn,
            batch_size_gan=request.batch_size_gan,
            save_interval_gan=request.save_interval_gan
        )

        logging.info("Training initiated for all models.")
        return TrainModelsResponse(task_id=task_id, status="started")
    except Exception as e:
        logging.error(f"Error training models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@models_api_router.get(
    "/status",
    response_model=ModelsStatusResponse,
    summary="Get training status of models"
)
def get_models_status():
    """
    Get the training status of all models.
    """
    try:
        markov_trained = hybrid_model.markov.model is not None and len(hybrid_model.markov.model) > 0
        rnn_trained = hybrid_model.rnn.model is not None
        gan_trained = hybrid_model.gan.generator is not None and hybrid_model.gan.discriminator is not None
        return ModelsStatusResponse(
            markov_trained=markov_trained,
            rnn_trained=rnn_trained,
            gan_trained=gan_trained
        )
    except Exception as e:
        logging.error(f"Error getting models status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@models_api_router.get(
    "/evaluate",
    response_model=ModelsEvaluationResponse,
    summary="Evaluate the performance of the models"
)
def evaluate_models_endpoint():
    """
    Evaluate the performance of all models.
    """
    try:
        scores = evaluate_models(hybrid_model)
        return ModelsEvaluationResponse(
            markov_score=scores.get('markov', 0.0),
            rnn_score=scores.get('rnn', 0.0),
            gan_score=scores.get('gan', 0.0)
        )
    except Exception as e:
        logging.error(f"Error evaluating models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

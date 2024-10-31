# app/api/__init__.py

from .password_generator import password_generator_router
from .hashcat_api import hashcat_api_router
from .models_api import models_api_router
from .train_models import train_models_router

__all__ = [
    "password_generator_router",
    "hashcat_api_router",
    "models_api_router",
    "train_models_router",
]

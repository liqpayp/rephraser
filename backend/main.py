from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.train_models import train_models_router
from app.api.password_generator import password_generator_router
from app.api.hashcat_api import hashcat_api_router
from app.api.models_api import models_api_router
import logging
import os

app = FastAPI(
    title="Hybrid Models API",
    description="API for training models, generating passwords, and integrating with Hashcat.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend's origin here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(train_models_router, prefix="/api/train", tags=["Train Models"])
app.include_router(password_generator_router, prefix="/api/generate", tags=["Generate Password"])
app.include_router(hashcat_api_router, prefix="/api/hashcat", tags=["Hashcat Integration"])
app.include_router(models_api_router, prefix="/api/models", tags=["Models API"])

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Hybrid Models API!"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

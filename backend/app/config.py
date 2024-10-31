# backend/app/config.py

import json
from typing import List
from pydantic import Field, field_validator
from pydantic_settings import SettingsConfigDict, BaseSettings


class Settings(BaseSettings):
    app_name: str = Field(default="Hybrid Models API", env="APP_NAME")
    admin_email: str = Field(..., env="ADMIN_EMAIL")

    allowed_origins: List[str] = Field(
        default=["http://localhost", "http://localhost:3000"],
        env="ALLOWED_ORIGINS"
    )

    @field_validator("allowed_origins", mode="before")
    def parse_allowed_origins(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    # Hashcat settings
    hashcat_path: str = Field(default="/usr/bin/hashcat", env="HASHCAT_PATH")
    hashcat_results_dir: str = Field(default="hashcat/results/", env="HASHCAT_RESULTS_DIR")
    hashcat_hashes_dir: str = Field(default="hashcat/hashes/", env="HASHCAT_HASHES_DIR")

    # Model paths
    markov_model_path: str = Field(default="models/markov_model.pkl", env="MARKOV_MODEL_PATH")
    rnn_model_path: str = Field(default="models/rnn_model.h5", env="RNN_MODEL_PATH")
    gan_model_path: str = Field(default="models/gan_model.h5", env="GAN_MODEL_PATH")
    hybrid_model_path: str = Field(default="models/hybrid_model.h5", env="HYBRID_MODEL_PATH")

    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/app.log", env="LOG_FILE")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")

    # Hashcat settings (additional if needed)
    hash_file_dir: str = Field(default="hashcat/hashes/", env="HASH_FILE_DIR")
    hashcat_results_dir: str = Field(default="hashcat/results/", env="HASHCAT_RESULTS_DIR")

    # Model Config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
    )


settings = Settings()
print(settings.dict())

# app/api/train_models.py

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Dict
import json
import os
from ..utils.pattern_analyzer import AutomaticPatternAnalyzer
import logging

train_models_router = APIRouter()
logger = logging.getLogger(__name__)


@train_models_router.post("/train-models")
async def train_models_endpoint(
        corpus_file: UploadFile = File(...),
        models: str = Form(...),
        incremental: bool = Form(False)
):
    try:
        # Сохраняем временный файл
        temp_dir = "backend/app/temp/"
        os.makedirs(temp_dir, exist_ok=True)
        temp_corpus_path = os.path.join(temp_dir, corpus_file.filename)

        with open(temp_corpus_path, "wb") as buffer:
            content = await corpus_file.read()
            buffer.write(content)

        # Читаем пароли из файла
        with open(temp_corpus_path, 'r', encoding='utf-8') as f:
            passwords = [line.strip() for line in f if line.strip()]

        # Инициализируем анализатор паттернов
        pattern_analyzer = AutomaticPatternAnalyzer()

        # Анализируем корпус
        patterns = pattern_analyzer.analyze_corpus(passwords)
        logger.info(f"Found patterns: {json.dumps(patterns, indent=2)}")

        # Настраиваем конфигурацию моделей с учетом обнаруженных паттернов
        model_configs = {
            'markov': {
                'patterns': patterns,
                'password_analyzer': pattern_analyzer
            },
            'rnn': {
                'patterns': patterns,
                'password_analyzer': pattern_analyzer
            },
            'gan': {
                'patterns': patterns,
                'password_analyzer': pattern_analyzer
            }
        }

        # Обучаем выбранные модели
        selected_models = json.loads(models)
        training_results = {}

        for model_name in selected_models:
            if model_name == "markov":
                from ..models.markov_model import ImprovedMarkovModel
                model = ImprovedMarkovModel(model_configs['markov'])
                training_results['markov'] = model.train(passwords, incremental)

            elif model_name == "rnn":
                from ..models.rnn_model import AdvancedRNNModel
                model = AdvancedRNNModel(model_configs['rnn'])
                training_results['rnn'] = model.train(passwords)

            elif model_name == "gan":
                from ..models.gan_model import AdvancedGANModel
                model = AdvancedGANModel(model_configs['gan'])
                training_results['gan'] = model.train(passwords)

            # Сохраняем модель и её паттерны
            model_path = f"app/models/{model_name}_model.pkl"
            patterns_path = f"app/models/{model_name}_patterns.json"

            model.save(model_path)
            with open(patterns_path, 'w') as f:
                json.dump(patterns, f, indent=2)

        # Очищаем временные файлы
        os.remove(temp_corpus_path)

        return {
            "message": "Models trained successfully",
            "pattern_analysis": {
                "total_patterns": {
                    ptype: len(pdict) for ptype, pdict in patterns.items()
                },
                "top_patterns": {
                    ptype: dict(sorted(pdict.items(), key=lambda x: x[1], reverse=True)[:5])
                    for ptype, pdict in patterns.items()
                }
            },
            "training_results": training_results
        }

    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

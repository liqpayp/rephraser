# app/models/markov_model.py
import pickle
from collections import defaultdict, Counter
from typing import Dict, List, Optional
import os
import logging
import random
import numpy as np

logger = logging.getLogger(__name__)


class ImprovedMarkovModel:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logger

        # Основные параметры
        self.order = self.config.get('order', 3)
        self.min_length = self.config.get('min_length', 8)
        self.max_length = self.config.get('max_length', 16)

        # Инициализация базовых структур
        self._reset_models()

        # Загружаем модель если существует
        model_path = self.config.get('model_path')
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            self.logger.info("No existing model found, starting fresh")

    def _reset_models(self):
        """Инициализация/сброс всех моделей и структур данных"""
        # Модели для разных порядков
        self.models = {
            i: defaultdict(lambda: defaultdict(float))
            for i in range(1, self.order + 1)
        }

        # Хранилище паттернов
        self.patterns = defaultdict(Counter)

        # Статистика успешности
        self.success_stats = {
            'patterns': defaultdict(Counter),
            'contexts': defaultdict(Counter),
            'transitions': defaultdict(Counter)
        }

        # Веса паттернов
        self.pattern_weights = defaultdict(float)

        # Кэш для частых переходов
        self.transition_cache = {}

        # Метрики генерации
        self.generation_stats = {
            'successful_patterns': Counter(),
            'failed_patterns': Counter(),
            'pattern_success_rate': defaultdict(float)
        }

    def train(self, passwords: List[str], incremental: bool = False) -> Dict:
        """Обучение модели с анализом паттернов"""
        if not incremental:
            self._reset_models()

        # Обучаем модели разных порядков
        for password in passwords:
            # Предобработка пароля
            processed = self._preprocess_password(password)

            # Обучаем модели всех порядков (для backoff)
            for order in range(1, self.order + 1):
                self._train_single_order(processed, order)

            # Обновляем статистику паттернов
            self._update_pattern_stats(password)

        # Нормализуем вероятности и обновляем веса
        self._normalize_probabilities()
        self._update_pattern_weights()

        return {
            'total_passwords': len(passwords),
            'patterns_found': len(self.patterns),
            'model_sizes': {order: len(model) for order, model in self.models.items()}
        }

    def _preprocess_password(self, password: str) -> str:
        """Предобработка пароля"""
        # Добавляем специальные токены начала и конца
        return '^' * self.order + password + '$'

    def _train_single_order(self, password: str, order: int):
        """Обучение модели определенного порядка"""
        for i in range(len(password) - order):
            context = password[i:i + order]
            next_char = password[i + order]
            self.models[order][context][next_char] += 1

    def _update_pattern_stats(self, password: str):
        """Обновление статистики паттернов"""
        # Находим подстроки разной длины
        for length in range(2, min(len(password), 5)):
            for i in range(len(password) - length + 1):
                pattern = password[i:i + length]
                self.patterns[length][pattern] += 1

    def _normalize_probabilities(self):
        """Нормализация вероятностей переходов"""
        for order in self.models:
            for context in self.models[order]:
                total = sum(self.models[order][context].values())
                if total > 0:
                    for char in self.models[order][context]:
                        self.models[order][context][char] /= total

    def _update_pattern_weights(self):
        """Обновление весов паттернов"""
        total_patterns = sum(len(patterns) for patterns in self.patterns.values())
        if total_patterns > 0:
            for length, patterns in self.patterns.items():
                for pattern, count in patterns.items():
                    self.pattern_weights[pattern] = count / total_patterns

    def generate_password(
            self,
            min_length: Optional[int] = None,
            max_length: Optional[int] = None,
            temperature: float = 0.8
    ) -> str:
        """Генерация пароля"""
        min_length = min_length or self.min_length
        max_length = max_length or self.max_length

        password = ""
        context = "^" * self.order

        while len(password) < max_length:
            next_char = self._predict_next_char(context, temperature)
            if next_char == "$" and len(password) >= min_length:
                break
            if next_char not in {"^", "$"}:
                password += next_char
            context = (context + next_char)[-self.order:]

        return password

    def _predict_next_char(self, context: str, temperature: float) -> str:
        """Предсказание следующего символа"""
        # Используем backoff - начинаем с наибольшего порядка
        for order in range(self.order, 0, -1):
            current_context = context[-order:]
            if current_context in self.models[order]:
                probs = self.models[order][current_context]
                if probs:
                    return self._sample_char(probs, temperature)

        # Если не нашли контекст, используем униграммную модель
        if '^' in self.models[1]:
            return self._sample_char(self.models[1]['^'], temperature)
        return '$'

    def _sample_char(self, probs: Dict[str, float], temperature: float) -> str:
        """Выборка символа с учетом temperature sampling"""
        if temperature != 1.0:
            # Применяем temperature
            probs = {k: v ** (1 / temperature) for k, v in probs.items()}

        total = sum(probs.values())
        if total == 0:
            return '$'

        # Нормализуем вероятности
        probs = {k: v / total for k, v in probs.items()}

        # Выбираем символ
        chars, weights = zip(*probs.items())
        return np.random.choice(chars, p=weights)

    def save(self, path: str):
        """Сохранение модели"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'config': self.config,
            'models': {k: dict(v) for k, v in self.models.items()},
            'patterns': dict(self.patterns),
            'pattern_weights': dict(self.pattern_weights),
            'success_stats': {k: dict(v) for k, v in self.success_stats.items()},
            'generation_stats': {k: dict(v) for k, v in self.generation_stats.items()}
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        self.logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Загрузка модели"""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.config = state['config']
        self.models = {k: defaultdict(lambda: defaultdict(float), v)
                       for k, v in state['models'].items()}
        self.patterns = defaultdict(Counter, state['patterns'])
        self.pattern_weights = defaultdict(float, state['pattern_weights'])
        self.success_stats = {k: Counter(v) for k, v in state['success_stats'].items()}
        self.generation_stats = {k: Counter(v) for k, v in state['generation_stats'].items()}

        self.logger.info(f"Model loaded from {path}")

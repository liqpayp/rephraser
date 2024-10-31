# app/models/hybrid_model.py

import numpy as np
from typing import List, Dict, Optional
import logging
import os
import pickle
from collections import Counter
import torch
from .markov_model import ImprovedMarkovModel
from .rnn_model import AdvancedRNNModel
from .gan_model import AdvancedGANModel

logger = logging.getLogger(__name__)


class AdvancedHybridModel:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logger

        # Инициализация компонентных моделей
        self.markov = ImprovedMarkovModel(self.config.get('markov', {}))
        self.rnn = AdvancedRNNModel(self.config.get('rnn', {}))
        self.gan = AdvancedGANModel(self.config.get('gan', {}))

        # Адаптивные веса моделей
        self.model_weights = {
            'markov': self.config.get('markov_weight', 0.3),
            'rnn': self.config.get('rnn_weight', 0.4),
            'gan': self.config.get('gan_weight', 0.3)
        }

        # Статистика успешности
        self.success_stats = {
            'markov': Counter(),
            'rnn': Counter(),
            'gan': Counter()
        }

        # Путь для сохранения
        self.model_path = self.config.get('model_path', 'app/models/hybrid_model.pkl')

        # Статистика генерации
        self.generation_stats = {
            'model_success_rates': {
                'markov': [],
                'rnn': [],
                'gan': []
            },
            'pattern_frequencies': Counter(),
            'successful_patterns': Counter(),
            'failed_patterns': Counter()
        }

    def train(self, passwords: List[str], validation_split: float = 0.1) -> Dict:
        """Обучение всех моделей с валидацией"""
        results = {}

        # Разделяем данные
        split_idx = int(len(passwords) * (1 - validation_split))
        train_passwords = passwords[:split_idx]
        val_passwords = passwords[split_idx:]

        # Обучаем Markov модель
        self.logger.info("Training Markov model...")
        markov_results = self.markov.train(train_passwords)
        results['markov'] = markov_results

        # Обучаем RNN
        self.logger.info("Training RNN model...")
        rnn_results = self.rnn.train(train_passwords, validation_split=validation_split)
        results['rnn'] = rnn_results

        # Обучаем GAN
        self.logger.info("Training GAN model...")
        gan_results = self.gan.train(train_passwords, validation_split=validation_split)
        results['gan'] = gan_results

        # Оцениваем модели на валидационной выборке
        if val_passwords:
            val_scores = self._evaluate_models(val_passwords)
            results['validation'] = val_scores

            # Обновляем веса моделей на основе валидации
            self._update_model_weights(val_scores)

        return results

    def generate_password(
            self,
            min_length: Optional[int] = None,
            max_length: Optional[int] = None,
            temperature: float = 0.8,
            num_candidates: int = 5
    ) -> str:
        """Генерация пароля с использованием всех моделей"""
        candidates = []
        scores = []

        # Количество паролей от каждой модели пропорционально их весам
        model_candidates = {
            'markov': max(1, int(num_candidates * self.model_weights['markov'])),
            'rnn': max(1, int(num_candidates * self.model_weights['rnn'])),
            'gan': max(1, int(num_candidates * self.model_weights['gan']))
        }

        # Генерируем кандидатов от каждой модели
        for model_name, num in model_candidates.items():
            model = getattr(self, model_name)
            for _ in range(num):
                try:
                    password = model.generate_password(
                        min_length=min_length,
                        max_length=max_length,
                        temperature=temperature
                    )

                    # Оцениваем пароль
                    score = self._evaluate_password(password)

                    candidates.append({
                        'password': password,
                        'score': score,
                        'model': model_name
                    })
                    scores.append(score)
                except Exception as e:
                    self.logger.error(f"Error generating password with {model_name}: {e}")

        if not candidates:
            self.logger.warning("No candidates generated, using fallback")
            return self._generate_fallback_password(min_length or 12)

        # Выбираем лучший пароль
        best_candidate = max(candidates, key=lambda x: x['score'])

        # Обновляем статистику
        self._update_generation_stats(best_candidate)

        return best_candidate['password']

    def _evaluate_password(self, password: str) -> float:
        """Комплексная оценка качества пароля"""
        if not password:
            return 0.0

        scores = []

        # Получаем оценки от каждой модели
        try:
            markov_score = self.markov.evaluate_password(password)
            scores.append(markov_score * self.model_weights['markov'])
        except:
            pass

        try:
            rnn_score = self.rnn.evaluate_password(password)
            scores.append(rnn_score * self.model_weights['rnn'])
        except:
            pass

        try:
            gan_score = self.gan.evaluate_password(password)
            scores.append(gan_score * self.model_weights['gan'])
        except:
            pass

        if not scores:
            return 0.0

        # Комбинируем оценки
        return np.mean(scores)

    def _evaluate_models(self, passwords: List[str]) -> Dict:
        """Оценка качества моделей на валидационной выборке"""
        scores = {
            'markov': [],
            'rnn': [],
            'gan': []
        }

        for password in passwords:
            # Генерируем пароли каждой моделью
            try:
                markov_pwd = self.markov.generate_password()
                scores['markov'].append(self._evaluate_password(markov_pwd))
            except:
                pass

            try:
                rnn_pwd = self.rnn.generate_password()
                scores['rnn'].append(self._evaluate_password(rnn_pwd))
            except:
                pass

            try:
                gan_pwd = self.gan.generate_password()
                scores['gan'].append(self._evaluate_password(gan_pwd))
            except:
                pass

        return {
            model: np.mean(model_scores) if model_scores else 0.0
            for model, model_scores in scores.items()
        }

    def _update_model_weights(self, scores: Dict[str, float]):
        """Обновление весов моделей на основе их производительности"""
        total_score = sum(scores.values())
        if total_score > 0:
            self.model_weights = {
                model: score / total_score
                for model, score in scores.items()
            }
            self.logger.info(f"Updated model weights: {self.model_weights}")

    def _update_generation_stats(self, candidate: Dict):
        """Обновление статистики генерации"""
        # Обновляем успешность модели
        model = candidate['model']
        score = candidate['score']
        self.generation_stats['model_success_rates'][model].append(score)

        # Ограничиваем историю
        max_history = 1000
        if len(self.generation_stats['model_success_rates'][model]) > max_history:
            self.generation_stats['model_success_rates'][model] = \
                self.generation_stats['model_success_rates'][model][-max_history:]

    def update_with_feedback(self, password: str, success: bool):
        """Обновление на основе результатов взлома"""
        # Обновляем статистику паттернов
        patterns = self._extract_patterns(password)
        stats_dict = (
            self.generation_stats['successful_patterns'] if success
            else self.generation_stats['failed_patterns']
        )

        for pattern in patterns:
            stats_dict[pattern] += 1

        # Обновляем каждую модель
        self.markov.update_with_cracked(password, success)
        self.rnn.update_with_feedback(password, success)
        self.gan.update_with_feedback(password, success)

        # Периодически обновляем веса моделей
        if len(self.generation_stats['successful_patterns']) % 100 == 0:
            self._update_weights_from_stats()

    def _update_weights_from_stats(self):
        """Обновление весов на основе статистики успешности"""
        for model in self.model_weights:
            rates = self.generation_stats['model_success_rates'][model]
            if rates:
                # Используем экспоненциальное скользящее среднее
                avg_rate = np.mean(rates[-100:])  # берем последние 100 результатов
                self.model_weights[model] = 0.7 * self.model_weights[model] + 0.3 * avg_rate

        # Нормализация весов
        total = sum(self.model_weights.values())
        if total > 0:
            self.model_weights = {
                k: v / total for k, v in self.model_weights.items()
            }

    def _extract_patterns(self, password: str, min_length: int = 2, max_length: int = 4) -> List[str]:
        """Извлечение паттернов из пароля"""
        patterns = []
        for length in range(min_length, max_length + 1):
            for i in range(len(password) - length + 1):
                pattern = password[i:i + length]
                patterns.append(pattern)
        return patterns

    def _generate_fallback_password(self, length: int) -> str:
        """Генерация базового пароля если все модели отказали"""
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
        return ''.join(np.random.choice(list(chars), size=length))

    def save(self, path: Optional[str] = None):
        """Сохранение гибридной модели"""
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state = {
            'config': self.config,
            'model_weights': self.model_weights,
            'generation_stats': self.generation_stats
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        # Сохраняем компонентные модели
        component_paths = {
            'markov': path.replace('.pkl', '_markov.pkl'),
            'rnn': path.replace('.pkl', '_rnn.h5'),
            'gan': path.replace('.pkl', '_gan.h5')
        }

        self.markov.save(component_paths['markov'])
        self.rnn.save_model(component_paths['rnn'])
        self.gan.save_model(component_paths['gan'])

        self.logger.info(f"Hybrid model saved to {path}")

    def load(self, path: Optional[str] = None):
        """Загрузка гибридной модели"""
        path = path or self.model_path

        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.config = state['config']
        self.model_weights = state['model_weights']
        self.generation_stats = state['generation_stats']

        # Загружаем компонентные модели
        component_paths = {
            'markov': path.replace('.pkl', '_markov.pkl'),
            'rnn': path.replace('.pkl', '_rnn.h5'),
            'gan': path.replace('.pkl', '_gan.h5')
        }

        self.markov.load(component_paths['markov'])
        self.rnn.load_model(component_paths['rnn'])
        self.gan.load_model(component_paths['gan'])

        self.logger.info(f"Hybrid model loaded from {path}")

    def get_stats(self) -> Dict:
        """Получение общей статистики"""
        return {
            'model_weights': self.model_weights,
            'average_success_rates': {
                model: np.mean(rates[-100:]) if rates else 0.0
                for model, rates in self.generation_stats['model_success_rates'].items()
            },
            'top_successful_patterns': dict(
                self.generation_stats['successful_patterns'].most_common(10)
            ),
            'top_failed_patterns': dict(
                self.generation_stats['failed_patterns'].most_common(10)
            ),
            'component_stats': {
                'markov': self.markov.get_stats(),
                'rnn': self.rnn.get_stats() if hasattr(self.rnn, 'get_stats') else {},
                'gan': self.gan.get_generation_stats()
            }
        }

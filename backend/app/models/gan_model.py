# app/models/gan_model.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Embedding, Dropout,
    BatchNormalization, LayerNormalization,
    Bidirectional, MultiHeadAttention, Reshape,
    LeakyReLU, Flatten
)
from tensorflow.keras.optimizers import Adam
import os
import pickle
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class AdvancedGANModel:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logger

        # Основные параметры
        self.latent_dim = self.config.get('latent_dim', 128)
        self.embedding_dim = self.config.get('embedding_dim', 256)
        self.sequence_length = self.config.get('sequence_length', 20)
        self.batch_size = self.config.get('batch_size', 64)

        # Маппинги символов
        self.char_to_idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx_to_char = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.vocab_size = len(self.char_to_idx)

        # Модели
        self.generator = None
        self.discriminator = None
        self.gan = None

        # Статистика генерации
        self.generation_stats = {
            'successful_patterns': {},
            'failed_patterns': {},
            'char_frequencies': {}
        }

        # Пути к файлам
        self.model_path = self.config.get('model_path', 'app/models/gan_model.h5')
        self.tokenizer_path = self.config.get('tokenizer_path', 'app/models/gan_tokenizer.pkl')

        # Загружаем существующую модель если есть
        if os.path.exists(self.model_path) and os.path.exists(self.tokenizer_path):
            self.load_model()
        else:
            self.logger.info("No existing model found, will create new one when training")

    def build_generator(self) -> Model:
        """Создание улучшенного генератора"""
        # Входной шум
        noise = Input(shape=(self.latent_dim,))

        # Первый dense блок
        x = Dense(self.latent_dim * 2)(noise)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization()(x)

        # Reshape для sequence
        x = Dense(self.sequence_length * self.embedding_dim)(x)
        x = Reshape((self.sequence_length, self.embedding_dim))(x)
        x = LayerNormalization()(x)

        # LSTM блок с attention
        lstm_out = Bidirectional(LSTM(
            self.embedding_dim,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2
        ))(x)

        # Multi-head attention
        attn_out = MultiHeadAttention(
            num_heads=8,
            key_dim=self.embedding_dim
        )(lstm_out, lstm_out, lstm_out)

        # Skip connection
        x = tf.keras.layers.Add()([lstm_out, attn_out])
        x = LayerNormalization()(x)

        # Выходной слой для каждой позиции в последовательности
        outputs = Dense(self.vocab_size, activation='softmax')(x)

        return Model(noise, outputs, name='generator')

    def build_discriminator(self) -> Model:
        """Создание улучшенного дискриминатора"""
        # Входная последовательность
        sequence = Input(shape=(self.sequence_length, self.vocab_size))

        # Embedding и нормализация
        x = Dense(self.embedding_dim)(sequence)
        x = LayerNormalization()(x)

        # Bidirectional LSTM
        lstm_out = Bidirectional(LSTM(
            self.embedding_dim,
            return_sequences=True,
            dropout=0.2
        ))(x)

        # Multi-head attention
        attn_out = MultiHeadAttention(
            num_heads=8,
            key_dim=self.embedding_dim
        )(lstm_out, lstm_out, lstm_out)

        # Skip connection
        x = tf.keras.layers.Add()([lstm_out, attn_out])
        x = LayerNormalization()(x)

        # Dense layers
        x = Flatten()(x)
        x = Dense(512)(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.3)(x)

        # Выход - вероятность реальности последовательности
        outputs = Dense(1, activation='sigmoid')(x)

        return Model(sequence, outputs, name='discriminator')

    def train(self, passwords: List[str], validation_split: float = 0.1) -> Dict:
        """Обучение GAN"""
        # Обновляем словарь
        self._update_vocab(passwords)

        # Создаем модели если нужно
        if self.generator is None:
            self.generator = self.build_generator()
        if self.discriminator is None:
            self.discriminator = self.build_discriminator()

        # Компилируем discriminator
        self.discriminator.compile(
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Создаем GAN
        self.discriminator.trainable = False
        gan_input = Input(shape=(self.latent_dim,))
        gen_sequence = self.generator(gan_input)
        gan_output = self.discriminator(gen_sequence)
        self.gan = Model(gan_input, gan_output)
        self.gan.compile(
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy'
        )

        # Подготавливаем данные
        real_sequences = self._prepare_sequences(passwords)

        # Обучение
        epochs = self.config.get('epochs', 100)
        batch_size = self.config.get('batch_size', 64)
        history = {
            'disc_loss': [],
            'disc_acc': [],
            'gen_loss': []
        }

        for epoch in range(epochs):
            # Обучаем discriminator
            idx = np.random.randint(0, real_sequences.shape[0], batch_size)
            real_batch = real_sequences[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            generated_sequences = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(
                real_batch,
                np.ones((batch_size, 1)) * 0.9  # label smoothing
            )
            d_loss_fake = self.discriminator.train_on_batch(
                generated_sequences,
                np.zeros((batch_size, 1))
            )
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Обучаем generator
            noise = np.random.normal(0, 1, (batch_size * 2, self.latent_dim))
            g_loss = self.gan.train_on_batch(
                noise,
                np.ones((batch_size * 2, 1))
            )

            # Сохраняем историю
            history['disc_loss'].append(d_loss[0])
            history['disc_acc'].append(d_loss[1])
            history['gen_loss'].append(g_loss)

            if epoch % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch}, D Loss: {d_loss[0]:.4f}, "
                    f"D Acc: {100 * d_loss[1]:.1f}%, G Loss: {g_loss:.4f}"
                )

        # Сохраняем модель и токенизатор
        self.save_model()
        self._save_tokenizer()

        return {
            'disc_loss': np.mean(history['disc_loss']),
            'disc_acc': np.mean(history['disc_acc']),
            'gen_loss': np.mean(history['gen_loss'])
        }

    def generate_password(
            self,
            min_length: Optional[int] = None,
            max_length: Optional[int] = None,
            temperature: float = 0.8,
            num_candidates: int = 5
    ) -> str:
        """Генерация пароля"""
        if self.generator is None:
            raise ValueError("Model not trained or loaded")

        min_length = min_length or self.config.get('min_length', 8)
        max_length = max_length or self.config.get('max_length', 16)

        candidates = []
        scores = []

        for _ in range(num_candidates):
            # Генерируем шум
            noise = np.random.normal(0, 1, (1, self.latent_dim))

            # Получаем последовательность
            sequence = self.generator.predict(noise)[0]

            # Применяем temperature sampling
            sequence = np.log(sequence) / temperature
            sequence = np.exp(sequence) / np.sum(np.exp(sequence), axis=-1, keepdims=True)

            # Преобразуем в пароль
            password = self._sequence_to_password(sequence)

            # Оцениваем качество
            score = self.evaluate_password(password)

            if min_length <= len(password) <= max_length:
                candidates.append(password)
                scores.append(score)

        if not candidates:
            # Если нет подходящих кандидатов, возвращаем базовый пароль
            return self._generate_basic_password(min_length)

        # Возвращаем лучший пароль
        best_idx = np.argmax(scores)
        return candidates[best_idx]

    def evaluate_password(self, password: str) -> float:
        """Оценка качества пароля"""
        if len(password) < 6:
            return 0.0

        score = 0.0

        # Оценка длины
        length_score = min(len(password) / 16, 1.0)
        score += length_score * 0.2

        # Оценка сложности
        has_lower = any(c.islower() for c in password)
        has_upper = any(c.isupper() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(not c.isalnum() for c in password)
        complexity_score = (has_lower + has_upper + has_digit + has_special) / 4
        score += complexity_score * 0.3

        # Оценка энтропии
        char_freq = {}
        for char in password:
            char_freq[char] = char_freq.get(char, 0) + 1
        entropy = sum(-freq / len(password) * np.log2(freq / len(password))
                      for freq in char_freq.values())
        entropy_score = min(entropy / 4, 1.0)
        score += entropy_score * 0.3

        # Оценка от дискриминатора
        if self.discriminator is not None:
            sequence = self._password_to_sequence(password)
            disc_score = self.discriminator.predict(np.array([sequence]), verbose=0)[0][0]
            score += disc_score * 0.2

        return score

    def _update_vocab(self, passwords: List[str]):
        """Обновление словаря символов"""
        chars = set(''.join(passwords))
        for char in chars:
            if char not in self.char_to_idx:
                idx = len(self.char_to_idx)
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char

        self.vocab_size = len(self.char_to_idx)
        self.logger.info(f"Vocabulary size: {self.vocab_size}")

    def _prepare_sequences(self, passwords: List[str]) -> np.ndarray:
        """Подготовка последовательностей для обучения"""
        sequences = []
        for password in passwords:
            sequence = ['<sos>'] + list(password) + ['<eos>']
            sequence = [self.char_to_idx.get(c, self.char_to_idx['<unk>'])
                        for c in sequence]
            # One-hot encoding
            one_hot = np.zeros((self.sequence_length, self.vocab_size))
            for i, idx in enumerate(sequence[:self.sequence_length]):
                one_hot[i, idx] = 1
            sequences.append(one_hot)
        return np.array(sequences)

    def _sequence_to_password(self, sequence: np.ndarray) -> str:
        """Преобразование последовательности в пароль"""
        password = []
        for probs in sequence:
            idx = np.argmax(probs)
            char = self.idx_to_char[idx]
            if char == '<eos>':
                break
            if char not in {'<pad>', '<sos>', '<unk>'}:
                password.append(char)
        return ''.join(password)

    def _password_to_sequence(self, password: str) -> np.ndarray:
        """Преобразование пароля в sequence"""
        sequence = ['<sos>'] + list(password) + ['<eos>']
        sequence = [self.char_to_idx.get(c, self.char_to_idx['<unk>'])
                    for c in sequence]

        # One-hot encoding
        one_hot = np.zeros((self.sequence_length, self.vocab_size))
        for i, idx in enumerate(sequence[:self.sequence_length]):
            one_hot[i, idx] = 1
        return one_hot

    def _generate_basic_password(self, length: int) -> str:
        """Генерация базового пароля если все кандидаты не подходят"""
        chars = list(self.char_to_idx.keys())
        chars = [c for c in chars if c not in {'<pad>', '<sos>', '<eos>', '<unk>'}]
        return ''.join(np.random.choice(chars, size=length))

    def save_model(self):
        """Сохранение моделей"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        # Сохраняем отдельные модели
        self.generator.save(self.model_path.replace('.h5', '_generator.h5'))
        self.discriminator.save(self.model_path.replace('.h5', '_discriminator.h5'))

        # Сохраняем статистику
        stats_path = self.model_path.replace('.h5', '_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(self.generation_stats, f)

        self.logger.info(f"Models saved to {self.model_path}")

    def _save_tokenizer(self):
        """Сохранение токенизатора"""
        tokenizer_data = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocab_size': self.vocab_size
        }

        os.makedirs(os.path.dirname(self.tokenizer_path), exist_ok=True)
        with open(self.tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer_data, f)

        self.logger.info(f"Tokenizer saved to {self.tokenizer_path}")

    def load_model(self):
        """Загрузка моделей и токенизатора"""
        try:
            # Загружаем токенизатор
            with open(self.tokenizer_path, 'rb') as f:
                tokenizer_data = pickle.load(f)
                self.char_to_idx = tokenizer_data['char_to_idx']
                self.idx_to_char = tokenizer_data['idx_to_char']
                self.vocab_size = tokenizer_data['vocab_size']

            # Загружаем модели
            self.generator = load_model(self.model_path.replace('.h5', '_generator.h5'))
            self.discriminator = load_model(self.model_path.replace('.h5', '_discriminator.h5'))

            # Загружаем статистику
            stats_path = self.model_path.replace('.h5', '_stats.pkl')
            if os.path.exists(stats_path):
                with open(stats_path, 'rb') as f:
                    self.generation_stats = pickle.load(f)

            self.logger.info("Models loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise

    def update_with_feedback(self, password: str, success: bool):
        """Обновление статистики на основе обратной связи"""
        # Обновляем статистику успешных/неуспешных паттернов
        patterns = self._extract_patterns(password)
        stats_dict = (
            self.generation_stats['successful_patterns'] if success
            else self.generation_stats['failed_patterns']
        )

        for pattern in patterns:
            stats_dict[pattern] = stats_dict.get(pattern, 0) + 1

        # Обновляем частоты символов
        for char in password:
            if char in self.char_to_idx:
                self.generation_stats['char_frequencies'][char] = \
                    self.generation_stats['char_frequencies'].get(char, 0) + 1

    def _extract_patterns(self, password: str, min_length: int = 2, max_length: int = 4) -> List[str]:
        """Извлечение паттернов из пароля"""
        patterns = []
        for length in range(min_length, max_length + 1):
            for i in range(len(password) - length + 1):
                pattern = password[i:i + length]
                patterns.append(pattern)
        return patterns

    def get_generation_stats(self) -> Dict:
        """Получение статистики генерации"""
        return {
            'vocab_size': self.vocab_size,
            'successful_patterns': dict(sorted(
                self.generation_stats['successful_patterns'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
            'failed_patterns': dict(sorted(
                self.generation_stats['failed_patterns'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
            'char_frequencies': dict(sorted(
                self.generation_stats['char_frequencies'].items(),
                key=lambda x: x[1],
                reverse=True
            ))
        }

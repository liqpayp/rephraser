# app/models/rnn_model.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Embedding, Bidirectional,
    Dropout, LayerNormalization, Input,
    MultiHeadAttention
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle
import os
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class AdvancedRNNModel:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logger

        # Основные параметры
        self.embedding_dim = self.config.get('embedding_dim', 256)
        self.hidden_size = self.config.get('hidden_size', 512)
        self.num_layers = self.config.get('num_layers', 3)
        self.dropout = self.config.get('dropout', 0.3)
        self.sequence_length = self.config.get('sequence_length', 20)

        # Инициализация маппингов символов
        self.char_to_idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx_to_char = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.vocab_size = len(self.char_to_idx)

        # Модель
        self.model = None

        # Получаем путь к модели
        self.model_path = self.config.get('model_path', 'app/models/rnn_model.h5')
        self.tokenizer_path = self.config.get('tokenizer_path', 'app/models/rnn_tokenizer.pkl')

        # Загружаем существующую модель если есть
        if os.path.exists(self.model_path) and os.path.exists(self.tokenizer_path):
            self.load_model()
        else:
            self.logger.info("No existing model found, will create new one when training")

    def build_model(self) -> Model:
        """Создание улучшенной архитектуры модели"""
        # Входной слой
        inputs = Input(shape=(self.sequence_length,))

        # Embedding слой
        x = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True
        )(inputs)

        # Layer normalization
        x = LayerNormalization()(x)

        # Bidirectional LSTM слои с residual connections
        for i in range(self.num_layers):
            lstm_out = Bidirectional(
                LSTM(
                    self.hidden_size,
                    return_sequences=True,
                    dropout=self.dropout,
                    recurrent_dropout=self.dropout / 2
                )
            )(x)

            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=8,
                key_dim=self.hidden_size
            )(lstm_out, lstm_out, lstm_out)

            # Skip connection
            x = tf.keras.layers.Add()([x, attention_output])
            x = LayerNormalization()(x)

        # Выходные слои
        x = Dropout(self.dropout)(x)
        outputs = Dense(self.vocab_size, activation='softmax')(x)

        # Создаем модель
        model = Model(inputs=inputs, outputs=outputs)

        # Компилируем
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, passwords: List[str], validation_split: float = 0.1) -> Dict:
        """Обучение модели"""
        # Обновляем словарь
        self._update_vocab(passwords)

        # Подготавливаем данные
        X, y = self._prepare_sequences(passwords)

        # Создаем модель если нужно
        if self.model is None:
            self.model = self.build_model()
            self.logger.info("Created new model")

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                self.model_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            )
        ]

        # Разделяем на train и validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Обучаем
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.get('epochs', 50),
            batch_size=self.config.get('batch_size', 64),
            callbacks=callbacks
        )

        # Сохраняем tokenizer
        self._save_tokenizer()

        return {
            'training_loss': history.history['loss'][-1],
            'validation_loss': history.history['val_loss'][-1],
            'training_accuracy': history.history['accuracy'][-1],
            'validation_accuracy': history.history['val_accuracy'][-1]
        }

    def generate_password(
            self,
            min_length: Optional[int] = None,
            max_length: Optional[int] = None,
            temperature: float = 0.8
    ) -> str:
        """Генерация пароля"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        min_length = min_length or self.config.get('min_length', 8)
        max_length = max_length or self.config.get('max_length', 16)

        # Начинаем с токена начала последовательности
        current_sequence = [self.char_to_idx['<sos>']]
        generated_password = []

        while len(generated_password) < max_length:
            # Паддинг последовательности
            padded_sequence = self._pad_sequence(current_sequence)

            # Получаем предсказания
            predictions = self.model.predict(np.array([padded_sequence]), verbose=0)[0, -1]

            # Применяем temperature sampling
            predictions = np.log(predictions) / temperature
            exp_preds = np.exp(predictions)
            predictions = exp_preds / np.sum(exp_preds)

            # Выбираем следующий символ
            next_char_idx = np.random.choice(len(predictions), p=predictions)
            next_char = self.idx_to_char[next_char_idx]

            # Проверяем специальные токены
            if next_char == '<eos>' and len(generated_password) >= min_length:
                break
            if next_char not in {'<pad>', '<sos>', '<eos>', '<unk>'}:
                generated_password.append(next_char)

            current_sequence.append(next_char_idx)

        return ''.join(generated_password)

    def _update_vocab(self, passwords: List[str]):
        """Обновление словаря символов"""
        # Собираем все уникальные символы
        chars = set(''.join(passwords))

        # Добавляем новые символы в словарь
        for char in chars:
            if char not in self.char_to_idx:
                idx = len(self.char_to_idx)
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char

        self.vocab_size = len(self.char_to_idx)
        self.logger.info(f"Vocabulary size: {self.vocab_size}")

    def _prepare_sequences(self, passwords: List[str]):
        """Подготовка последовательностей для обучения"""
        X, y = [], []

        for password in passwords:
            # Добавляем специальные токены
            sequence = ['<sos>'] + list(password) + ['<eos>']

            # Создаем последовательности для обучения
            for i in range(1, len(sequence)):
                input_seq = sequence[:i]
                target_char = sequence[i]

                # Конвертируем в индексы
                input_seq = [self.char_to_idx.get(c, self.char_to_idx['<unk>'])
                             for c in input_seq]
                target_idx = self.char_to_idx.get(target_char, self.char_to_idx['<unk>'])

                # Паддинг
                input_seq = self._pad_sequence(input_seq)

                X.append(input_seq)
                y.append(target_idx)

        return np.array(X), np.array(y)

    def _pad_sequence(self, sequence: List[int]) -> List[int]:
        """Паддинг последовательности до нужной длины"""
        if len(sequence) >= self.sequence_length:
            return sequence[-self.sequence_length:]
        else:
            padding = [self.char_to_idx['<pad>']] * (self.sequence_length - len(sequence))
            return padding + sequence

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
        """Загрузка модели и токенизатора"""
        # Загружаем токенизатор
        with open(self.tokenizer_path, 'rb') as f:
            tokenizer_data = pickle.load(f)
            self.char_to_idx = tokenizer_data['char_to_idx']
            self.idx_to_char = tokenizer_data['idx_to_char']
            self.vocab_size = tokenizer_data['vocab_size']

        # Загружаем модель
        self.model = load_model(self.model_path)
        self.logger.info(f"Model loaded from {self.model_path}")

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

        # Оценка предсказуемости
        pred_score = self._evaluate_predictability(password)
        score += pred_score * 0.2

        return score

    def _evaluate_predictability(self, password: str) -> float:
        """Оценка предсказуемости пароля"""
        if self.model is None:
            return 0.5

        sequence = ['<sos>'] + list(password)
        total_prob = 0

        for i in range(1, len(sequence)):
            # Подготовка входной последовательности
            input_seq = sequence[:i]
            input_seq = [self.char_to_idx.get(c, self.char_to_idx['<unk>'])
                         for c in input_seq]
            input_seq = self._pad_sequence(input_seq)

            # Получаем предсказания
            predictions = self.model.predict(np.array([input_seq]), verbose=0)[0, -1]

            # Вероятность следующего символа
            next_char = sequence[i]
            next_idx = self.char_to_idx.get(next_char, self.char_to_idx['<unk>'])
            total_prob += predictions[next_idx]

        # Среднее значение вероятности
        avg_prob = total_prob / (len(sequence) - 1)

        # Инвертируем оценку (менее предсказуемые пароли получают более высокий счет)
        return 1 - avg_prob

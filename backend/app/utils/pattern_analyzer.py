# app/utils/pattern_analyzer.py

from typing import List, Dict, Set, Tuple
from collections import defaultdict, Counter
import numpy as np
import re
from itertools import combinations
import logging


class AutomaticPatternAnalyzer:
    """Анализатор для автоматического обнаружения паттернов в паролях"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Хранилище обнаруженных паттернов
        self.discovered_patterns = defaultdict(Counter)

        # Статистика успешных взломов
        self.successful_patterns = defaultdict(Counter)

        # Веса для разных типов паттернов
        self.pattern_weights = defaultdict(float)

        # Минимальная частота для признания паттерна
        self.min_pattern_frequency = 0.05  # 5% от общего количества

    def analyze_corpus(self, passwords: List[str]) -> Dict:
        """Полный анализ корпуса паролей"""
        total_passwords = len(passwords)
        analysis = {
            'character_patterns': self._analyze_character_patterns(passwords),
            'structural_patterns': self._analyze_structural_patterns(passwords),
            'sequence_patterns': self._analyze_sequences(passwords),
            'position_patterns': self._analyze_position_patterns(passwords),
            'frequency_patterns': self._analyze_frequency_patterns(passwords),
            'substitution_patterns': self._analyze_substitutions(passwords)
        }

        # Обновляем общие паттерны с учетом частоты
        for pattern_type, patterns in analysis.items():
            for pattern, count in patterns.items():
                frequency = count / total_passwords
                if frequency >= self.min_pattern_frequency:
                    self.discovered_patterns[pattern_type][pattern] = frequency

        return analysis

    def _analyze_character_patterns(self, passwords: List[str]) -> Counter:
        """Анализ паттернов на уровне символов"""
        patterns = Counter()

        for password in passwords:
            # Анализ подстрок разной длины
            for length in range(2, 6):  # паттерны от 2 до 5 символов
                for i in range(len(password) - length + 1):
                    substring = password[i:i + length]
                    patterns[substring] += 1

            # Анализ повторяющихся символов
            repeats = re.finditer(r'(.)\1+', password)
            for match in repeats:
                patterns[f"repeat_{match.group()}"] += 1

        return patterns

    def _analyze_structural_patterns(self, passwords: List[str]) -> Counter:
        """Анализ структурных паттернов"""
        patterns = Counter()

        for password in passwords:
            # Получаем структуру пароля (l-lower, u-upper, d-digit, s-special)
            structure = self._get_password_structure(password)
            patterns[structure] += 1

            # Анализ подструктур
            for length in range(2, len(structure) + 1):
                for i in range(len(structure) - length + 1):
                    substructure = structure[i:i + length]
                    patterns[f"substruct_{substructure}"] += 1

        return patterns

    def _analyze_sequences(self, passwords: List[str]) -> Counter:
        """Анализ последовательностей"""
        sequences = Counter()

        for password in passwords:
            # Числовые последовательности
            nums = re.finditer(r'\d+', password)
            for num in nums:
                if self._is_sequence(num.group()):
                    sequences[f"num_seq_{num.group()}"] += 1

            # Буквенные последовательности
            chars = re.finditer(r'[a-zA-Z]+', password)
            for char in chars:
                if self._is_char_sequence(char.group()):
                    sequences[f"char_seq_{char.group()}"] += 1

            # Клавиатурные последовательности
            keyboard_sequences = self._find_keyboard_sequences(password)
            for seq in keyboard_sequences:
                sequences[f"keyboard_{seq}"] += 1

        return sequences

    def _analyze_position_patterns(self, passwords: List[str]) -> Counter:
        """Анализ позиционных паттернов"""
        positions = Counter()

        for password in passwords:
            length = len(password)

            # Анализ начала пароля
            positions[f"start_{password[:3].lower()}"] += 1

            # Анализ конца пароля
            positions[f"end_{password[-3:].lower()}"] += 1

            # Анализ позиций типов символов
            for i, char in enumerate(password):
                char_type = self._get_char_type(char)
                relative_pos = i / length  # Относительная позиция
                pos_category = int(relative_pos * 4)  # 4 категории позиций
                positions[f"{char_type}_pos_{pos_category}"] += 1

        return positions

    def _analyze_frequency_patterns(self, passwords: List[str]) -> Counter:
        """Анализ частотных паттернов"""
        freqs = Counter()

        # Общий счетчик символов
        char_counts = Counter(''.join(passwords))
        total_chars = sum(char_counts.values())

        # Находим часто встречающиеся комбинации
        for password in passwords:
            # Биграммы и триграммы
            for n in [2, 3]:
                for i in range(len(password) - n + 1):
                    ngram = password[i:i + n]
                    freqs[f"ngram_{ngram}"] += 1

            # Анализ чередования типов символов
            char_types = [self._get_char_type(c) for c in password]
            for i in range(len(char_types) - 1):
                pattern = f"{char_types[i]}_{char_types[i + 1]}"
                freqs[f"alternation_{pattern}"] += 1

        return freqs

    def _analyze_substitutions(self, passwords: List[str]) -> Counter:
        """Анализ замен символов"""
        substitutions = Counter()

        # Определяем возможные замены
        common_subs = {
            'a': ['@', '4'],
            'e': ['3'],
            'i': ['1', '!'],
            'o': ['0'],
            's': ['$', '5'],
            't': ['7'],
            'b': ['8'],
            'g': ['9'],
            'l': ['1'],
            'z': ['2']
        }

        for password in passwords:
            # Ищем все возможные замены
            for char, replacements in common_subs.items():
                for replacement in replacements:
                    if replacement in password:
                        # Проверяем контекст замены
                        indices = [i for i, c in enumerate(password) if c == replacement]
                        for idx in indices:
                            context = self._get_substitution_context(password, idx)
                            substitutions[f"sub_{char}_to_{replacement}_in_{context}"] += 1

        return substitutions

    def update_with_cracked(self, password: str, success: bool):
        """Обновление статистики на основе результатов взлома"""
        # Анализируем пароль
        patterns = {
            'character': self._analyze_character_patterns([password]),
            'structural': self._analyze_structural_patterns([password]),
            'sequence': self._analyze_sequences([password]),
            'position': self._analyze_position_patterns([password]),
            'frequency': self._analyze_frequency_patterns([password]),
            'substitution': self._analyze_substitutions([password])
        }

        # Обновляем статистику успешных паттернов
        weight = 1 if success else -0.5
        for pattern_type, pattern_dict in patterns.items():
            for pattern, _ in pattern_dict.items():
                self.successful_patterns[pattern_type][pattern] += weight

        # Обновляем веса паттернов
        self._update_pattern_weights()

    def _update_pattern_weights(self):
        """Обновление весов паттернов на основе статистики успехов"""
        for pattern_type, patterns in self.successful_patterns.items():
            total = sum(patterns.values())
            if total > 0:
                for pattern, count in patterns.items():
                    self.pattern_weights[f"{pattern_type}_{pattern}"] = count / total

    @staticmethod
    def _get_char_type(char: str) -> str:
        """Определение типа символа"""
        if char.isupper():
            return 'U'
        elif char.islower():
            return 'L'
        elif char.isdigit():
            return 'D'
        else:
            return 'S'

    @staticmethod
    def _get_password_structure(password: str) -> str:
        """Получение структуры пароля"""
        return ''.join(AutomaticPatternAnalyzer._get_char_type(c) for c in password)

    @staticmethod
    def _is_sequence(s: str) -> bool:
        """Проверка является ли строка последовательностью"""
        if len(s) < 3:
            return False

        # Для цифр
        if s.isdigit():
            nums = [int(d) for d in s]
            diffs = [nums[i + 1] - nums[i] for i in range(len(nums) - 1)]
            return len(set(diffs)) == 1

        return False

    @staticmethod
    def _is_char_sequence(s: str) -> bool:
        """Проверка является ли строка последовательностью букв"""
        if len(s) < 3:
            return False

        s = s.lower()
        # Проверяем последовательность в алфавите
        diffs = [ord(s[i + 1]) - ord(s[i]) for i in range(len(s) - 1)]
        return len(set(diffs)) == 1

    @staticmethod
    def _get_substitution_context(password: str, pos: int, context_size: int = 1) -> str:
        """Получение контекста вокруг замены"""
        start = max(0, pos - context_size)
        end = min(len(password), pos + context_size + 1)
        return password[start:end]

    @staticmethod
    def _find_keyboard_sequences(password: str, min_length: int = 3) -> Set[str]:
        """Поиск клавиатурных последовательностей"""
        # Определяем раскладку клавиатуры
        keyboard_layout = {
            'row1': 'qwertyuiop',
            'row2': 'asdfghjkl',
            'row3': 'zxcvbnm',
            'numbers': '1234567890'
        }

        sequences = set()
        password = password.lower()

        # Проверяем каждую строку клавиатуры
        for row in keyboard_layout.values():
            for i in range(len(password) - min_length + 1):
                substring = password[i:i + min_length]
                # Проверяем прямое и обратное вхождение в строку клавиатуры
                if substring in row or substring in row[::-1]:
                    sequences.add(substring)

        return sequences

    def get_pattern_recommendations(self) -> Dict[str, float]:
        """Получение рекомендаций по использованию паттернов"""
        recommendations = {}

        # Сортируем паттерны по успешности
        for pattern_type, patterns in self.successful_patterns.items():
            for pattern, count in patterns.most_common(10):  # топ-10 для каждого типа
                if count > 0:  # только успешные паттерны
                    weight = self.pattern_weights[f"{pattern_type}_{pattern}"]
                    recommendations[f"{pattern_type}_{pattern}"] = weight

        return recommendations

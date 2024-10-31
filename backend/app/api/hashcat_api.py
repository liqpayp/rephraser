# app/api/hashcat_api.py

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
import json
import os
from ..utils.pattern_analyzer import AutomaticPatternAnalyzer
import logging

hashcat_api_router = APIRouter()
logger = logging.getLogger(__name__)


@hashcat_api_router.post("/transfer-passwords")
async def transfer_passwords_endpoint(
        hash_file: UploadFile = File(...),
        attack_mode: str = Form(...),
        generated_passwords: str = Form(...),
        hashcat_path: str = Form(...),
        analyze_patterns: bool = Form(True)
):
    try:
        # Сохраняем хэш-файл
        temp_dir = "backend/app/temp/"
        os.makedirs(temp_dir, exist_ok=True)
        temp_hash_path = os.path.join(temp_dir, hash_file.filename)

        with open(temp_hash_path, "wb") as buffer:
            content = await hash_file.read()
            buffer.write(content)

        # Парсим пароли
        passwords = json.loads(generated_passwords)

        # Анализируем паттерны если требуется
        if analyze_patterns:
            analyzer = AutomaticPatternAnalyzer()
            patterns = analyzer.analyze_corpus(passwords)

            # Сохраняем анализ для последующего использования
            with open(os.path.join(temp_dir, "pattern_analysis.json"), 'w') as f:
                json.dump(patterns, f, indent=2)

        # Запускаем hashcat
        result = run_hashcat(
            hashcat_path,
            temp_hash_path,
            passwords,
            attack_mode,
            patterns if analyze_patterns else None
        )

        # Обновляем статистику паттернов на основе результатов
        if analyze_patterns and result.get('cracked_passwords'):
            for password in result['cracked_passwords']:
                analyzer.update_with_cracked(password, success=True)

            # Сохраняем обновленную статистику
            with open(os.path.join(temp_dir, "updated_patterns.json"), 'w') as f:
                json.dump(analyzer.get_pattern_recommendations(), f, indent=2)

        return {
            "cracked_passwords": result.get('cracked_passwords', []),
            "pattern_analysis": patterns if analyze_patterns else None,
            "success_rate": len(result.get('cracked_passwords', [])) / len(passwords)
        }

    except Exception as e:
        logger.error(f"Hashcat transfer error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_hashcat(
        hashcat_path: str,
        hash_file: str,
        passwords: List[str],
        attack_mode: str,
        patterns: dict = None
) -> dict:
    """
    Запуск hashcat с учетом паттернов
    """
    try:
        # Создаем словарь с паролями
        dict_file = "hashcat/dicts/generated_passwords.txt"
        os.makedirs(os.path.dirname(dict_file), exist_ok=True)

        with open(dict_file, 'w') as f:
            for password in passwords:
                f.write(f"{password}\n")

        # Если есть паттерны, создаем правила на их основе
        if patterns:
            rules = create_hashcat_rules(patterns)
            rules_file = "hashcat/rules/generated_rules.txt"

            os.makedirs(os.path.dirname(rules_file), exist_ok=True)
            with open(rules_file, 'w') as f:
                for rule in rules:
                    f.write(f"{rule}\n")

        # Запускаем hashcat
        # ... (остальной код hashcat)

        return {"cracked_passwords": cracked_passwords}

    except Exception as e:
        logger.error(f"Hashcat error: {e}")
        raise


def create_hashcat_rules(patterns: dict) -> List[str]:
    """
    Создание правил hashcat на основе паттернов
    """
    rules = []

    # Правила на основе символьных паттернов
    if 'character_patterns' in patterns:
        for pattern, freq in patterns['character_patterns'].items():
            if len(pattern) > 1:
                rules.append(f"i{pattern}")  # вставка паттерна
                rules.append(f"${pattern}")  # добавление в конец

    # Правила на основе структурных паттернов
    if 'structural_patterns' in patterns:
        for pattern in patterns['structural_patterns']:
            if 'ULSD' in pattern:  # Upper, Lower, Special, Digit
                rules.append(f"c")  # capitalize
                rules.append(f"C")  # lowercase
                rules.append(f"$1")  # add digit
                rules.append(f"$!")  # add special

    # Правила на основе замен
    if 'substitution_patterns' in patterns:
        for sub_pattern in patterns['substitution_patterns']:
            if 'to' in sub_pattern:
                from_char, to_char = sub_pattern.split('_to_')
                rules.append(f"s{from_char}{to_char}")

    return rules

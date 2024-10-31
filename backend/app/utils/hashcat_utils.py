# backend/app/utils/hashcat_utils.py

import os
import shutil
import subprocess
import logging

logger = logging.getLogger(__name__)


def execute_hashcat(hashcat_path: str, hash_type: str, attack_mode: str, hash_file: str, password_file: str,
                    output_file: str):
    """
    Execute Hashcat with the specified parameters.

    - **hashcat_path**: Path to the Hashcat executable.
    - **hash_type**: Hash type (e.g., '0' for MD5).
    - **attack_mode**: Attack mode number.
    - **hash_file**: Path to the hash file.
    - **password_file**: Path to the password file.
    - **output_file**: Path to save cracked passwords.
    """
    try:
        command = [
            hashcat_path,
            "-m", hash_type,
            "-a", attack_mode,
            hash_file,
            password_file,
            "--outfile", output_file,
            "--quiet"
        ]
        logger.info(f"Executing Hashcat command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Hashcat error: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout, stderr=result.stderr)
        logger.info("Hashcat executed successfully.")
    except Exception as e:
        logger.error(f"Error executing Hashcat: {e}")
        raise e

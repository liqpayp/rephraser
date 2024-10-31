# tests/test_hashcat.py

import pytest
from unittest.mock import patch, MagicMock
from ..utils.hashcat_utils import HashcatManager
from typing import List


def test_start_cracking():
    with patch('app.utils.hashcat_utils.subprocess.Popen') as mock_popen:
        # Mock the subprocess.Popen object and its behavior
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        mock_process.communicate.return_value = (b'', b'')
        mock_process.returncode = 0

        hashcat_manager = HashcatManager()
        task_id = hashcat_manager.start_cracking(
            hash_type=0,
            hashes=["5f4dcc3b5aa765d61d8327deb882cf99"],
            wordlist="tests/wordlists/test_wordlist.txt"
        )

        assert task_id in hashcat_manager.tasks, "Task ID should be registered in the manager."
        task = hashcat_manager.tasks[task_id]
        assert task.status == "running", "Task status should be 'running'."


def test_get_status():
    hashcat_manager = HashcatManager()
    # Create a dummy task
    task_id = "test-task-id"
    task = hashcat_manager.tasks[task_id] = MagicMock()
    task.task_id = task_id
    task.status = "completed"
    task.progress = 100.0
    task.cracked = 1
    task.total = 1

    status = hashcat_manager.get_status(task_id)
    assert status["task_id"] == task_id, "Task ID should match."
    assert status["status"] == "completed", "Task status should be 'completed'."
    assert status["progress"] == 100.0, "Task progress should be 100.0."
    assert status["cracked"] == 1, "Cracked count should be 1."
    assert status["total"] == 1, "Total count should be 1."


def test_get_cracked_passwords():
    hashcat_manager = HashcatManager()
    # Create a dummy task with cracked passwords
    task_id = "test-task-id-2"
    task = MagicMock()
    task.cracked_passwords = ["password123"]
    hashcat_manager.tasks[task_id] = task

    cracked = hashcat_manager.get_cracked_passwords(task_id)
    assert cracked == ["password123"], "Cracked passwords should match the expected list."


def test_submit_hash_missing_hashcat():
    # Simulate Hashcat not being found
    with patch('app.utils.hashcat_utils.shutil.which', return_value=None):
        with patch('app.utils.hashcat_utils.os.path.exists', return_value=False):
            with pytest.raises(Exception) as exc_info:
                hashcat_manager = HashcatManager()
                hashcat_manager.start_cracking(
                    hash_type=0,
                    hashes=["5f4dcc3b5aa765d61d8327deb882cf99"],
                    wordlist="tests/wordlists/test_wordlist.txt"
                )
            assert "Hashcat not found on the system." in str(exc_info.value)

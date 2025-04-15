import json
import os

import pytest
from dotenv import load_dotenv


@pytest.fixture(scope="session", autouse=True)
def load_test_env():
    """Load test environment variables."""
    load_dotenv(".env.test")


@pytest.fixture
def mock_audio_data():
    """Provide mock audio data for testing."""
    return b"mock_audio_data"


@pytest.fixture
def settings():
    """Load settings.json for tests."""
    with open("config/settings.json") as f:
        return json.load(f)


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables."""
    monkeypatch.setenv("PICOVOICE_ACCESS_KEY", "test_picovoice_key_123")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key_456")
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("DEBUG", "true")


@pytest.fixture
def mock_settings():
    """Provide mock settings for testing."""
    return {
        "wake_word": "jarvis",
        "wake_word_sensitivity": 0.7,
        "audio": {
            "sample_rate": 16000,
            "chunk_size": 1024,
            "channels": 1,
            "format": "paInt16",
        },
        "speech_recognition": {
            "local_confidence_threshold": 0.8,
            "timeout": 10,
            "silence_threshold": 500,
            "silence_duration": 1.5,
        },
    }

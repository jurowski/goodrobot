import json
import os
import sys

from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_env_variables():
    """Test environment variables are properly loaded."""
    load_dotenv()

    # Test required environment variables exist
    assert "PICOVOICE_ACCESS_KEY" in os.environ
    assert "OPENAI_API_KEY" in os.environ
    assert "ENVIRONMENT" in os.environ
    assert "DEBUG" in os.environ

    # Test environment variable types
    assert isinstance(os.getenv("PICOVOICE_ACCESS_KEY"), str)
    assert isinstance(os.getenv("OPENAI_API_KEY"), str)
    assert os.getenv("ENVIRONMENT") in ["development", "test", "production"]
    assert os.getenv("DEBUG").lower() in ["true", "false"]


def test_settings_json():
    """Test settings.json configuration."""
    with open("config/settings.json") as f:
        settings = json.load(f)

    # Test wake word settings
    assert "wake_word" in settings
    assert isinstance(settings["wake_word"], str)
    assert isinstance(settings["wake_word_sensitivity"], (int, float))
    assert 0 <= settings["wake_word_sensitivity"] <= 1

    # Test audio settings
    assert "audio" in settings
    audio_settings = settings["audio"]
    assert isinstance(audio_settings["sample_rate"], int)
    assert isinstance(audio_settings["chunk_size"], int)
    assert isinstance(audio_settings["channels"], int)
    assert audio_settings["format"] == "paInt16"

    # Test speech recognition settings
    assert "speech_recognition" in settings
    sr_settings = settings["speech_recognition"]
    assert isinstance(sr_settings["local_confidence_threshold"], (int, float))
    assert 0 <= sr_settings["local_confidence_threshold"] <= 1
    assert isinstance(sr_settings["timeout"], (int, float))
    assert isinstance(sr_settings["silence_threshold"], int)
    assert isinstance(sr_settings["silence_duration"], (int, float))


def test_logging_config():
    """Test logging configuration."""
    with open("config/settings.json") as f:
        settings = json.load(f)

    assert "logging" in settings
    log_settings = settings["logging"]
    assert log_settings["level"] in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    assert isinstance(log_settings["max_size"], int)
    assert isinstance(log_settings["backup_count"], int)

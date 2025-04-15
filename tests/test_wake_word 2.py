import pytest
from unittest.mock import Mock, patch
import os
from src.voice_recognition.wake_word import WakeWordDetector

@pytest.fixture
def wake_word_detector():
    detector = WakeWordDetector()
    yield detector
    detector.cleanup()

def test_wake_word_detector_initialization(wake_word_detector):
    assert wake_word_detector is not None
    assert hasattr(wake_word_detector, 'wake_word')
    assert hasattr(wake_word_detector, 'sensitivity')

@patch('pvporcupine.create')
def test_wake_word_detection(mock_porcupine, wake_word_detector):
    mock_porcupine.return_value.process.return_value = 1
    
    result = wake_word_detector.check_audio(b"mock_audio_data")
    assert result is True

@patch('pvporcupine.create')
def test_no_wake_word_detected(mock_porcupine, wake_word_detector):
    mock_porcupine.return_value.process.return_value = -1
    
    result = wake_word_detector.check_audio(b"mock_audio_data")
    assert result is False

def test_sensitivity_range():
    with pytest.raises(ValueError):
        WakeWordDetector(sensitivity=1.5)  # Too high
    with pytest.raises(ValueError):
        WakeWordDetector(sensitivity=-0.1)  # Too low

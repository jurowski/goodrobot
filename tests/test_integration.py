import os
from unittest.mock import patch

import pytest

from src.voice_recognition.speech_to_text import HybridSpeechRecognizer
from src.voice_recognition.wake_word import WakeWordDetector
from src.voice_recognition.whisper_transcriber import WhisperTranscriber


@pytest.fixture
def setup_components():
    recognizer = HybridSpeechRecognizer()
    wake_word = WakeWordDetector()
    whisper = WhisperTranscriber()
    yield recognizer, wake_word, whisper
    recognizer.cleanup()
    wake_word.cleanup()
    whisper.cleanup()


@patch("pvporcupine.create")
@patch("speech_recognition.Recognizer")
def test_wake_word_to_transcription_flow(
    mock_recognizer, mock_porcupine, setup_components
):
    recognizer, wake_word, _ = setup_components

    # Mock wake word detection
    mock_porcupine.return_value.process.return_value = 1

    # Mock speech recognition
    mock_recognizer.return_value.recognize_google.return_value = "test command"
    mock_recognizer.return_value.confidence = 0.9

    # Test full flow
    wake_word_detected = wake_word.check_audio(b"mock_audio_data")
    assert wake_word_detected is True

    if wake_word_detected:
        text, metadata = recognizer.speech_to_text(b"mock_audio_data")
        assert text == "test command"
        assert metadata["source"] == "local"
        assert metadata["confidence"] >= 0.8


@patch("pvporcupine.create")
@patch("speech_recognition.Recognizer")
def test_fallback_to_whisper_flow(mock_recognizer, mock_porcupine, setup_components):
    recognizer, wake_word, _ = setup_components

    # Mock wake word detection
    mock_porcupine.return_value.process.return_value = 1

    # Mock low confidence local recognition
    mock_recognizer.return_value.recognize_google.return_value = "test command"
    mock_recognizer.return_value.confidence = 0.3

    # Test fallback flow
    wake_word_detected = wake_word.check_audio(b"mock_audio_data")
    assert wake_word_detected is True

    if wake_word_detected:
        text, metadata = recognizer.speech_to_text(b"mock_audio_data")
        assert metadata["source"] == "whisper"

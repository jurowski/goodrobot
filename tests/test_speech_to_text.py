import os
from unittest.mock import Mock, patch

import pytest

from src.voice_recognition.speech_to_text import HybridSpeechRecognizer
from src.voice_recognition.whisper_transcriber import WhisperTranscriber


@pytest.fixture
def hybrid_recognizer():
    recognizer = HybridSpeechRecognizer()
    yield recognizer
    recognizer.cleanup()


@pytest.fixture
def mock_audio():
    return b"mock_audio_data"


def test_hybrid_recognizer_initialization(hybrid_recognizer):
    assert hybrid_recognizer is not None
    assert hasattr(hybrid_recognizer, "local_confidence_threshold")
    assert 0 <= hybrid_recognizer.local_confidence_threshold <= 1


@patch("speech_recognition.Recognizer")
def test_local_recognition_success(mock_recognizer, hybrid_recognizer, mock_audio):
    mock_recognizer.return_value.recognize_google.return_value = "test transcription"
    mock_recognizer.return_value.confidence = 0.9

    text, metadata = hybrid_recognizer._recognize_local(mock_audio)
    assert text == "test transcription"
    assert metadata["confidence"] >= hybrid_recognizer.local_confidence_threshold
    assert metadata["source"] == "local"


@patch("speech_recognition.Recognizer")
def test_local_recognition_low_confidence(
    mock_recognizer, hybrid_recognizer, mock_audio
):
    mock_recognizer.return_value.recognize_google.return_value = "test transcription"
    mock_recognizer.return_value.confidence = 0.3

    text, metadata = hybrid_recognizer._recognize_local(mock_audio)
    assert text == "test transcription"
    assert metadata["confidence"] < hybrid_recognizer.local_confidence_threshold


@patch("src.voice_recognition.whisper_transcriber.WhisperTranscriber")
def test_whisper_fallback(mock_whisper, hybrid_recognizer, mock_audio):
    mock_whisper.return_value.transcribe_audio.return_value = (
        "whisper result",
        {"confidence": 0.95},
    )

    text, metadata = hybrid_recognizer.speech_to_text(mock_audio)
    assert text == "whisper result"
    assert metadata["source"] == "whisper"
    assert metadata["confidence"] == 0.95

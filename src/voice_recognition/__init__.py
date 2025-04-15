"""Voice recognition module for speech processing and transcription."""

from .speech_to_text import HybridSpeechRecognizer
from .whisper_transcriber import WhisperTranscriber

__all__ = ["HybridSpeechRecognizer", "WhisperTranscriber"]

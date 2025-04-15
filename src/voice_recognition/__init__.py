"""Voice recognition module for speech processing and transcription."""

from .speech_to_text import SpeechToText
from .wake_word import WakeWordDetector

__all__ = ["SpeechToText", "WakeWordDetector"]

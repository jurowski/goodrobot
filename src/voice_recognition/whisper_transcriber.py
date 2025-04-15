"""Cloud-based transcription using OpenAI's Whisper API."""

import json
import os
import wave
from datetime import datetime
from typing import Any, Dict, List, Tuple

from config.settings import Settings


class WhisperTranscriber:
    """Speech-to-text conversion using Whisper model."""

    def __init__(self, settings: Settings):
        """Initialize the Whisper transcriber.

        Args:
            settings: Application settings containing API keys and directories
        """
        self.settings = settings
        self._ensure_data_directory()

    def _ensure_data_directory(self):
        """Ensure the data directory exists."""
        os.makedirs(self.settings.data_dir, exist_ok=True)

    def transcribe_audio(self, audio_data: bytes) -> Tuple[str, Dict[str, Any]]:
        """Transcribe audio data to text using Whisper API.

        Args:
            audio_data: Raw audio bytes to transcribe

        Returns:
            Tuple containing:
            - transcribed text
            - metadata dictionary with confidence scores and other info
        """
        # Save audio to temporary file
        temp_file = os.path.join(
            self.settings.data_dir,
            f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
        )

        try:
            with wave.open(temp_file, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_data)

            # Transcribe using Whisper API
            text, metadata = self._call_whisper_api(temp_file)

            return text, metadata

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def _call_whisper_api(self, audio_file: str) -> Tuple[str, Dict[str, Any]]:
        """Call the Whisper API to transcribe audio.

        Args:
            audio_file: Path to the audio file to transcribe

        Returns:
            Tuple containing:
            - transcribed text
            - metadata dictionary with confidence scores and other info
        """
        import openai

        try:
            with open(audio_file, "rb") as f:
                response = openai.Audio.transcribe(
                    "whisper-1", f, api_key=self.settings.openai_api_key
                )

            return response.text, {
                "model": "whisper-1",
                "confidence": response.get("confidence", 1.0),
                "language": response.get("language", "en"),
                "duration": response.get("duration", 0),
            }

        except Exception as e:
            print(f"Whisper API error: {e}")
            return "", {"error": str(e)}

    def save_transcription(self, text: str, metadata: Dict[str, Any]):
        """Save transcription to file with metadata.

        Args:
            text: Transcribed text
            metadata: Additional metadata about the transcription
        """
        transcription = {
            "text": text,
            "metadata": metadata,
            "created_at": datetime.now().isoformat(),
        }

        # Save to file
        transcript_file = os.path.join(
            self.settings.data_dir,
            f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

        with open(transcript_file, "w") as f:
            json.dump(transcription, f, indent=2)

    def cleanup(self):
        """Clean up any resources."""
        pass  # No cleanup needed for this class


if __name__ == "__main__":
    # Example usage
    settings = Settings()
    stt = WhisperTranscriber(settings)

    # Sample audio data (simulated)
    audio_data = b"\x00\x01\x02\x03" * 1000

    # Transcribe audio
    transcription, metadata = stt.transcribe_audio(audio_data)
    print("Transcription:", transcription)

    # Save transcription
    stt.save_transcription(transcription, metadata)

    # Cleanup
    stt.cleanup()

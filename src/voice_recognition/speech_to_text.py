import json
import os
import wave
from datetime import datetime
from typing import Any, Dict, List

from config.settings import Settings
from src.audio.audio_processor import AudioProcessor


class SpeechToText:
    """Speech-to-text conversion using Whisper model."""

    def __init__(self, settings: Settings):
        """Initialize the speech-to-text converter."""
        self.settings = settings
        self.audio_processor = AudioProcessor(settings)
        self._ensure_data_directory()

    def _ensure_data_directory(self):
        """Ensure the data directory exists."""
        os.makedirs(self.settings.data_dir, exist_ok=True)

    def transcribe_audio(self, audio_data: bytes) -> str:
        """Transcribe audio data to text."""
        # Process audio data
        processed_audio = self.audio_processor.process(audio_data)

        # Save audio to temporary file
        temp_file = os.path.join(
            self.settings.data_dir,
            f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
        )

        with wave.open(temp_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(processed_audio)

        # Transcribe using Whisper API
        transcription = self._call_whisper_api(temp_file)

        # Clean up temporary file
        os.remove(temp_file)

        return transcription

    def _call_whisper_api(self, audio_file: str) -> str:
        """Call the Whisper API to transcribe audio."""
        # Implementation of Whisper API call
        return "Sample transcription"

    def process_streaming_audio(self, audio_chunks: List[bytes]) -> str:
        """Process streaming audio data and transcribe it."""
        # Combine audio chunks
        combined_audio = b"".join(audio_chunks)

        # Process and transcribe
        return self.transcribe_audio(combined_audio)

    def save_transcription(self, text: str, metadata: Dict[str, Any]):
        """Save transcription to file with metadata."""
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
        """Clean up resources."""
        self.audio_processor.cleanup()


if __name__ == "__main__":
    # Example usage
    settings = Settings()
    stt = SpeechToText(settings)

    # Sample audio data (simulated)
    audio_data = b"\x00\x01\x02\x03" * 1000

    # Transcribe audio
    transcription = stt.transcribe_audio(audio_data)
    print("Transcription:", transcription)

    # Save transcription
    stt.save_transcription(
        transcription, {"source": "microphone", "duration": 5.0, "language": "en"}
    )

    # Cleanup
    stt.cleanup()

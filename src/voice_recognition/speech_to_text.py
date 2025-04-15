import json
import os
import wave
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import openai
import numpy as np
import logging
import soundfile as sf
import io
import tempfile
import traceback
import librosa

from src.settings import Settings
from src.audio.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

class SpeechToText:
    """Speech-to-text conversion using Whisper model."""

    def __init__(self, settings: Settings):
        """Initialize the speech-to-text converter."""
        logger.info("Initializing SpeechToText")
        self.settings = settings
        self.audio_processor = AudioProcessor(settings)
        self._ensure_data_directory()
        
        # Initialize OpenAI client with API key from settings
        api_key = settings.openai_api_key
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self.client = openai.OpenAI(api_key=api_key)
        self.sample_rate = settings.sample_rate
        self.channels = settings.channels
        logger.debug(f"Sample rate: {self.sample_rate}, Channels: {self.channels}")

    def _ensure_data_directory(self):
        """Ensure the data directory exists."""
        os.makedirs(self.settings.data_dir, exist_ok=True)

    def validate_audio_format(self, audio_data: bytes) -> np.ndarray:
        """
        Validate and convert audio data to the correct format.
        """
        try:
            # If the data is already a numpy array, return it
            if isinstance(audio_data, np.ndarray):
                return audio_data

            # Try to read as WAV file first
            try:
                audio_buffer = io.BytesIO(audio_data)
                audio_array, file_sample_rate = sf.read(audio_buffer)
                if file_sample_rate != self.sample_rate:
                    audio_array = librosa.resample(
                        audio_array,
                        orig_sr=file_sample_rate,
                        target_sr=self.sample_rate
                    )
                return audio_array
            except Exception as e:
                logger.debug(f"Failed to read with soundfile: {str(e)}, trying direct conversion")
                # If that fails, assume it's raw PCM data
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                # Normalize to float32 between -1 and 1
                audio_array = audio_array.astype(np.float32) / 32768.0
                return audio_array

        except Exception as e:
            logger.error(f"Error validating audio format: {str(e)}")
            raise ValueError(f"Invalid audio format: {str(e)}")

    async def transcribe_audio(self, audio_data: Union[bytes, np.ndarray]) -> str:
        """
        Transcribe audio data to text.
        """
        try:
            # Validate and convert audio format
            audio_array = self.validate_audio_format(audio_data)
            
            # Ensure audio is in the correct format for OpenAI
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Create a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                sf.write(temp_file.name, audio_array, self.sample_rate)
                
                # Transcribe using OpenAI
                with open(temp_file.name, 'rb') as audio_file:
                    response = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text"
                    )
                    transcription = str(response)
                
                # Clean up the temporary file
                os.unlink(temp_file.name)
                
                return transcription
                
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Transcription failed: {str(e)}")

    async def process_streaming_audio(self, audio_chunks: List[bytes]) -> str:
        """Process streaming audio data and transcribe it."""
        # Combine audio chunks
        combined_audio = b"".join(audio_chunks)

        # Process and transcribe
        return await self.transcribe_audio(combined_audio)

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
        logger.info("Cleaning up SpeechToText")
        self.audio_processor.cleanup()


if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Example usage
        settings = Settings()
        stt = SpeechToText(settings)

        # Sample audio data (simulated)
        audio_data = b"\x00\x01\x02\x03" * 1000

        # Transcribe audio
        transcription = await stt.transcribe_audio(audio_data)
        print("Transcription:", transcription)

        # Save transcription
        stt.save_transcription(
            transcription, {"source": "microphone", "duration": 5.0, "language": "en"}
        )

        # Cleanup
        stt.cleanup()

    asyncio.run(main())

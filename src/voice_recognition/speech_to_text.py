import json
import os
import wave
from datetime import datetime
from typing import Any, Dict, List, Optional
import openai
import numpy as np
import logging
import soundfile as sf
import io

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
        """Validate and convert audio data to the correct format."""
        try:
            # Convert bytes to numpy array using soundfile
            with io.BytesIO(audio_data) as audio_buffer:
                audio_array, file_sample_rate = sf.read(audio_buffer)
                logger.debug(f"Input audio: shape={audio_array.shape}, sample_rate={file_sample_rate}")

            # Convert to mono if needed
            if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                logger.warning("Converting multi-channel audio to mono")
                audio_array = np.mean(audio_array, axis=1)

            # Resample if needed
            if file_sample_rate != self.sample_rate:
                logger.warning(f"Resampling audio from {file_sample_rate} to {self.sample_rate}")
                # TODO: Implement resampling if needed

            # Normalize audio
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            if np.max(np.abs(audio_array)) > 1.0:
                audio_array = audio_array / np.max(np.abs(audio_array))

            return audio_array

        except Exception as e:
            logger.error(f"Error validating audio format: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    async def transcribe_audio(self, audio_data: bytes) -> str:
        """Transcribe audio data to text."""
        try:
            # Validate and convert audio format
            audio_array = self.validate_audio_format(audio_data)
            
            # Convert to WAV format for OpenAI API
            with io.BytesIO() as wav_buffer:
                sf.write(wav_buffer, audio_array, self.sample_rate, format='WAV')
                wav_buffer.seek(0)
                
                logger.info("Sending audio to OpenAI for transcription")
                response = await self.client.audio.transcriptions.create(
                    file=wav_buffer,
                    model="whisper-1",
                    response_format="text"
                )
                
                logger.info(f"Transcription received: {response}")
                return response

        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

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

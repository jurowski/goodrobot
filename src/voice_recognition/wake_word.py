# src/voice_recognition/wake_word.py

import logging
import os
import struct
from typing import Callable, Optional
import numpy as np
import pvporcupine
import pyaudio
import io
import soundfile as sf

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """
    Wake word detection using Porcupine library.
    Listens for a specific wake word and triggers a callback when detected.
    """

    def __init__(
        self,
        settings,
        callback: Optional[Callable] = None,
    ):
        """
        Initialize wake word detector.

        Args:
            settings: Settings object containing configuration
            callback: Function to call when wake word is detected
        """
        logger.info("Initializing WakeWordDetector")
        self.settings = settings
        self.wake_word = settings.wake_word
        self.sensitivity = settings.sensitivity
        self.callback = callback
        self.access_key = settings.picovoice_access_key
        self.sample_rate = settings.sample_rate
        logger.debug(f"Wake word: {self.wake_word}, Sensitivity: {self.sensitivity}, Sample rate: {self.sample_rate}")
        
        self.porcupine = None
        self.audio = None
        self.stream = None
        self._is_listening = False
        self.initialize()

    def initialize(self):
        """Initialize Porcupine."""
        try:
            logger.info("Creating Porcupine instance...")
            
            # Use built-in keywords
            built_in_keywords = pvporcupine.KEYWORDS
            logger.info(f"Available keywords: {built_in_keywords}")
            
            if self.wake_word not in built_in_keywords:
                logger.warning(f"Wake word '{self.wake_word}' not in built-in keywords, defaulting to 'jarvis'")
                self.wake_word = "jarvis"
            
            self.porcupine = pvporcupine.create(
                access_key=self.access_key,
                keywords=[self.wake_word],
                sensitivities=[0.8]  # Increased sensitivity for better detection
            )
            
            logger.info(
                f"Wake word detector initialized successfully with wake word '{self.wake_word}'. "
                f"Sample rate: {self.porcupine.sample_rate}, Frame length: {self.porcupine.frame_length}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize wake word detector: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.cleanup()
            return False

    def process_audio_chunk(self, audio_data: bytes) -> bool:
        """
        Process a chunk of audio data and check for wake word.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            bool: True if wake word detected, False otherwise
        """
        try:
            # Convert WebM audio to raw PCM
            with io.BytesIO(audio_data) as audio_buffer:
                try:
                    # Try reading as WebM/WAV first
                    audio_array, sample_rate = sf.read(audio_buffer)
                    logger.debug(f"Successfully read audio with soundfile: shape={audio_array.shape}, sample_rate={sample_rate}")
                    
                    # Convert to mono if needed
                    if len(audio_array.shape) > 1:
                        audio_array = np.mean(audio_array, axis=1)
                        logger.debug("Converted stereo to mono")
                    
                    # Normalize audio to increase detection chances
                    max_val = np.max(np.abs(audio_array))
                    if max_val > 0:
                        audio_array = audio_array / max_val
                    
                    # Convert to int16
                    audio_array = (audio_array * 32767).astype(np.int16)
                    
                except Exception as e:
                    logger.debug(f"Failed to read with soundfile: {e}, trying direct conversion")
                    # If soundfile fails, try direct numpy conversion
                    # Ensure the buffer size is even (for int16)
                    if len(audio_data) % 2 != 0:
                        audio_data = audio_data[:-1]
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    # Normalize int16 audio
                    max_val = np.max(np.abs(audio_array))
                    if max_val > 0:
                        audio_array = (audio_array / max_val * 32767).astype(np.int16)

            logger.debug(f"Final audio array: shape={audio_array.shape}, dtype={audio_array.dtype}, min={np.min(audio_array)}, max={np.max(audio_array)}")
            
            # Ensure correct number of samples
            frame_length = self.porcupine.frame_length
            if len(audio_array) < frame_length:
                logger.debug(f"Padding audio from {len(audio_array)} to {frame_length} samples")
                audio_array = np.pad(audio_array, (0, frame_length - len(audio_array)))
            elif len(audio_array) > frame_length:
                logger.debug(f"Truncating audio from {len(audio_array)} to {frame_length} samples")
                # Take samples from the middle for better detection
                start = (len(audio_array) - frame_length) // 2
                audio_array = audio_array[start:start + frame_length]
            
            # Process the audio frame
            result = self.porcupine.process(audio_array)
            if result >= 0:
                logger.info(f"Wake word '{self.wake_word}' detected!")
                if self.callback:
                    self.callback()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up WakeWordDetector")
        if hasattr(self, 'porcupine'):
            self.porcupine.delete()
            self.porcupine = None
            logger.debug("Porcupine instance deleted")

    def set_callback(self, callback: Callable):
        """Set the callback function to be called when wake word is detected."""
        self.callback = callback


# Example usage
if __name__ == "__main__":
    import time

    # Callback function when wake word is detected
    def on_wake_word():
        print("Wake word detected! Listening for command...")
        # In a real implementation, this would trigger the speech-to-text process

    # Initialize wake word detector with your access key
    # You need to sign up for a free Picovoice account to get an access key
    access_key = os.getenv("PICOVOICE_ACCESS_KEY", "YOUR_ACCESS_KEY")

    detector = WakeWordDetector(
        access_key=access_key,
        wake_word="jarvis",  # Or choose another available keyword
        sensitivity=0.7,
        callback=on_wake_word,
    )

    # Start listening
    if detector.start():
        print("Listening for wake word 'jarvis'... Press Ctrl+C to exit")

        try:
            # Keep the program running
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            # Clean up resources
            detector.cleanup()
    else:
        print("Failed to start wake word detector")

# src/voice_recognition/__init__.py

from .wake_word import WakeWordDetector

__all__ = ["WakeWordDetector"]

# src/voice_recognition/wake_word.py

import logging
import os
import struct
from typing import Callable, Optional

import pvporcupine
import pyaudio

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """
    Wake word detection using Porcupine library.
    Listens for a specific wake word and triggers a callback when detected.
    """

    def __init__(
        self,
        access_key: str,
        wake_word: str = "jarvis",
        sensitivity: float = 0.5,
        callback: Optional[Callable] = None,
    ):
        """
        Initialize wake word detector.

        Args:
            access_key: Picovoice access key
            wake_word: Wake word to listen for (options depend on Porcupine)
            sensitivity: Detection sensitivity (0-1)
            callback: Function to call when wake word is detected
        """
        self.wake_word = wake_word
        self.sensitivity = sensitivity
        self.callback = callback
        self.access_key = access_key

        self.porcupine = None
        self.audio = None
        self.stream = None
        self._is_listening = False

    def initialize(self):
        """Initialize Porcupine and audio stream."""
        try:
            # Initialize Porcupine with the specified wake word
            self.porcupine = pvporcupine.create(
                access_key=self.access_key,
                keywords=[self.wake_word],
                sensitivities=[self.sensitivity],
            )

            # Initialize PyAudio
            self.audio = pyaudio.PyAudio()

            # Create audio stream
            self.stream = self.audio.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length,
                stream_callback=self._audio_callback,
            )

            logger.info(
                "Wake word detector initialized with wake word: %s",
                self.wake_word,
            )
            return True

        except Exception as e:
            logger.error("Failed to initialize wake word detector: %s", e)
            self.cleanup()
            return False

    def _audio_callback(self, in_data, frame_count, time_info, status_flags):
        """Process audio data and detect wake word."""
        if not self._is_listening:
            return (in_data, pyaudio.paContinue)

        try:
            # Convert audio data to PCM
            pcm = struct.unpack_from("h" * self.porcupine.frame_length, in_data)

            # Process audio with Porcupine
            keyword_index = self.porcupine.process(pcm)

            # If wake word detected (keyword_index >= 0)
            if keyword_index >= 0:
                logger.info("Wake word '%s' detected!", self.wake_word)

                # Call the callback function if provided
                if self.callback:
                    self.callback()

        except Exception as e:
            logger.error("Error processing audio: %s", e)

        # Continue listening
        return (in_data, pyaudio.paContinue)

    def start(self):
        """Start listening for wake word."""
        if not self.porcupine or not self.stream:
            success = self.initialize()
            if not success:
                return False

        # Start the audio stream
        self.stream.start_stream()
        self._is_listening = True
        logger.info("Wake word detector started")
        return True

    def stop(self):
        """Stop listening for wake word."""
        self._is_listening = False
        if self.stream:
            self.stream.stop_stream()
        logger.info("Wake word detector stopped")

    def cleanup(self):
        """Clean up resources."""
        self._is_listening = False

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        if self.audio:
            self.audio.terminate()
            self.audio = None

        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None

        logger.info("Wake word detector resources cleaned up")

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

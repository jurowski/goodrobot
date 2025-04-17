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
from scipy import signal

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
        self.noise_threshold = 0.15  # Default noise threshold
        self.calibration_data = []
        logger.debug(f"Wake word: {self.wake_word}, Sensitivity: {self.sensitivity}, Sample rate: {self.sample_rate}")
        
        self.porcupine = None
        self.audio = None
        self.stream = None
        self._is_listening = False
        self.initialize()

    def update_config(self, config: dict):
        """Update configuration parameters."""
        if 'sensitivity' in config:
            self.sensitivity = config['sensitivity']
            logger.info(f"Updated sensitivity to {self.sensitivity}")
            self.reinitialize()
            
        if 'noiseThreshold' in config:
            self.noise_threshold = config['noiseThreshold']
            logger.info(f"Updated noise threshold to {self.noise_threshold}")

    def add_calibration_sample(self, audio_data: bytes):
        """Add a calibration sample."""
        if len(self.calibration_data) < 5:
            self.calibration_data.append(audio_data)
            logger.info(f"Added calibration sample {len(self.calibration_data)}/5")
            
        if len(self.calibration_data) == 5:
            self._process_calibration()

    def _process_calibration(self):
        """Process calibration samples to optimize detection parameters."""
        try:
            logger.info("Processing calibration samples...")
            
            # Convert all samples to numpy arrays
            samples = []
            for audio_data in self.calibration_data:
                with io.BytesIO(audio_data) as audio_buffer:
                    audio_array, _ = sf.read(audio_buffer)
                    if len(audio_array.shape) > 1:
                        audio_array = np.mean(audio_array, axis=1)
                    samples.append(audio_array)
            
            # Calculate average energy and zero-crossing rate
            energies = []
            zcrs = []
            for sample in samples:
                energy = np.mean(np.square(sample))
                zcr = np.mean(np.abs(np.diff(np.signbit(sample))))
                energies.append(energy)
                zcrs.append(zcr)
            
            # Calculate optimal sensitivity based on sample consistency
            energy_std = np.std(energies)
            zcr_std = np.std(zcrs)
            consistency = 1 - min(1, (energy_std + zcr_std) / 2)
            
            # More consistent samples = lower sensitivity value (more sensitive)
            new_sensitivity = max(0.1, min(0.9, 0.5 - (consistency * 0.4)))
            
            # Calculate optimal noise threshold
            min_energy = min(energies)
            new_noise_threshold = max(0.05, min(0.3, min_energy * 0.8))
            
            # Update settings
            self.sensitivity = new_sensitivity
            self.noise_threshold = new_noise_threshold
            
            logger.info(f"Calibration complete. New sensitivity: {new_sensitivity}, New noise threshold: {new_noise_threshold}")
            
            # Reinitialize with new settings
            self.reinitialize()
            
            # Clear calibration data
            self.calibration_data = []
            
        except Exception as e:
            logger.error(f"Error processing calibration samples: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def reinitialize(self):
        """Reinitialize the wake word detector with current settings."""
        if self.porcupine:
            self.porcupine.delete()
        
        self.porcupine = pvporcupine.create(
            access_key=self.access_key,
            keywords=[self.wake_word],
            sensitivities=[self.sensitivity]
        )
        
        logger.info(
            f"Wake word detector reinitialized with sensitivity {self.sensitivity}"
        )

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
                sensitivities=[self.sensitivity]
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
            logger.debug(f"Processing audio chunk of size {len(audio_data)} bytes")
            
            # Convert WebM audio to raw PCM
            with io.BytesIO(audio_data) as audio_buffer:
                try:
                    # Try reading as WebM/WAV first
                    audio_array, sample_rate = sf.read(audio_buffer)
                    logger.debug(f"Audio format: shape={audio_array.shape}, sample_rate={sample_rate}, dtype={audio_array.dtype}")
                    
                    # Convert to mono if needed
                    if len(audio_array.shape) > 1:
                        audio_array = np.mean(audio_array, axis=1)
                        logger.debug("Converted stereo to mono")
                    
                    # Resample if needed
                    if sample_rate != self.sample_rate:
                        logger.debug(f"Resampling from {sample_rate}Hz to {self.sample_rate}Hz")
                        samples = int((len(audio_array) * self.sample_rate) / sample_rate)
                        audio_array = np.interp(
                            np.linspace(0, len(audio_array), samples, endpoint=False),
                            np.arange(len(audio_array)),
                            audio_array
                        )
                    
                    # Apply pre-emphasis filter to enhance high frequencies
                    pre_emphasis = 0.97
                    emphasized_audio = np.append(audio_array[0], audio_array[1:] - pre_emphasis * audio_array[:-1])
                    
                    # Apply bandpass filter for speech frequencies (80Hz - 4000Hz)
                    nyquist = self.sample_rate // 2
                    low = 80 / nyquist
                    high = 4000 / nyquist
                    b, a = signal.butter(4, [low, high], btype='band')
                    filtered_audio = signal.filtfilt(b, a, emphasized_audio)
                    
                    # Check if audio energy is above noise threshold
                    energy = np.mean(np.square(filtered_audio))
                    if energy < self.noise_threshold:
                        logger.debug(f"Audio energy {energy} below noise threshold {self.noise_threshold}")
                        return False
                    
                    # Normalize audio to increase detection chances
                    max_val = np.max(np.abs(filtered_audio))
                    if max_val > 0:
                        filtered_audio = filtered_audio / max_val
                        logger.debug(f"Normalized audio: max_val={max_val}")
                    
                    # Convert to int16
                    audio_array = (filtered_audio * 32767).astype(np.int16)
                    
                except Exception as e:
                    logger.warning(f"Failed to read with soundfile: {e}, trying direct conversion")
                    # If soundfile fails, try direct numpy conversion
                    if len(audio_data) % 2 != 0:
                        audio_data = audio_data[:-1]
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    max_val = np.max(np.abs(audio_array))
                    if max_val > 0:
                        audio_array = (audio_array / max_val * 32767).astype(np.int16)

            # Ensure correct number of samples
            frame_length = self.porcupine.frame_length
            if len(audio_array) < frame_length:
                logger.debug(f"Padding audio from {len(audio_array)} to {frame_length} samples")
                audio_array = np.pad(audio_array, (0, frame_length - len(audio_array)))
            elif len(audio_array) > frame_length:
                logger.debug(f"Processing audio in chunks of {frame_length} samples")
                # Process audio in overlapping chunks
                step = frame_length // 2  # 50% overlap
                for i in range(0, len(audio_array) - frame_length + 1, step):
                    chunk = audio_array[i:i + frame_length]
                    result = self.porcupine.process(chunk)
                    if result >= 0:
                        logger.info(f"Wake word '{self.wake_word}' detected!")
                        if self.callback:
                            self.callback()
                        return True
                return False
            
            # Process single frame
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

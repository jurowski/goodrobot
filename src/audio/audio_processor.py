"""Audio processing module for voice input handling and preprocessing."""

import array
import logging
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Configuration for audio processing."""

    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # 16-bit audio
    chunk_size: int = 1024
    max_amplitude: int = 32767  # Maximum value for 16-bit audio
    min_duration: float = 0.1  # Minimum duration in seconds
    max_duration: float = 10.0  # Maximum duration in seconds
    noise_threshold: float = 0.1  # Noise gate threshold (0-1)
    normalize_target: float = 0.9  # Target amplitude for normalization (0-1)


class AudioProcessor:
    """Audio processing for voice input with comprehensive error handling."""

    def __init__(self, config: Optional[AudioConfig] = None):
        """Initialize audio processor with configuration.

        Args:
            config: Optional audio processing configuration
        """
        self.config = config or AudioConfig()
        self._validate_config()
        self._reset_state()
        self._is_running = False
        logger.debug("AudioProcessor initialized with config: %s", self.config)

    async def start(self) -> None:
        """Start audio processing."""
        try:
            self._is_running = True
            self._reset_state()
            logger.info("Audio processor started")
        except Exception as e:
            logger.error("Error starting audio processor: %s", str(e))
            raise

    async def stop(self) -> None:
        """Stop audio processing."""
        try:
            self._is_running = False
            self._reset_state()
            logger.info("Audio processor stopped")
        except Exception as e:
            logger.error("Error stopping audio processor: %s", str(e))
            raise

    async def configure(self, config: Dict[str, Any]) -> None:
        """Configure the audio processor with new settings.

        Args:
            config: Dictionary of configuration parameters
        """
        try:
            # Update configuration
            if "sample_rate" in config:
                self.config.sample_rate = int(config["sample_rate"])
            if "channels" in config:
                self.config.channels = int(config["channels"])
            if "sample_width" in config:
                self.config.sample_width = int(config["sample_width"])
            if "chunk_size" in config:
                self.config.chunk_size = int(config["chunk_size"])
            if "noise_threshold" in config:
                self.config.noise_threshold = float(config["noise_threshold"])
            if "normalize_target" in config:
                self.config.normalize_target = float(config["normalize_target"])

            # Validate updated configuration
            self._validate_config()
            logger.info("Audio processor configuration updated: %s", self.config)

        except Exception as e:
            logger.error("Error configuring audio processor: %s", str(e))
            raise

    async def process_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """Process audio data and return results.

        Args:
            audio_data: Raw audio bytes to process

        Returns:
            Dictionary containing processing results
        """
        try:
            if not self._is_running:
                return {
                    "status": "error",
                    "error": "Audio processor is not running"
                }

            # Process the audio data
            processed_data = self.process(audio_data)

            # Convert to numpy array for analysis
            audio_array = np.frombuffer(processed_data, dtype=np.int16)

            # Calculate metrics
            metrics = {
                "peak_amplitude": float(np.max(np.abs(audio_array))),
                "rms": float(np.sqrt(np.mean(audio_array**2))),
                "duration": len(audio_array) / self.config.sample_rate,
                "processed_chunks": self._processed_chunks
            }

            return {
                "status": "success",
                "data": processed_data,
                "metrics": metrics
            }

        except Exception as e:
            logger.error("Error processing audio: %s", str(e))
            return {
                "status": "error",
                "error": str(e)
            }

    def _validate_config(self) -> None:
        """Validate audio configuration parameters."""
        if self.config.sample_rate <= 0:
            raise ValueError(f"Invalid sample rate: {self.config.sample_rate}")
        if self.config.channels <= 0:
            raise ValueError(f"Invalid channel count: {self.config.channels}")
        if self.config.sample_width not in [1, 2, 4]:
            raise ValueError(f"Invalid sample width: {self.config.sample_width}")

    def _reset_state(self) -> None:
        """Reset internal processing state."""
        self._buffer = array.array("h")
        self._peak_amplitude = 0
        self._dc_offset = 0
        self._processed_chunks = 0

    def process(
        self, audio_data: bytes, metadata: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """Process audio data with comprehensive error handling.

        Args:
            audio_data: Raw audio bytes to process
            metadata: Optional processing metadata

        Returns:
            Processed audio bytes

        Raises:
            ValueError: If audio data is invalid
            RuntimeError: If processing fails
        """
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Validate input
            self._validate_input(audio_array)

            # Apply processing pipeline
            processed = self._apply_processing_pipeline(audio_array)

            # Update state
            self._update_state(processed)

            # Log processing metrics
            self._log_metrics(processed, metadata)

            return processed.tobytes()

        except Exception as e:
            logger.error("Audio processing failed: %s", str(e))
            raise RuntimeError(f"Audio processing failed: {str(e)}") from e

    def _validate_input(self, audio: np.ndarray) -> None:
        """Validate input audio data.

        Args:
            audio: Input audio array

        Raises:
            ValueError: If validation fails
        """
        if len(audio) == 0:
            raise ValueError("Empty audio data")

        duration = len(audio) / self.config.sample_rate
        if duration < self.config.min_duration:
            raise ValueError(f"Audio duration too short: {duration:.2f}s")
        if duration > self.config.max_duration:
            raise ValueError(f"Audio duration too long: {duration:.2f}s")

    def _apply_processing_pipeline(self, audio: np.ndarray) -> np.ndarray:
        """Apply audio processing pipeline.

        Args:
            audio: Input audio array

        Returns:
            Processed audio array
        """
        # Remove DC offset
        audio = self._remove_dc_offset(audio)

        # Apply noise gate
        audio = self._apply_noise_gate(audio)

        # Normalize amplitude
        audio = self._normalize_amplitude(audio)

        return audio.astype(np.int16)

    def _remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """Remove DC offset from audio.

        Args:
            audio: Input audio array

        Returns:
            Audio array with DC offset removed
        """
        self._dc_offset = np.mean(audio)
        return audio - self._dc_offset

    def _apply_noise_gate(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise gate to remove low-level noise.

        Args:
            audio: Input audio array

        Returns:
            Noise-gated audio array
        """
        rms = np.sqrt(np.mean(audio**2))
        threshold = self.config.noise_threshold * self.config.max_amplitude

        if rms < threshold:
            return np.zeros_like(audio)
        return audio

    def _normalize_amplitude(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio amplitude.

        Args:
            audio: Input audio array

        Returns:
            Normalized audio array
        """
        if np.max(np.abs(audio)) > 0:
            target = self.config.normalize_target * self.config.max_amplitude
            audio = audio * (target / np.max(np.abs(audio)))
        return audio

    def _update_state(self, audio: np.ndarray) -> None:
        """Update internal state with processed audio.

        Args:
            audio: Processed audio array
        """
        self._peak_amplitude = max(self._peak_amplitude, np.max(np.abs(audio)))
        self._processed_chunks += 1

    def _log_metrics(
        self, audio: np.ndarray, metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Log audio processing metrics.

        Args:
            audio: Processed audio array
            metadata: Optional processing metadata
        """
        metrics = {
            "peak_amplitude": float(np.max(np.abs(audio))),
            "rms": float(np.sqrt(np.mean(audio**2))),
            "dc_offset": float(self._dc_offset),
            "duration": len(audio) / self.config.sample_rate,
            "processed_chunks": self._processed_chunks,
        }
        if metadata:
            metrics.update(metadata)

        logger.debug("Audio processing metrics: %s", metrics)

    def save_wav(
        self, audio_data: bytes, filename: str, sample_rate: Optional[int] = None
    ) -> None:
        """Save audio data to WAV file.

        Args:
            audio_data: Audio bytes to save
            filename: Output filename
            sample_rate: Optional sample rate override

        Raises:
            IOError: If file cannot be written
            ValueError: If parameters are invalid
        """
        if not isinstance(sample_rate, (type(None), int)):
            raise ValueError("Sample rate must be an integer")

        try:
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)

            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(self.config.channels)
                wf.setsampwidth(self.config.sample_width)
                wf.setframerate(sample_rate or self.config.sample_rate)
                wf.writeframes(audio_data)

            logger.info("Saved audio to %s", filename)

        except (IOError, wave.Error) as e:
            logger.error("Failed to save WAV file: %s", str(e))
            raise

    def load_wav(self, filename: str) -> Tuple[bytes, int]:
        """Load audio data from WAV file.

        Args:
            filename: WAV file to load

        Returns:
            Tuple of (audio_data, sample_rate)

        Raises:
            FileNotFoundError: If file does not exist
            wave.Error: If file is invalid
        """
        try:
            with wave.open(filename, "rb") as wf:
                if wf.getnchannels() != self.config.channels:
                    logger.warning(
                        "Channel count mismatch: expected %d, got %d",
                        self.config.channels,
                        wf.getnchannels(),
                    )

                audio_data = wf.readframes(wf.getnframes())
                sample_rate = wf.getframerate()

                logger.info("Loaded audio from %s", filename)
                return audio_data, sample_rate

        except (FileNotFoundError, wave.Error) as e:
            logger.error("Failed to load WAV file: %s", str(e))
            raise

    async def cleanup(self) -> None:
        """Clean up resources."""
        self._reset_state()
        logger.debug("AudioProcessor cleaned up")

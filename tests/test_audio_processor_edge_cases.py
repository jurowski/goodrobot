"""Edge case tests for audio processing implementation."""

import os
import tempfile
import unittest
import wave

import numpy as np

from src.audio.audio_processor import AudioConfig, AudioProcessor


class TestAudioProcessorEdgeCases(unittest.TestCase):
    """Test edge cases for AudioProcessor class."""

    def setUp(self):
        """Set up test environment."""
        self.config = AudioConfig()
        self.processor = AudioProcessor(self.config)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        for f in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, f))
        os.rmdir(self.temp_dir)

    def test_empty_audio(self):
        """Test handling empty audio data."""
        with self.assertRaises(ValueError):
            self.processor.process(bytes())

    def test_invalid_audio_length(self):
        """Test audio with invalid byte length."""
        # Create audio bytes with invalid length
        audio_bytes = np.random.randint(-32768, 32767, 1000, dtype=np.int16).tobytes()[
            :-1
        ]

        with self.assertRaises(ValueError):
            self.processor.process(audio_bytes)

    def test_very_short_audio(self):
        """Test very short audio duration."""
        # Create audio shorter than min_duration
        samples = int(self.config.sample_rate * (self.config.min_duration / 2))
        audio = np.random.randint(-32768, 32767, samples, dtype=np.int16)

        with self.assertRaises(ValueError):
            self.processor.process(audio.tobytes())

    def test_very_long_audio(self):
        """Test very long audio duration."""
        # Create audio longer than max_duration
        samples = int(self.config.sample_rate * (self.config.max_duration * 2))
        audio = np.random.randint(-32768, 32767, samples, dtype=np.int16)

        with self.assertRaises(ValueError):
            self.processor.process(audio.tobytes())

    def test_silence(self):
        """Test processing silent audio."""
        # Create silent audio
        audio = np.zeros(16000, dtype=np.int16)
        processed = self.processor.process(audio.tobytes())

        # Verify silence is preserved
        processed_array = np.frombuffer(processed, dtype=np.int16)
        self.assertTrue(np.all(processed_array == 0))

    def test_maximum_amplitude(self):
        """Test processing maximum amplitude audio."""
        # Create maximum amplitude audio
        audio = np.ones(16000, dtype=np.int16) * 32767
        processed = self.processor.process(audio.tobytes())

        # Verify amplitude is preserved within limits
        processed_array = np.frombuffer(processed, dtype=np.int16)
        self.assertTrue(np.all(processed_array <= 32767))
        self.assertTrue(np.all(processed_array >= -32768))

    def test_invalid_sample_rate(self):
        """Test invalid sample rate configuration."""
        with self.assertRaises(ValueError):
            AudioProcessor(AudioConfig(sample_rate=0))

    def test_invalid_channels(self):
        """Test invalid channel configuration."""
        with self.assertRaises(ValueError):
            AudioProcessor(AudioConfig(channels=0))

    def test_invalid_sample_width(self):
        """Test invalid sample width configuration."""
        with self.assertRaises(ValueError):
            AudioProcessor(AudioConfig(sample_width=3))

    def test_save_wav_invalid_path(self):
        """Test saving WAV to invalid path."""
        audio = np.zeros(16000, dtype=np.int16)
        invalid_path = os.path.join(self.temp_dir, "nonexistent", "test.wav")

        with self.assertRaises(IOError):
            self.processor.save_wav(audio.tobytes(), invalid_path)

    def test_load_wav_nonexistent(self):
        """Test loading nonexistent WAV file."""
        nonexistent_file = os.path.join(self.temp_dir, "nonexistent.wav")

        with self.assertRaises(FileNotFoundError):
            self.processor.load_wav(nonexistent_file)

    def test_load_wav_invalid_format(self):
        """Test loading invalid WAV file."""
        invalid_file = os.path.join(self.temp_dir, "invalid.wav")
        with open(invalid_file, "wb") as f:
            f.write(b"not a wav file")

        with self.assertRaises(wave.Error):
            self.processor.load_wav(invalid_file)

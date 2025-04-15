"""Unit tests for audio processing implementation."""

import os
import tempfile
import unittest
import wave

import numpy as np

from src.audio.audio_processor import AudioConfig, AudioProcessor


class TestAudioProcessor(unittest.TestCase):
    """Test cases for AudioProcessor class."""

    def setUp(self):
        """Set up test environment."""
        self.config = AudioConfig(
            sample_rate=16000,
            channels=1,
            sample_width=2,
            chunk_size=1024,
            noise_threshold=0.1,
            normalize_target=0.9,
        )
        self.processor = AudioProcessor(self.config)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        for f in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, f))
        os.rmdir(self.temp_dir)

    def test_init(self):
        """Test processor initialization."""
        self.assertEqual(self.processor.config.sample_rate, 16000)
        self.assertEqual(self.processor.config.channels, 1)
        self.assertEqual(self.processor.config.sample_width, 2)

    def test_process_normal_audio(self):
        """Test processing normal audio data."""
        # Create test audio
        duration = 1.0  # seconds
        t = np.linspace(0, duration, int(self.config.sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 32767  # 440 Hz tone
        audio_data = audio.astype(np.int16).tobytes()

        # Process audio
        processed = self.processor.process(audio_data)

        # Verify processing
        processed_array = np.frombuffer(processed, dtype=np.int16)
        self.assertEqual(len(processed_array), len(audio))
        self.assertLess(np.max(np.abs(processed_array)), 32768)

    def test_dc_offset_removal(self):
        """Test DC offset removal."""
        # Create audio with DC offset
        audio = np.ones(16000, dtype=np.int16) * 1000
        processed = self.processor.process(audio.tobytes())

        # Verify DC offset removal
        processed_array = np.frombuffer(processed, dtype=np.int16)
        self.assertAlmostEqual(np.mean(processed_array), 0, delta=1)

    def test_noise_gate(self):
        """Test noise gate functionality."""
        # Create quiet noise
        noise = np.random.normal(0, 100, 16000).astype(np.int16)
        processed = self.processor.process(noise.tobytes())

        # Verify noise removal
        processed_array = np.frombuffer(processed, dtype=np.int16)
        self.assertTrue(np.all(processed_array == 0))

    def test_amplitude_normalization(self):
        """Test amplitude normalization."""
        # Create quiet audio
        audio = np.random.normal(0, 1000, 16000).astype(np.int16)
        processed = self.processor.process(audio.tobytes())

        # Verify normalization
        processed_array = np.frombuffer(processed, dtype=np.int16)
        max_amplitude = np.max(np.abs(processed_array))
        target = int(self.config.normalize_target * self.config.max_amplitude)
        self.assertAlmostEqual(max_amplitude, target, delta=1)

    def test_save_load_wav(self):
        """Test saving and loading WAV files."""
        # Create test audio
        audio = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
        audio_data = audio.tobytes()

        # Save and load
        filename = os.path.join(self.temp_dir, "test.wav")
        self.processor.save_wav(audio_data, filename)
        loaded_data, sample_rate = self.processor.load_wav(filename)

        # Verify
        self.assertEqual(sample_rate, self.config.sample_rate)
        np.testing.assert_array_equal(np.frombuffer(loaded_data, dtype=np.int16), audio)

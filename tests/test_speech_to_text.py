import os
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
import io
import soundfile as sf
import pyaudio
import openai
from src.voice_recognition.speech_to_text import SpeechToText
from src.voice_recognition.wake_word import WakeWordDetector
from src.settings import Settings
import psutil
import gc
import time
import asyncio


@pytest.fixture
def mock_env():
    """Mock environment variables."""
    os.environ["OPENAI_API_KEY"] = "test_key"
    os.environ["PICOVOICE_ACCESS_KEY"] = "test_key"
    yield
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("PICOVOICE_ACCESS_KEY", None)


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        openai_api_key="test_key",
        picovoice_access_key="test_key",
        data_dir="data",
        wake_word="jarvis",
        sensitivity=0.5,
        sample_rate=16000,
        channels=1,
        chunk_size=1024,
        sample_width=2,
        min_duration=0.1,
        max_duration=10.0,
        noise_threshold=0.1,
        max_amplitude=32767,
        normalize_target=0.9
    )


@pytest.fixture
def mock_audio():
    """Create mock audio data."""
    audio_data = np.zeros(16000, dtype=np.float32)  # 1 second of silence
    with io.BytesIO() as buffer:
        sf.write(buffer, audio_data, 16000, format='WAV')
        return buffer.getvalue()


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    mock_transcription = AsyncMock()
    mock_transcription.return_value = "test transcription"
    
    mock_audio = Mock()
    mock_audio.transcriptions = Mock()
    mock_audio.transcriptions.create = mock_transcription
    
    with patch('openai.OpenAI') as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.audio = mock_audio
        yield mock_instance


@pytest.fixture
def speech_to_text(settings, mock_openai_client):
    """Create SpeechToText instance with mocked OpenAI client."""
    return SpeechToText(settings)


@pytest.fixture
def wake_word_detector(settings):
    detector = WakeWordDetector(settings)
    detector._audio_callback = Mock()
    detector._audio_callback.return_value = (None, pyaudio.paContinue)
    return detector


@pytest.mark.asyncio
async def test_speech_to_text_initialization(speech_to_text):
    """Test SpeechToText initialization."""
    assert speech_to_text is not None
    assert speech_to_text.settings is not None
    assert speech_to_text.client is not None


@pytest.mark.asyncio
async def test_speech_to_text_initialization_missing_api_key():
    """Test SpeechToText initialization with missing API key."""
    settings = Settings(
        openai_api_key="",
        picovoice_access_key="test_key",
        data_dir="data"
    )
    with pytest.raises(ValueError, match="OpenAI API key is required"):
        SpeechToText(settings)


@pytest.mark.asyncio
async def test_transcribe_audio_success(speech_to_text, mock_audio):
    """Test successful audio transcription."""
    result = await speech_to_text.transcribe_audio(mock_audio)
    assert result == "test transcription"


@pytest.mark.asyncio
async def test_transcribe_audio_api_error(speech_to_text, mock_audio, mock_openai_client):
    """Test audio transcription with API error."""
    mock_openai_client.audio.transcriptions.create.side_effect = Exception("API Error")
    with pytest.raises(Exception, match="API Error"):
        await speech_to_text.transcribe_audio(mock_audio)


def get_process_memory():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


@pytest.mark.asyncio
async def test_memory_usage_with_large_chunks(speech_to_text, settings):
    """Test memory usage when processing large audio chunks."""
    # Force garbage collection before test
    gc.collect()
    initial_memory = get_process_memory()
    
    # Create large audio chunks (4MB each)
    large_chunks = [np.random.bytes(4 * 1024 * 1024) for _ in range(5)]
    
    for chunk in large_chunks:
        await speech_to_text.transcribe_audio(chunk)
    
    # Force garbage collection after test
    gc.collect()
    final_memory = get_process_memory()
    
    # Memory increase should not exceed 10MB
    assert final_memory - initial_memory < 10, "Memory usage increased significantly"


@pytest.mark.asyncio
async def test_memory_usage_with_long_stream(speech_to_text, settings):
    """Test memory usage during long streaming session."""
    gc.collect()
    initial_memory = get_process_memory()
    
    # Simulate 1000 small chunks (10KB each)
    small_chunks = [np.random.bytes(10 * 1024) for _ in range(1000)]
    
    for chunk in small_chunks:
        await speech_to_text.transcribe_audio(chunk)
    
    gc.collect()
    final_memory = get_process_memory()
    
    # Memory increase should not exceed 5MB for streaming
    assert final_memory - initial_memory < 5, "Memory leak detected in streaming"


@pytest.mark.asyncio
async def test_memory_usage_with_multiple_wake_words(wake_word_detector):
    """Test memory usage during multiple wake word detections."""
    gc.collect()
    initial_memory = get_process_memory()
    
    # Simulate 100 wake word detections
    for _ in range(100):
        wake_word_detector._audio_callback(np.random.bytes(1024), 1024, None, None)
        await asyncio.sleep(0.01)  # Small delay to simulate real-world usage
    
    gc.collect()
    final_memory = get_process_memory()
    
    # Memory increase should not exceed 2MB
    assert final_memory - initial_memory < 2, "Memory leak in wake word detection"


@pytest.mark.asyncio
async def test_memory_usage_with_error_handling(speech_to_text, mock_openai_client):
    """Test memory stability during error conditions."""
    gc.collect()
    initial_memory = get_process_memory()
    
    # Simulate errors during processing
    mock_openai_client.audio.transcriptions.create.side_effect = Exception("API Error")
    
    # Process multiple chunks with errors
    for _ in range(50):
        try:
            await speech_to_text.transcribe_audio(np.random.bytes(1024))
        except Exception:
            pass
    
    gc.collect()
    final_memory = get_process_memory()
    
    # Memory increase should not exceed 2MB even with errors
    assert final_memory - initial_memory < 2, "Memory leak during error handling"


@pytest.mark.asyncio
@pytest.mark.timeout(120)  # Extend timeout for this longer test
async def test_memory_leak_continuous_operation(speech_to_text, wake_word_detector, settings, mock_openai_client):
    """Test for memory leaks during continuous operation over several minutes."""
    gc.collect()
    initial_memory = get_process_memory()
    
    start_time = time.time()
    duration = 60  # Run for 60 seconds
    wake_word_detections = 0
    transcriptions = 0

    async def simulate_audio_stream():
        nonlocal wake_word_detections, transcriptions
        while time.time() - start_time < duration:
            # Simulate receiving a small audio chunk
            chunk_data = np.random.bytes(settings.chunk_size * settings.sample_width)
            
            # Simulate wake word processing (every ~50ms)
            if time.time() % 0.05 < 0.01: # Rough simulation
                 wake_word_detector._audio_callback(chunk_data, settings.chunk_size, None, None)
            
            # Simulate wake word detection occasionally (e.g., every 5 seconds)
            if time.time() % 5 < 0.05: 
                wake_word_detections += 1
                wake_word_detector.callback() # Simulate callback trigger
                # Simulate transcribing after wake word
                try:
                    await speech_to_text.transcribe_audio(chunk_data)
                    transcriptions += 1
                except Exception:
                    pass # Ignore transcription errors for this memory test
            
            await asyncio.sleep(0.01) # Small delay to prevent busy-waiting

    # Run the simulation
    await simulate_audio_stream()

    print(f"\nContinuous operation test completed: {wake_word_detections} wake words, {transcriptions} transcriptions over {duration}s")

    gc.collect()
    final_memory = get_process_memory()
    memory_increase = final_memory - initial_memory
    print(f"Initial Memory: {initial_memory:.2f} MB, Final Memory: {final_memory:.2f} MB, Increase: {memory_increase:.2f} MB")

    # Allow a slightly larger but still modest increase for continuous operation (e.g., < 10MB)
    assert memory_increase < 10, f"Potential memory leak detected during continuous operation: {memory_increase:.2f} MB increase"

import os
import sys
import subprocess
import soundfile as sf
import numpy as np
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.api.api import calculate_audio_metrics, validate_sample
from src.voice_recognition.wake_word import WakeWordDetector
from src.settings import Settings

def convert_m4a_to_wav(m4a_path, wav_path):
    """Convert m4a file to wav using ffmpeg"""
    try:
        subprocess.run([
            'ffmpeg',
            '-i', str(m4a_path),
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-f', 'wav',
            str(wav_path)
        ], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {m4a_path}: {e.stderr.decode()}")
        return False

def process_audio_file(wav_path, wake_word_detector):
    """Process a single audio file and return results"""
    try:
        # Read the audio file
        audio_array, sample_rate = sf.read(wav_path)
        
        # Convert to mono if stereo
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        # Normalize audio to [-1, 1] range
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        
        if np.abs(audio_array).max() > 0:
            audio_array = audio_array / np.abs(audio_array).max()
        
        # Calculate metrics before converting to int16
        metrics = calculate_audio_metrics(audio_array, sample_rate)
        
        # Convert to int16 for wake word detector
        audio_array_int16 = (audio_array * 32767).astype(np.int16)
        
        # Process with wake word detector
        detected = wake_word_detector.process_audio_chunk(audio_array_int16)
        metrics['wake_word'] = detected
        
        # Validate sample
        validation_result = validate_sample(audio_array_int16, sample_rate, metrics)
        
        return {
            'success': validation_result['is_valid'],
            'error': validation_result['error'],
            'metrics': metrics,
            'audio_array': audio_array_int16,
            'sample_rate': sample_rate
        }
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'metrics': None
        }

def main():
    # Initialize settings and wake word detector
    settings = Settings()
    wake_word_detector = WakeWordDetector(settings)
    
    # Create output directory
    output_dir = project_root / "tests" / "audio_samples" / "wake_word" / "human"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each file
    for i in range(1, 6):
        m4a_path = Path.home() / "Downloads" / f"jarvis-test-sj-{i}.m4a"
        wav_path = output_dir / f"jarvis-test-sj-{i}.wav"
        
        print(f"\nProcessing file {i}:")
        print(f"Input: {m4a_path}")
        print(f"Output: {wav_path}")
        
        # Convert m4a to wav
        if not convert_m4a_to_wav(m4a_path, wav_path):
            print("Conversion failed, skipping...")
            continue
        
        # Process the audio
        result = process_audio_file(wav_path, wake_word_detector)
        
        # Print results
        print("\nResults:")
        print(f"Success: {result['success']}")
        if not result['success']:
            print(f"Error: {result['error']}")
        
        if result['metrics']:
            print("\nMetrics:")
            for key, value in result['metrics'].items():
                print(f"{key}: {value}")
        
        # Save the file if successful
        if result['success']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_path = output_dir / f"jarvis_human_{timestamp}.wav"
            sf.write(final_path, result['audio_array'], result['sample_rate'])
            print(f"\nSaved as: {final_path}")

if __name__ == "__main__":
    main() 
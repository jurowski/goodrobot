import os
import time
import json
import requests
import sounddevice as sd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple
import librosa
from elevenlabs.client import ElevenLabs
from src.voice_recognition.wake_word import WakeWordDetector
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Simple settings class for testing"""
    def __init__(self):
        self.wake_word = "jarvis"
        self.sensitivity = 0.7
        self.sample_rate = 16000
        self.picovoice_access_key = os.getenv("PICOVOICE_ACCESS_KEY")

class SampleRecorder:
    def __init__(self):
        self.sample_rate = 16000
        self.duration = 2
        self.wake_word = "jarvis"
        
        # Initialize wake word detector with settings
        settings = Settings()
        self.wake_word_detector = WakeWordDetector(settings)
        
        # Initialize ElevenLabs client
        elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        if elevenlabs_api_key:
            self.eleven_labs = ElevenLabs(api_key=elevenlabs_api_key)
            self.elevenlabs_enabled = True
            print("ElevenLabs API key loaded successfully")
        else:
            self.elevenlabs_enabled = False
            print("Warning: ElevenLabs API key not found in environment")

    def record_sample(self) -> np.ndarray:
        """Record a single audio sample"""
        print(f"Recording for {self.duration} seconds...")
        recording = sd.rec(int(self.duration * self.sample_rate), 
                         samplerate=self.sample_rate, 
                         channels=1)
        sd.wait()
        return recording

    def play_sample(self, audio_data: np.ndarray):
        """Play back the recorded sample"""
        print("Playing back recording...")
        sd.play(audio_data, self.sample_rate)
        sd.wait()

    def save_sample(self, audio_data: np.ndarray, filename: str):
        """Save audio data to WAV file"""
        import soundfile as sf
        sf.write(filename, audio_data, self.sample_rate)
        print(f"Saved to {filename}")

    def check_wake_word(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """Check if the wake word was detected in the sample"""
        # Convert to int16 for wake word detector
        audio_data = (audio_data * 32767).astype(np.int16)
        detected = self.wake_word_detector.process_audio_chunk(audio_data)
        return detected, 0.0  # Return confidence of 0.0 since it's not provided

    def generate_ai_sample(self) -> np.ndarray:
        """Generate AI sample using ElevenLabs"""
        if not self.elevenlabs_enabled:
            raise ValueError("ElevenLabs API key not set")
        
        print("Generating AI sample...")
        audio = self.eleven_labs.text_to_speech.convert(
            text=f"Hey {self.wake_word}",
            voice_id="JBFqnCBsd6RMkjVDRZzb",  # Josh voice
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )
        
        # Convert audio bytes to numpy array
        import io
        audio_data, _ = librosa.load(io.BytesIO(audio), sr=self.sample_rate)
        return audio_data

    def record_human_samples(self, num_samples: int = 5):
        """Record human samples with quality checks"""
        print("\nRecording human samples...")
        for i in range(num_samples):
            print(f"\nSample {i+1}/{num_samples}")
            print(f"Say '{self.wake_word}' clearly...")
            
            while True:
                audio_data = self.record_sample()
                self.play_sample(audio_data)
                
                detected, confidence = self.check_wake_word(audio_data)
                if detected:
                    print(f"Wake word detected!")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join("tests/audio_samples/wake_word/human", 
                                         f"{self.wake_word}_human_{timestamp}.wav")
                    self.save_sample(audio_data, filename)
                    break
                else:
                    print("Wake word not detected. Please try again.")
                    time.sleep(1)
            
            time.sleep(1)  # Pause between recordings

    def record_ai_samples(self, num_samples: int = 5):
        """Record AI samples with quality checks"""
        if not self.elevenlabs_enabled:
            print("ElevenLabs API key not set. Skipping AI samples.")
            return

        print("\nRecording AI samples...")
        for i in range(num_samples):
            print(f"\nSample {i+1}/{num_samples}")
            
            while True:
                try:
                    audio_data = self.generate_ai_sample()
                    self.play_sample(audio_data)
                    
                    detected, confidence = self.check_wake_word(audio_data)
                    if detected:
                        print(f"Wake word detected!")
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = os.path.join("tests/audio_samples/wake_word/ai", 
                                             f"{self.wake_word}_ai_{timestamp}.wav")
                        self.save_sample(audio_data, filename)
                        break
                    else:
                        print("Wake word not detected in AI sample. Retrying...")
                except Exception as e:
                    print(f"Error generating AI sample: {e}")
                    time.sleep(1)
            
            time.sleep(1)  # Pause between recordings

def main():
    # Create directories if they don't exist
    os.makedirs("tests/audio_samples/wake_word/human", exist_ok=True)
    os.makedirs("tests/audio_samples/wake_word/ai", exist_ok=True)

    recorder = SampleRecorder()

    print("Wake Word Test Sample Recorder")
    print("==============================")
    print("1. Record human samples")
    print("2. Record AI samples")
    print("3. Record both")
    print("4. Exit")
    
    choice = input("Enter your choice (1-4): ")
    
    if choice in ["1", "3"]:
        recorder.record_human_samples()
    
    if choice in ["2", "3"]:
        recorder.record_ai_samples()
    
    print("\nRecording complete!")

if __name__ == "__main__":
    main() 
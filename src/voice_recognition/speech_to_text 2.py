"""Hybrid speech recognition system combining local and cloud-based processing."""

import logging
import speech_recognition as sr
from typing import Tuple, Dict, Any, Optional, List

from .whisper_transcriber import WhisperTranscriber
from config.settings import Settings

logger = logging.getLogger(__name__)

class HybridSpeechRecognizer:
    """Hybrid speech recognition combining local and cloud-based processing."""
    
    def __init__(
        self,
        settings: Settings,
        local_confidence_threshold: float = 0.7,
    ):
        """Initialize the hybrid speech recognizer.
        
        Args:
            settings: Application settings
            local_confidence_threshold: Confidence threshold for local recognition
        """
        self.settings = settings
        self.local_confidence_threshold = local_confidence_threshold
        
        # Initialize recognizers
        self.recognizer = sr.Recognizer()
        self.whisper = WhisperTranscriber(settings)
        
        # Adjust recognition settings
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 4000
        
    def record_audio(self, timeout: Optional[float] = None) -> bytes:
        """Record audio from microphone.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            Raw audio data as bytes
        """
        with sr.Microphone() as source:
            logger.info("Listening for speech...")
            self.recognizer.adjust_for_ambient_noise(source)
            
            try:
                audio = self.recognizer.listen(source, timeout=timeout)
                return audio.get_raw_data()
            except sr.WaitTimeoutError:
                logger.info("No speech detected within timeout")
                return b""
                
    def speech_to_text(self, audio_data: bytes) -> Tuple[str, Dict[str, Any]]:
        """Convert speech to text using hybrid approach.
        
        First attempts local recognition, falls back to Whisper if confidence is low.
        
        Args:
            audio_data: Raw audio data to transcribe
            
        Returns:
            Tuple containing:
            - transcribed text
            - metadata dictionary with confidence scores and other info
        """
        # Try local recognition first
        text, metadata = self._recognize_local(audio_data)
        
        # If local recognition failed or confidence is low, try Whisper
        if not text or metadata.get("confidence", 0) < self.local_confidence_threshold:
            logger.info("Local recognition failed or low confidence, trying Whisper...")
            text, cloud_metadata = self.whisper.transcribe_audio(audio_data)
            metadata.update({
                "used_cloud": True,
                "cloud_metadata": cloud_metadata
            })
        else:
            metadata["used_cloud"] = False
            
        return text, metadata
        
    def _recognize_local(self, audio_data: bytes) -> Tuple[str, Dict[str, Any]]:
        """Perform local speech recognition.
        
        Args:
            audio_data: Raw audio data to transcribe
            
        Returns:
            Tuple containing:
            - transcribed text
            - metadata dictionary with confidence scores
        """
        audio = sr.AudioData(audio_data, 16000, 2)
        
        try:
            result = self.recognizer.recognize_google(
                audio,
                show_all=True  # Get confidence scores
            )
            
            if result and isinstance(result, dict) and "alternative" in result:
                best_result = result["alternative"][0]
                return best_result["transcript"], {
                    "confidence": best_result.get("confidence", 0),
                    "alternatives": result["alternative"][1:]
                }
            
            return "", {"confidence": 0, "error": "No speech detected"}
            
        except sr.UnknownValueError:
            return "", {"confidence": 0, "error": "Speech not understood"}
        except sr.RequestError as e:
            return "", {"confidence": 0, "error": f"Recognition error: {str(e)}"}
            
    def cleanup(self):
        """Clean up resources."""
        self.whisper.cleanup()

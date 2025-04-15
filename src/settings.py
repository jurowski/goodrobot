import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Settings:
    """Configuration settings for the application."""
    openai_api_key: Optional[str] = field(default=None)
    picovoice_access_key: Optional[str] = field(default=None)
    data_dir: str = field(default="data")
    wake_word: str = field(default="jarvis")
    sensitivity: float = field(default=0.7)  # Increased sensitivity
    sample_rate: int = field(default=16000)
    channels: int = field(default=1)
    chunk_size: int = field(default=512)  # Reduced chunk size for better detection
    sample_width: int = field(default=2)
    min_duration: float = field(default=0.1)  # Minimum audio duration in seconds
    max_duration: float = field(default=10.0)  # Maximum audio duration in seconds
    noise_threshold: float = field(default=0.1)  # Noise gate threshold (0-1)
    max_amplitude: int = field(default=32767)  # Maximum value for 16-bit audio
    normalize_target: float = field(default=0.9)  # Target amplitude for normalization (0-1)
    debug_audio: bool = field(default=True)  # Enable audio debugging

    def __post_init__(self):
        """Initialize settings from environment variables if not provided."""
        self.openai_api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        self.picovoice_access_key = self.picovoice_access_key or os.getenv("PICOVOICE_ACCESS_KEY")
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True) 
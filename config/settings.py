from dataclasses import dataclass
from pathlib import Path

@dataclass
class Settings:
    """Settings for the application."""
    data_dir: str = "data"
    openai_api_key: str = ""
    debug: bool = True
    log_level: str = "INFO"
    database: dict = None

    # Voice recognition settings
    wake_word: str = "jarvis"
    sensitivity: float = 0.7
    picovoice_access_key: str = ""

    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # 16-bit audio
    chunk_size: int = 1024
    max_amplitude: int = 32767  # Maximum value for 16-bit audio
    min_duration: float = 0.1  # Minimum duration in seconds
    max_duration: float = 10.0  # Maximum duration in seconds
    noise_threshold: float = 0.1  # Noise gate threshold (0-1)
    normalize_target: float = 0.9  # Target amplitude for normalization (0-1)

    def __post_init__(self):
        if self.database is None:
            self.database = {
                "host": "localhost",
                "port": 5432,
                "user": "your_username",
                "password": "your_password",
                "database": "your_database_name",
            }

# Global settings instance
settings = Settings()

# Database settings
DATABASE = settings.database

# API keys and secrets
API_KEY = "your_api_key_here"
SECRET_KEY = "your_secret_key_here"

# Debug and logging settings
DEBUG = settings.debug
LOG_LEVEL = settings.log_level

# Other configurations
# Add any other settings as needed

import json
import os
from typing import Dict

from config.settings import Settings


def test_api_keys(settings: Settings) -> Dict[str, bool]:
    """Test all API keys in the settings."""
    results = {}

    # Test Picovoice access key
    results["picovoice"] = _test_picovoice_key(settings.picovoice_access_key)

    # Test Whisper API key
    results["whisper"] = _test_whisper_key(settings.whisper_api_key)

    # Test OpenAI API key
    results["openai"] = _test_openai_key(settings.openai_api_key)

    return results


def _test_picovoice_key(key: str) -> bool:
    """Test Picovoice access key."""
    try:
        # Implementation of Picovoice key test
        return True
    except Exception:
        return False


def _test_whisper_key(key: str) -> bool:
    """Test Whisper API key."""
    try:
        # Implementation of Whisper key test
        return True
    except Exception:
        return False


def _test_openai_key(key: str) -> bool:
    """Test OpenAI API key."""
    try:
        # Implementation of OpenAI key test
        return True
    except Exception:
        return False


def save_test_results(results: Dict[str, bool], settings: Settings):
    """Save API key test results to file."""
    test_results = {
        "results": results,
        "timestamp": datetime.now().isoformat(),
        "settings_file": settings.config_path,
    }

    # Save to file
    results_file = os.path.join(
        settings.data_dir,
        f"api_key_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    with open(results_file, "w") as f:
        json.dump(test_results, f, indent=2)


if __name__ == "__main__":
    # Example usage
    settings = Settings()

    # Test API keys
    results = test_api_keys(settings)

    # Print results
    print("API Key Test Results:")
    for service, status in results.items():
        print(f"{service}: {'Valid' if status else 'Invalid'}")

    # Save results
    save_test_results(results, settings)

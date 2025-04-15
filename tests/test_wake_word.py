"""Test suite for wake word detection functionality."""

import json
import os
import wave
from datetime import datetime
from typing import Any, Dict, List

from config.settings import Settings
from src.wake_word.porcupine_detector import WakeWordDetector


def test_wake_word_detection(
    settings: Settings, test_audio_files: List[str]
) -> Dict[str, Any]:
    """Test wake word detection on a set of audio files."""
    results = {
        "total_tests": len(test_audio_files),
        "successful_detections": 0,
        "failed_detections": 0,
        "test_results": [],
    }

    # Initialize wake word detector
    detector = WakeWordDetector(settings)

    try:
        for audio_file in test_audio_files:
            # Load audio file
            audio_data = _load_audio_file(audio_file)

            # Test detection
            detected = detector.detect(audio_data)

            # Record result
            result = {
                "file": audio_file,
                "detected": detected,
                "timestamp": datetime.now().isoformat(),
            }

            results["test_results"].append(result)

            if detected:
                results["successful_detections"] += 1
            else:
                results["failed_detections"] += 1

    finally:
        detector.cleanup()

    return results


def _load_audio_file(file_path: str) -> bytes:
    """Load audio data from a WAV file."""
    with wave.open(file_path, "rb") as wf:
        return wf.readframes(wf.getnframes())


def save_test_results(results: Dict[str, Any], settings: Settings):
    """Save wake word test results to file."""
    test_results = {
        "results": results,
        "timestamp": datetime.now().isoformat(),
        "settings_file": settings.config_path,
        "wake_word": settings.wake_word,
    }

    # Save to file
    results_file = os.path.join(
        settings.data_dir,
        f"wake_word_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    with open(results_file, "w") as f:
        json.dump(test_results, f, indent=2)


if __name__ == "__main__":
    # Example usage
    settings = Settings()

    # List of test audio files
    test_files = [
        "data/audio/wake_word_positive.wav",
        "data/audio/wake_word_negative.wav",
    ]

    # Test wake word detection
    results = test_wake_word_detection(settings, test_files)

    # Print results
    print("Wake Word Test Results:")
    print(f"Total tests: {results['total_tests']}")
    print(f"Successful detections: {results['successful_detections']}")
    print(f"Failed detections: {results['failed_detections']}")

    # Save results
    save_test_results(results, settings)

import unittest
from datetime import datetime

from config.settings import Settings
from src.notebook_llm.knowledge_manager import NotebookLLM
from src.prioritization.rl_model import PrioritizationEngine, Task
from src.speech_to_text.whisper_model import WhisperModel
from src.wake_word.porcupine_detector import WakeWordDetector


class TestWakeWordDetection(unittest.TestCase):
    """Test wake word detection functionality."""

    def setUp(self):
        """Set up test environment."""
        self.settings = Settings()
        self.detector = WakeWordDetector(self.settings)

    def tearDown(self):
        """Clean up test environment."""
        self.detector.cleanup()

    def test_wake_word_detection(self):
        """Test wake word detection on sample audio."""
        # Load sample audio data
        audio_data = self._load_sample_audio()

        # Test detection
        detected = self.detector.detect(audio_data)

        # Assert result
        self.assertTrue(detected, "Wake word should be detected")

    def _load_sample_audio(self) -> bytes:
        """Load sample audio data."""
        # Implementation of loading sample audio
        return b"\x00\x01\x02\x03" * 1000


class TestSpeechToText(unittest.TestCase):
    """Test speech-to-text conversion functionality."""

    def setUp(self):
        """Set up test environment."""
        self.settings = Settings()
        self.stt = WhisperModel(self.settings)

    def tearDown(self):
        """Clean up test environment."""
        self.stt.cleanup()

    def test_audio_transcription(self):
        """Test audio transcription."""
        # Load sample audio data
        audio_data = self._load_sample_audio()

        # Test transcription
        transcription = self.stt.transcribe(audio_data)

        # Assert result
        self.assertIsInstance(transcription, str)
        self.assertTrue(len(transcription) > 0)

    def _load_sample_audio(self) -> bytes:
        """Load sample audio data."""
        # Implementation of loading sample audio
        return b"\x00\x01\x02\x03" * 1000


class TestNotebookLLM(unittest.TestCase):
    """Test notebook LLM functionality."""

    def setUp(self):
        """Set up test environment."""
        self.settings = Settings()
        self.notebook = NotebookLLM(self.settings)

    def tearDown(self):
        """Clean up test environment."""
        self.notebook.cleanup()

    def test_task_management(self):
        """Test task management functionality."""
        # Add a task
        task = self.notebook.add_task(
            title="Test task", description="This is a test task", priority=1
        )

        # Get tasks
        tasks = self.notebook.get_tasks()

        # Assert result
        self.assertIn(task, tasks)

    def test_reminder_management(self):
        """Test reminder management functionality."""
        # Add a reminder
        reminder = self.notebook.add_reminder(
            title="Test reminder", reminder_time=datetime.now().isoformat()
        )

        # Get reminders
        reminders = self.notebook.get_reminders()

        # Assert result
        self.assertIn(reminder, reminders)


class TestPrioritizationEngine(unittest.TestCase):
    """Test prioritization engine functionality."""

    def setUp(self):
        """Set up test environment."""
        self.settings = Settings()
        self.engine = PrioritizationEngine(self.settings)

    def tearDown(self):
        """Clean up test environment."""
        self.engine.cleanup()

    def test_task_prioritization(self):
        """Test task prioritization."""
        # Create sample tasks
        tasks = [
            Task(
                title="High priority task",
                description="This is a high priority task",
                priority=1,
            ),
            Task(
                title="Low priority task",
                description="This is a low priority task",
                priority=1,
            ),
        ]

        # Prioritize tasks
        prioritized_tasks = self.engine.prioritize_tasks(tasks)

        # Assert result
        self.assertEqual(len(prioritized_tasks), 2)
        self.assertTrue(prioritized_tasks[0].priority >= prioritized_tasks[1].priority)


if __name__ == "__main__":
    unittest.main()

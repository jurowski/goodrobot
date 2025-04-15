# src/interface/voice_interface.py

"""Voice interface module for handling user interactions."""  # Add module docstring at top

import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from src.notebook_llm.knowledge_manager import NotebookLLM
from src.prioritization.rl_model import PrioritizationEngine, Task
from src.voice_recognition.speech_to_text import HybridSpeechRecognizer
from src.voice_recognition.wake_word import WakeWordDetector

logger = logging.getLogger(__name__)


class VoiceInterface:
    """
    Voice interface for the AI assistant.
    Handles wake word detection, speech recognition, and response generation.
    """

    def __init__(
        self,
        wake_word: str = "jarvis",
        wake_word_sensitivity: float = 0.7,
        picovoice_access_key: Optional[str] = None,
        whisper_api_key: Optional[str] = None,
        tts_engine: str = "local",  # "local" or "cloud"
        data_dir: str = "data",
    ):
        """
        Initialize voice interface.

        Args:
            wake_word: Wake word to listen for
            wake_word_sensitivity: Wake word detection sensitivity
            picovoice_access_key: Picovoice access key for wake word detection
            whisper_api_key: OpenAI API key for Whisper
            tts_engine: Text-to-speech engine to use
            data_dir: Data directory
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Get API keys from environment if not provided
        self.picovoice_access_key = picovoice_access_key or os.getenv(
            "PICOVOICE_ACCESS_KEY"
        )
        self.whisper_api_key = whisper_api_key or os.getenv("OPENAI_API_KEY")

        # Initialize wake word detector
        self.wake_word_detector = WakeWordDetector(
            access_key=self.picovoice_access_key,
            wake_word=wake_word,
            sensitivity=wake_word_sensitivity,
            callback=self._on_wake_word,
        )

        # Initialize speech recognizer
        self.speech_recognizer = HybridSpeechRecognizer(
            local_confidence_threshold=0.7, whisper_api_key=self.whisper_api_key
        )

        # Initialize knowledge manager
        self.notebook_llm = NotebookLLM(data_dir=os.path.join(data_dir, "notebook_llm"))

        # Initialize prioritization engine
        self.prioritization_engine = PrioritizationEngine(
            data_dir=os.path.join(data_dir, "prioritization")
        )

        # Initialize TTS engine
        self.tts_engine = tts_engine
        self._initialize_tts()

        # State management
        self.current_context = {
            "location": "unknown",
            "device": "default",
            "user_state": {
                "energy_level": 0.5,
                "stress_level": 0.5,
                "focus_level": 0.5,
                "mood": "neutral",
            },
            "time_features": self._get_time_features(),
        }

        self.session_history = []
        self.is_listening = False
        self.is_processing = False
        self.is_speaking = False

    def _initialize_tts(self):
        """Initialize text-to-speech engine."""
        if self.tts_engine == "local":
            try:
                import pyttsx3

                self.tts = pyttsx3.init()
                logger.info("Initialized local TTS engine")
            except ImportError:
                logger.warning("pyttsx3 not available, falling back to print-only mode")
                self.tts = None
        else:
            logger.info(
                "Cloud TTS not implemented yet, falling back to print-only mode"
            )
            self.tts = None

    def _get_time_features(self) -> Dict[str, Any]:
        """Extract time-related features."""
        now = datetime.now()

        return {
            "hour": now.hour,
            "minute": now.minute,
            "day_of_week": now.weekday(),
            "day_of_month": now.day,
            "month": now.month,
            "year": now.year,
            "is_weekend": now.weekday() >= 5,
            "is_morning": 5 <= now.hour < 12,
            "is_afternoon": 12 <= now.hour < 17,
            "is_evening": 17 <= now.hour < 22,
            "is_night": now.hour >= 22 or now.hour < 5,
            "today": now.strftime("%Y-%m-%d"),
        }

    def start(self):
        """Start the voice interface."""
        logger.info("Starting voice interface...")

        # Start wake word detection
        if not self.wake_word_detector.start():
            logger.error("Failed to start wake word detector")
            return False

        self.is_listening = True
        logger.info("Voice interface started, listening for wake word")

        # Run startup message
        self._speak("Voice assistant initialized and ready.")

        return True

    def stop(self):
        """Stop the voice interface."""
        logger.info("Stopping voice interface...")

        # Stop wake word detection
        self.wake_word_detector.stop()
        self.is_listening = False

        # Clean up resources
        self.wake_word_detector.cleanup()
        self.speech_recognizer.cleanup()

        logger.info("Voice interface stopped")

    def _on_wake_word(self):
        """Handle wake word detection."""
        # Avoid processing if already processing a command
        if self.is_processing:
            logger.info("Already processing a command, ignoring wake word")
            return

        self.is_processing = True

        try:
            # Acknowledge wake word (optional)
            self._acknowledge_wake_word()

            # Process voice command
            self._process_voice_command()
        finally:
            self.is_processing = False

    def _acknowledge_wake_word(self):
        """Acknowledge wake word detection with a short sound or visual cue."""
        # This could be a light, sound, or other feedback
        logger.info("Wake word detected!")

    def _process_voice_command(self):
        """Process voice command after wake word detection."""
        try:
            # Record and convert speech to text
            logger.info("Listening for command...")
            audio_data = self.speech_recognizer.record_audio()
            text, metadata = self.speech_recognizer.speech_to_text(audio_data)

            if not text:
                logger.info("No speech detected or recognition failed")
                self._speak("I didn't catch that. Can you please try again?")
                return

            logger.info(
                f"Recognized: '{text}' (confidence: {metadata.get('confidence', 0):.2f})"
            )

            # Update current context with time features
            self.current_context["time_features"] = self._get_time_features()

            # Process the command
            self._handle_command(text)

        except Exception as e:
            logger.error(f"Error processing voice command: {e}")
            self._speak("Sorry, I encountered an error processing your request.")

    def _handle_command(self, text: str):
        """
        Handle a voice command.

        Args:
            text: Recognized text from speech
        """
        # Extract intent and entities
        intent, entities = self._extract_intent(text)

        # Update session history
        self.session_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "text": text,
                "intent": intent,
                "entities": entities,
            }
        )

        # Store in knowledge repository
        self.notebook_llm.add_knowledge(
            text=text, source="conversation", importance=0.5, entities=entities
        )

        # Process based on intent
        if intent["type"] == "query":
            self._handle_query(text, intent, entities)
        elif intent["type"] == "command":
            self._handle_action_command(text, intent, entities)
        elif intent["type"] == "information":
            self._handle_information(text, intent, entities)
        else:
            # Default fallback response
            self._speak("I'm not sure how to help with that yet.")

    def _extract_intent(self, text: str) -> tuple:
        """
        Extract intent and entities from text.

        Args:
            text: Input text

        Returns:
            Tuple of (intent, entities)
        """
        # This is a simple rule-based implementation
        # In a production system, use a proper NLU model

        text_lower = text.lower()

        # Command patterns
        if text_lower.startswith(("add ", "create ", "schedule ", "remind me")):
            if "task" in text_lower or "to do" in text_lower or "to-do" in text_lower:
                return {
                    "type": "command",
                    "action": "add_task",
                }, self._extract_task_entities(text)
            elif "reminder" in text_lower or "remind me" in text_lower:
                return {
                    "type": "command",
                    "action": "add_reminder",
                }, self._extract_reminder_entities(text)
            elif (
                "event" in text_lower
                or "appointment" in text_lower
                or "schedule" in text_lower
            ):
                return {
                    "type": "command",
                    "action": "add_event",
                }, self._extract_event_entities(text)

        # Query patterns
        if text_lower.startswith(
            ("what", "when", "where", "who", "how", "why", "tell me", "show me")
        ):
            if "task" in text_lower or "to do" in text_lower or "to-do" in text_lower:
                return {"type": "query", "subject": "tasks"}, []
            elif "reminder" in text_lower:
                return {"type": "query", "subject": "reminders"}, []
            elif (
                "event" in text_lower
                or "appointment" in text_lower
                or "schedule" in text_lower
            ):
                return {"type": "query", "subject": "events"}, []
            elif (
                "suggest" in text_lower
                or "recommendation" in text_lower
                or "should i" in text_lower
            ):
                return {"type": "query", "subject": "suggestion"}, []
            else:
                return {"type": "query", "subject": "general"}, []

        # Information patterns
        if "i am" in text_lower or "i'm" in text_lower or "i feel" in text_lower:
            return {
                "type": "information",
                "subject": "user_state",
            }, self._extract_user_state_entities(text)

        # Default: treat as general information
        return {"type": "information", "subject": "general"}, []

    def _extract_task_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract task-related entities from text."""
        entities = []

        # Extract task title/description
        # This is a simple implementation - use NER in production
        if "to" in text.lower():
            parts = text.split("to", 1)
            if len(parts) > 1:
                task_description = parts[1].strip()
                if task_description:
                    entities.append(
                        {
                            "type": "task_description",
                            "value": task_description,
                            "text": task_description,
                        }
                    )

        # Extract due date if mentioned
        date_keywords = [
            "tomorrow",
            "today",
            "next week",
            "on monday",
            "on tuesday",
            "on wednesday",
            "on thursday",
            "on friday",
            "on saturday",
            "on sunday",
        ]

        for keyword in date_keywords:
            if keyword in text.lower():
                entities.append({"type": "due_date", "value": keyword, "text": keyword})
                break

        return entities

    def _extract_reminder_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract reminder-related entities from text."""
        entities = []

        # Extract reminder content
        if "to" in text.lower():
            parts = text.split("to", 1)
            if len(parts) > 1:
                reminder_text = parts[1].strip()
                if reminder_text:
                    entities.append(
                        {
                            "type": "reminder_text",
                            "value": reminder_text,
                            "text": reminder_text,
                        }
                    )

        # Extract time if mentioned
        time_keywords = [
            "today",
            "tomorrow",
            "in an hour",
            "in 10 minutes",
            "this evening",
            "tonight",
            "this afternoon",
        ]

        for keyword in time_keywords:
            if keyword in text.lower():
                entities.append(
                    {"type": "reminder_time", "value": keyword, "text": keyword}
                )
                break

        return entities

    def _extract_event_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract event-related entities from text."""
        # Similar to tasks and reminders
        return []

    def _extract_user_state_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract user state information from text."""
        entities = []

        # Extract energy level
        energy_keywords = {
            "tired": 0.2,
            "exhausted": 0.1,
            "low energy": 0.3,
            "energetic": 0.8,
            "full of energy": 0.9,
            "rested": 0.7,
        }

        for keyword, value in energy_keywords.items():
            if keyword in text.lower():
                entities.append(
                    {"type": "energy_level", "value": value, "text": keyword}
                )
                break

        # Extract stress level
        stress_keywords = {
            "stressed": 0.8,
            "anxious": 0.7,
            "overwhelmed": 0.9,
            "calm": 0.2,
            "relaxed": 0.1,
            "comfortable": 0.3,
        }

        for keyword, value in stress_keywords.items():
            if keyword in text.lower():
                entities.append(
                    {"type": "stress_level", "value": value, "text": keyword}
                )
                break

        # Extract mood
        mood_keywords = {
            "happy": "positive",
            "good": "positive",
            "great": "positive",
            "excellent": "positive",
            "sad": "negative",
            "upset": "negative",
            "down": "negative",
            "depressed": "negative",
            "okay": "neutral",
            "fine": "neutral",
            "alright": "neutral",
        }

        for keyword, value in mood_keywords.items():
            if keyword in text.lower():
                entities.append({"type": "mood", "value": value, "text": keyword})
                break

        return entities

    def _handle_query(
        self, text: str, intent: Dict[str, Any], entities: List[Dict[str, Any]]
    ):
        """
        Handle a query intent.

        Args:
            text: Original text
            intent: Extracted intent
            entities: Extracted entities
        """
        subject = intent.get("subject", "general")

        if subject == "tasks":
            self._handle_task_query()
        elif subject == "reminders":
            self._handle_reminder_query()
        elif subject == "events":
            self._handle_event_query()
        elif subject == "suggestion":
            self._provide_suggestion()
        else:
            self._speak("I'm not sure how to answer that question yet.")

    def _handle_task_query(self):
        """Handle query about tasks."""
        # Retrieve tasks from knowledge repository
        knowledge = self.notebook_llm.retrieve_relevant(
            query="tasks", include_tasks=True, include_reminders=False
        )

        tasks = knowledge.get("tasks", [])

        if not tasks:
            self._speak("You don't have any tasks at the moment.")
            return

        # Sort by urgency and importance
        sorted_tasks = sorted(
            tasks,
            key=lambda t: (t.get("urgency", 0) + t.get("importance", 0)),
            reverse=True,
        )

        # Construct response
        if len(sorted_tasks) == 1:
            task = sorted_tasks[0]
            response = f"You have one task: {task.get('title', 'Untitled task')}."
        else:
            top_tasks = sorted_tasks[:3]  # Limit to top 3
            task_list = ", ".join(t.get("title", "Untitled task") for t in top_tasks)
            response = f"You have {len(sorted_tasks)} tasks. The most important ones are: {task_list}."

        self._speak(response)

    def _handle_reminder_query(self):
        """Handle query about reminders."""
        # Similar to task query
        knowledge = self.notebook_llm.retrieve_relevant(
            query="reminders", include_tasks=False, include_reminders=True
        )

        reminders = knowledge.get("reminders", [])

        if not reminders:
            self._speak("You don't have any reminders at the moment.")
            return

        # Sort by reminder time
        now = datetime.now().isoformat()
        upcoming_reminders = [r for r in reminders if r.get("reminder_time", "") >= now]

        if not upcoming_reminders:
            self._speak("You don't have any upcoming reminders.")
            return

        sorted_reminders = sorted(
            upcoming_reminders, key=lambda r: r.get("reminder_time", "")
        )

        # Construct response
        if len(sorted_reminders) == 1:
            reminder = sorted_reminders[0]
            response = (
                f"You have one reminder: {reminder.get('title', 'Untitled reminder')}."
            )
        else:
            top_reminders = sorted_reminders[:3]  # Limit to top 3
            reminder_list = ", ".join(
                r.get("title", "Untitled reminder") for r in top_reminders
            )
            response = f"You have {len(sorted_reminders)} reminders. The next ones are: {reminder_list}."

        self._speak(response)

    def _handle_event_query(self):
        """Handle query about events."""
        # Similar to task and reminder queries
        self._speak("You don't have any upcoming events in your calendar.")

    def _handle_action_command(
        self, text: str, intent: Dict[str, Any], entities: List[Dict[str, Any]]
    ):
        """
        Handle an action command intent.

        Args:
            text: Original text
            intent: Extracted intent
            entities: Extracted entities
        """
        action = intent.get("action", "")

        if action == "add_task":
            self._handle_add_task(text, entities)
        elif action == "add_reminder":
            self._handle_add_reminder(text, entities)
        elif action == "add_event":
            self._handle_add_event(text, entities)
        else:
            self._speak("I'm not sure how to perform that action yet.")

    def _handle_add_task(self, text: str, entities: List[Dict[str, Any]]):
        """Add a new task based on voice command."""
        # Extract task description
        task_description = ""
        for entity in entities:
            if entity["type"] == "task_description":
                task_description = entity["value"]
                break

        if not task_description:
            self._speak(
                "I didn't catch what task you want to add. Can you please try again?"
            )
            return

        # Extract due date if available
        due_date = None
        for entity in entities:
            if entity["type"] == "due_date":
                # Convert relative date to actual date
                due_date = self._parse_relative_date(entity["value"])
                break

        # Create task
        task_id = self.notebook_llm.add_task(
            title=task_description,
            description=None,
            due_date=due_date,
            priority=3,  # Medium priority
            status="pending",
        )

        # Confirm to user
        if due_date:
            self._speak(
                f"I've added the task: {task_description}, due {entity['value']}."
            )
        else:
            self._speak(f"I've added the task: {task_description}.")

    def _handle_add_reminder(self, text: str, entities: List[Dict[str, Any]]):
        """Add a new reminder based on voice command."""
        # Extract reminder text
        reminder_text = ""
        for entity in entities:
            if entity["type"] == "reminder_text":
                reminder_text = entity["value"]
                break

        if not reminder_text:
            self._speak(
                "I didn't catch what you want to be reminded about. Can you please try again?"
            )
            return

        # Extract reminder time if available
        reminder_time = None
        time_text = None
        for entity in entities:
            if entity["type"] == "reminder_time":
                time_text = entity["value"]
                # Convert relative time to actual time
                reminder_time = self._parse_relative_time(entity["value"])
                break

        # If no time specified, default to 1 hour from now
        if not reminder_time:
            reminder_time = (datetime.now() + timedelta(hours=1)).isoformat()
            time_text = "in one hour"

        # Create reminder
        reminder_id = self.notebook_llm.add_reminder(
            title=reminder_text, reminder_time=reminder_time, description=None
        )

        # Confirm to user
        self._speak(f"I'll remind you to {reminder_text} {time_text}.")

    def _handle_add_event(self, text: str, entities: List[Dict[str, Any]]):
        """Add a new event based on voice command."""
        # This would be implemented similar to tasks and reminders
        self._speak("I've added the event to your calendar.")

    def _handle_information(
        self, text: str, intent: Dict[str, Any], entities: List[Dict[str, Any]]
    ):
        """
        Handle information sharing intent.

        Args:
            text: Original text
            intent: Extracted intent
            entities: Extracted entities
        """
        subject = intent.get("subject", "general")

        if subject == "user_state":
            self._update_user_state(entities)
        else:
            # Store as general knowledge
            self._speak("Thanks for sharing that information.")

    def _update_user_state(self, entities: List[Dict[str, Any]]):
        """Update user state based on shared information."""
        updated = False

        for entity in entities:
            if entity["type"] in ["energy_level", "stress_level"]:
                self.current_context["user_state"][entity["type"]] = entity["value"]
                updated = True
            elif entity["type"] == "mood":
                self.current_context["user_state"]["mood"] = entity["value"]
                updated = True

        if updated:
            self._speak("I've updated your status. Thanks for letting me know.")
        else:
            self._speak("I understand. Is there anything else you'd like to share?")

    def _provide_suggestion(self):
        """Provide a suggestion based on current context."""
        # Retrieve relevant knowledge
        knowledge = self.notebook_llm.retrieve_relevant(
            query="next task", include_tasks=True, include_reminders=True
        )

        # Generate suggestion using prioritization engine
        task = self.prioritization_engine.generate_suggestion(
            self.current_context, knowledge, self.session_history
        )

        if not task:
            self._speak("I don't have any specific suggestions at the moment.")
            return

        # Provide suggestion
        response = f"I suggest you work on: {task.title}."

        # Add due date info if available
        if task.due_date:
            due_date = datetime.fromisoformat(task.due_date)
            now = datetime.now()

            days_until_due = (due_date - now).days

            if days_until_due <= 0:
                response += " This is due today."
            elif days_until_due == 1:
                response += " This is due tomorrow."
            elif days_until_due < 7:
                response += f" This is due in {days_until_due} days."

        # Add importance info
        if task.importance > 0.8:
            response += " This is a high importance task."
        elif task.importance < 0.3:
            response += (
                " This is a low importance task that shouldn't take much effort."
            )

        self._speak(response)

        # Record suggestion in history
        self.session_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "suggestion": {"task_id": task.id, "title": task.title},
            }
        )

    def _parse_relative_date(self, text: str) -> Optional[str]:
        """Convert relative date expression to ISO format date string."""
        now = datetime.now()
        text_lower = text.lower()

        if "today" in text_lower:
            return now.strftime("%Y-%m-%d")
        elif "tomorrow" in text_lower:
            return (now + timedelta(days=1)).strftime("%Y-%m-%d")
        elif "next week" in text_lower:
            return (now + timedelta(days=7)).strftime("%Y-%m-%d")
        elif "monday" in text_lower:
            days_ahead = 7 - now.weekday()
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
            return (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        elif "tuesday" in text_lower:
            days_ahead = 1 - now.weekday()
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
            return (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        # Add more day cases as needed

        return None

    def _parse_relative_time(self, text: str) -> Optional[str]:
        """Convert relative time expression to ISO format datetime string."""
        now = datetime.now()
        text_lower = text.lower()

        if "now" in text_lower:
            return now.isoformat()
        elif "in an hour" in text_lower or "in 1 hour" in text_lower:
            return (now + timedelta(hours=1)).isoformat()
        elif "in 10 minutes" in text_lower:
            return (now + timedelta(minutes=10)).isoformat()
        elif "this evening" in text_lower or "tonight" in text_lower:
            evening = now.replace(hour=18, minute=0, second=0, microsecond=0)
            if now >= evening:
                evening = evening + timedelta(days=1)
            return evening.isoformat()
        elif "this afternoon" in text_lower:
            afternoon = now.replace(hour=14, minute=0, second=0, microsecond=0)
            if now >= afternoon:
                afternoon = afternoon + timedelta(days=1)
            return afternoon.isoformat()
        elif "tomorrow" in text_lower:
            tomorrow = now + timedelta(days=1)
            return tomorrow.replace(
                hour=9, minute=0, second=0, microsecond=0
            ).isoformat()

        return None

    def _speak(self, text: str):
        """
        Speak text to the user.

        Args:
            text: Text to speak
        """
        logger.info(f"Speaking: {text}")

        # Print text (always)
        print(f"Assistant: {text}")

        # Use TTS engine if available
        if self.tts:
            self.is_speaking = True

            try:
                self.tts.say(text)
                self.tts.runAndWait()
            except Exception as e:
                logger.error(f"TTS error: {e}")
            finally:
                self.is_speaking = False

    def process_command(self, text: str):
        """
        Process a text command directly (without voice).

        Args:
            text: Command text

        Returns:
            Response text
        """
        # Record response text
        response_text = None

        # Override speak method temporarily
        original_speak = self._speak

        def capture_speak(text):
            nonlocal response_text
            response_text = text
            original_speak(text)

        self._speak = capture_speak

        try:
            # Process the command
            self._handle_command(text)
            return response_text
        finally:
            # Restore original speak method
            self._speak = original_speak

    def get_current_context(self) -> Dict[str, Any]:
        """Get current context."""
        return self.current_context.copy()

    def update_context(self, updates: Dict[str, Any]):
        """
        Update current context.

        Args:
            updates: Dictionary of updates
        """
        # Update top-level keys
        for key, value in updates.items():
            if key == "user_state" and isinstance(value, dict):
                # Special handling for user_state to do partial update
                self.current_context["user_state"].update(value)
            else:
                self.current_context[key] = value

        logger.info(f"Updated context: {updates}")

    def get_session_history(self) -> List[Dict[str, Any]]:
        """Get session history."""
        return self.session_history.copy()


# Update __init__.py
# src/interface/__init__.py

from .voice_interface import VoiceInterface

__all__ = ["VoiceInterface"]

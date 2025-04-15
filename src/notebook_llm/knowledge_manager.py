import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from config.settings import Settings
from src.vector_db.vector_database import VectorDatabase


class NotebookLLM:
    """Knowledge management system for the voice AI assistant."""

    def __init__(self, settings: Settings):
        """Initialize the NotebookLLM with settings."""
        self.settings = settings
        self.vector_db = VectorDatabase(settings)
        self._ensure_data_directories()

    def _ensure_data_directories(self):
        """Ensure all required data directories exist."""
        directories = [
            self.settings.data_dir,
            os.path.join(self.settings.data_dir, "tasks"),
            os.path.join(self.settings.data_dir, "reminders"),
            os.path.join(self.settings.data_dir, "preferences"),
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def add_task(
        self,
        title: str,
        description: str,
        priority: int = 1,
        due_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add a new task to the system."""
        task = {
            "title": title,
            "description": description,
            "priority": priority,
            "due_date": due_date,
            "created_at": datetime.now().isoformat(),
            "completed": False,
        }

        # Save task to file
        task_file = os.path.join(
            self.settings.data_dir,
            "tasks",
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

        with open(task_file, "w") as f:
            json.dump(task, f, indent=2)

        # Add to vector database for semantic search
        self.vector_db.add_document(
            text=f"{title} {description}", metadata={"type": "task", "file": task_file}
        )

        return task

    def add_reminder(
        self, title: str, reminder_time: str, description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add a new reminder to the system."""
        reminder = {
            "title": title,
            "description": description,
            "reminder_time": reminder_time,
            "created_at": datetime.now().isoformat(),
            "completed": False,
        }

        # Save reminder to file
        reminder_file = os.path.join(
            self.settings.data_dir,
            "reminders",
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

        with open(reminder_file, "w") as f:
            json.dump(reminder, f, indent=2)

        return reminder

    def update_preferences(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Update user preferences."""
        pref_file = os.path.join(self.settings.data_dir, "preferences.json")

        # Load existing preferences if they exist
        if os.path.exists(pref_file):
            with open(pref_file, "r") as f:
                current_prefs = json.load(f)
        else:
            current_prefs = {}

        # Update preferences
        current_prefs.update(preferences)

        # Save updated preferences
        with open(pref_file, "w") as f:
            json.dump(current_prefs, f, indent=2)

        return current_prefs

    def get_tasks(
        self, completed: Optional[bool] = None, priority: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get tasks matching the given criteria."""
        tasks = []
        tasks_dir = os.path.join(self.settings.data_dir, "tasks")

        for task_file in os.listdir(tasks_dir):
            if not task_file.endswith(".json"):
                continue

            with open(os.path.join(tasks_dir, task_file), "r") as f:
                task = json.load(f)

            if completed is not None and task["completed"] != completed:
                continue

            if priority is not None and task["priority"] != priority:
                continue

            tasks.append(task)

        return tasks

    def get_reminders(self, completed: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Get reminders matching the given criteria."""
        reminders = []
        reminders_dir = os.path.join(self.settings.data_dir, "reminders")

        for reminder_file in os.listdir(reminders_dir):
            if not reminder_file.endswith(".json"):
                continue

            with open(os.path.join(reminders_dir, reminder_file), "r") as f:
                reminder = json.load(f)

            if completed is not None and reminder["completed"] != completed:
                continue

            reminders.append(reminder)

        return reminders

    def search_knowledge(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search the knowledge base using semantic search."""
        return self.vector_db.search(query, limit=limit)

    def cleanup(self):
        """Clean up resources."""
        self.vector_db.cleanup()


if __name__ == "__main__":
    # Example usage
    settings = Settings()
    notebook = NotebookLLM(settings)

    # Add a sample task
    task = notebook.add_task(
        title="Complete project documentation",
        description="Write detailed documentation for the voice AI assistant project",
        priority=2,
        due_date=(datetime.now() + timedelta(days=7)).isoformat(),
    )

    # Add a sample reminder
    reminder = notebook.add_reminder(
        title="Team meeting",
        reminder_time=(datetime.now() + timedelta(hours=1)).isoformat(),
        description="Weekly team sync meeting",
    )

    # Update preferences
    prefs = notebook.update_preferences(
        {
            "important_categories": ["work", "health"],
            "notification_preferences": {"email": True, "push": True, "sound": True},
        }
    )

    # Search knowledge
    results = notebook.search_knowledge("project documentation")
    print("Search results:", results)

    # Cleanup
    notebook.cleanup()

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from src.rl_model import RLModel

from config.settings import Settings
from src.prioritization.exploration_manager import AdaptiveExplorationManager
from src.prioritization.state_encoder import StateEncoder
from src.prioritization.strategy_ensemble import StrategyEnsemble
from src.prioritization.task_evaluator import TaskEvaluator


class Task:
    """Task class for the prioritization engine."""

    def __init__(
        self,
        title: str,
        description: str,
        priority: int = 1,
        due_date: Optional[str] = None,
    ):
        """Initialize a new task."""
        self.title = title
        self.description = description
        self.priority = priority
        self.due_date = due_date
        self.created_at = datetime.now().isoformat()
        self.completed = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "due_date": self.due_date,
            "created_at": self.created_at,
            "completed": self.completed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create task from dictionary."""
        task = cls(
            title=data["title"],
            description=data["description"],
            priority=data["priority"],
            due_date=data["due_date"],
        )
        task.created_at = data["created_at"]
        task.completed = data["completed"]
        return task


class PrioritizationEngine:
    """Task prioritization engine using reinforcement learning."""

    def __init__(self, settings: Settings):
        """Initialize the prioritization engine."""
        self.settings = settings
        self.rl_model = RLModel(settings)
        self.state_encoder = StateEncoder(settings)
        self.task_evaluator = TaskEvaluator(settings)
        self.exploration_manager = AdaptiveExplorationManager(settings)
        self.strategy_ensemble = StrategyEnsemble(settings)
        self._ensure_data_directories()

    def _ensure_data_directories(self):
        """Ensure all required data directories exist."""
        directories = [
            self.settings.data_dir,
            os.path.join(self.settings.data_dir, "tasks"),
            os.path.join(self.settings.data_dir, "preferences"),
            os.path.join(self.settings.data_dir, "learning"),
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def add_task(self, task: Task) -> Dict[str, Any]:
        """Add a new task to the system."""
        # Save task to file
        task_file = os.path.join(
            self.settings.data_dir,
            "tasks",
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

        with open(task_file, "w") as f:
            json.dump(task.to_dict(), f, indent=2)

        return task.to_dict()

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
                task_data = json.load(f)
                task = Task.from_dict(task_data)

            if completed is not None and task.completed != completed:
                continue

            if priority is not None and task.priority != priority:
                continue

            tasks.append(task.to_dict())

        return tasks

    def prioritize_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize tasks using the RL model and exploration strategies."""
        # Encode current state
        state = self.state_encoder.encode(tasks)

        # Get possible tasks
        possible_tasks = self._generate_possible_tasks(tasks)

        # Select exploration strategy based on context
        strategy = self.exploration_manager.select_strategy(state)

        # Use strategy to select task
        selected_tasks = strategy.select_tasks(state, possible_tasks)

        # Record selection for learning
        self._record_selection(state, selected_tasks, strategy)

        return selected_tasks

    def _generate_possible_tasks(
        self, tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate possible task variations for exploration."""
        return self.strategy_ensemble.generate_variations(tasks)

    def _record_selection(
        self, state: Dict[str, Any], selected_tasks: List[Dict[str, Any]], strategy: Any
    ):
        """Record task selection for learning."""
        learning_data = {
            "state": state,
            "selected_tasks": selected_tasks,
            "strategy": strategy.__class__.__name__,
            "timestamp": datetime.now().isoformat(),
        }

        # Save learning data
        learning_file = os.path.join(
            self.settings.data_dir,
            "learning",
            f"selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

        with open(learning_file, "w") as f:
            json.dump(learning_data, f, indent=2)

    def cleanup(self):
        """Clean up resources."""
        self.rl_model.cleanup()
        self.state_encoder.cleanup()
        self.task_evaluator.cleanup()
        self.exploration_manager.cleanup()
        self.strategy_ensemble.cleanup()


if __name__ == "__main__":
    # Example usage
    settings = Settings()
    engine = PrioritizationEngine(settings)

    # Add sample tasks
    task1 = Task(
        title="Complete project documentation",
        description="Write detailed documentation for the voice AI assistant project",
        priority=2,
        due_date=(datetime.now() + timedelta(days=7)).isoformat(),
    )

    task2 = Task(
        title="Fix bugs in the code",
        description="Address critical bugs in the voice AI assistant",
        priority=1,
        due_date=(datetime.now() + timedelta(days=3)).isoformat(),
    )

    # Add tasks to the system
    engine.add_task(task1)
    engine.add_task(task2)

    # Update preferences
    prefs = engine.update_preferences(
        {
            "important_categories": ["work", "health"],
            "notification_preferences": {"email": True, "push": True, "sound": True},
        }
    )

    # Get and prioritize tasks
    tasks = engine.get_tasks()
    prioritized_tasks = engine.prioritize_tasks(tasks)
    print("Prioritized tasks:", prioritized_tasks)

    # Cleanup
    engine.cleanup()

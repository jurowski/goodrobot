import os
from datetime import datetime
from typing import Any, Dict, List

from config.settings import Settings


class StateEncoder:
    """Encodes task states for the RL model."""

    def __init__(self, settings: Settings):
        """Initialize the state encoder."""
        self.settings = settings
        self._ensure_data_directory()

    def _ensure_data_directory(self):
        """Ensure the data directory exists."""
        os.makedirs(self.settings.data_dir, exist_ok=True)

    def encode(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Encode a list of tasks into a state representation."""
        # Calculate task statistics
        task_stats = self._calculate_task_stats(tasks)

        # Encode temporal features
        temporal_features = self._encode_temporal_features(tasks)

        # Encode priority distribution
        priority_dist = self._encode_priority_distribution(tasks)

        # Combine all features
        state = {
            "task_stats": task_stats,
            "temporal_features": temporal_features,
            "priority_distribution": priority_dist,
            "timestamp": datetime.now().isoformat(),
        }

        return state

    def _calculate_task_stats(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics about the tasks."""
        total_tasks = len(tasks)
        completed_tasks = sum(1 for task in tasks if task.get("completed", False))
        pending_tasks = total_tasks - completed_tasks

        # Calculate average priority
        priorities = [task.get("priority", 1) for task in tasks]
        avg_priority = sum(priorities) / len(priorities) if priorities else 0

        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "pending_tasks": pending_tasks,
            "average_priority": avg_priority,
        }

    def _encode_temporal_features(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Encode temporal features of tasks."""
        now = datetime.now()

        # Calculate time until due dates
        due_dates = []
        for task in tasks:
            if "due_date" in task and task["due_date"]:
                try:
                    due_date = datetime.fromisoformat(task["due_date"])
                    time_until_due = (due_date - now).total_seconds()
                    due_dates.append(time_until_due)
                except ValueError:
                    continue

        # Calculate temporal statistics
        if due_dates:
            min_time_until_due = min(due_dates)
            max_time_until_due = max(due_dates)
            avg_time_until_due = sum(due_dates) / len(due_dates)
        else:
            min_time_until_due = max_time_until_due = avg_time_until_due = 0

        return {
            "min_time_until_due": min_time_until_due,
            "max_time_until_due": max_time_until_due,
            "avg_time_until_due": avg_time_until_due,
            "tasks_with_due_dates": len(due_dates),
        }

    def _encode_priority_distribution(
        self, tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Encode the distribution of task priorities."""
        priority_counts = {}
        for task in tasks:
            priority = task.get("priority", 1)
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

        return priority_counts

    def cleanup(self):
        """Clean up resources."""

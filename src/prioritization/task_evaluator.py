import os
from datetime import datetime
from typing import Any, Dict, Optional

from config.settings import Settings


class TaskEvaluator:
    """Evaluates tasks based on various criteria."""

    def __init__(self, settings: Settings):
        """Initialize the task evaluator."""
        self.settings = settings
        self._ensure_data_directory()

    def _ensure_data_directory(self):
        """Ensure the data directory exists."""
        os.makedirs(self.settings.data_dir, exist_ok=True)

    def evaluate_task(
        self, task: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate a task based on various criteria."""
        # Calculate urgency score
        urgency_score = self._calculate_urgency_score(task)

        # Calculate importance score
        importance_score = self._calculate_importance_score(task)

        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(task)

        # Calculate context score if context is provided
        context_score = self._calculate_context_score(task, context) if context else 0

        # Combine scores
        total_score = self._combine_scores(
            urgency_score, importance_score, complexity_score, context_score
        )

        return {
            "urgency_score": urgency_score,
            "importance_score": importance_score,
            "complexity_score": complexity_score,
            "context_score": context_score,
            "total_score": total_score,
            "timestamp": datetime.now().isoformat(),
        }

    def _calculate_urgency_score(self, task: Dict[str, Any]) -> float:
        """Calculate urgency score based on due date and priority."""
        now = datetime.now()
        urgency_score = 0.0

        # Check due date
        if "due_date" in task and task["due_date"]:
            try:
                due_date = datetime.fromisoformat(task["due_date"])
                time_until_due = (due_date - now).total_seconds()

                # Convert to days
                days_until_due = time_until_due / (24 * 3600)

                # Calculate urgency based on days until due
                if days_until_due <= 1:
                    urgency_score = 1.0
                elif days_until_due <= 3:
                    urgency_score = 0.8
                elif days_until_due <= 7:
                    urgency_score = 0.6
                elif days_until_due <= 14:
                    urgency_score = 0.4
                else:
                    urgency_score = 0.2
            except ValueError:
                pass

        # Adjust based on priority
        priority = task.get("priority", 1)
        priority_factor = priority / 5.0  # Assuming max priority is 5

        # Combine factors
        return (urgency_score * 0.7) + (priority_factor * 0.3)

    def _calculate_importance_score(self, task: Dict[str, Any]) -> float:
        """Calculate importance score based on task content and metadata."""
        importance_score = 0.0

        # Check for important keywords in title and description
        important_keywords = ["critical", "urgent", "important", "high priority"]
        text = f"{task.get('title', '')} {task.get('description', '')}".lower()

        for keyword in important_keywords:
            if keyword in text:
                importance_score += 0.2

        # Cap at 1.0
        return min(importance_score, 1.0)

    def _calculate_complexity_score(self, task: Dict[str, Any]) -> float:
        """Calculate complexity score based on task description length and content."""
        complexity_score = 0.0

        # Check description length
        description = task.get("description", "")
        if len(description) > 200:
            complexity_score += 0.3
        elif len(description) > 100:
            complexity_score += 0.2
        elif len(description) > 50:
            complexity_score += 0.1

        # Check for complexity indicators
        complexity_indicators = ["multiple", "several", "complex", "detailed"]
        text = f"{task.get('title', '')} {task.get('description', '')}".lower()

        for indicator in complexity_indicators:
            if indicator in text:
                complexity_score += 0.2

        # Cap at 1.0
        return min(complexity_score, 1.0)

    def _calculate_context_score(
        self, task: Dict[str, Any], context: Dict[str, Any]
    ) -> float:
        """Calculate context score based on current state and preferences."""
        context_score = 0.0

        # Check if task matches important categories
        if "important_categories" in context:
            important_categories = context["important_categories"]
            task_categories = task.get("categories", [])

            for category in task_categories:
                if category in important_categories:
                    context_score += 0.3

        # Check if task matches current focus
        if "current_focus" in context:
            current_focus = context["current_focus"]
            if current_focus in task.get("title", "").lower():
                context_score += 0.4

        # Cap at 1.0
        return min(context_score, 1.0)

    def _combine_scores(
        self, urgency: float, importance: float, complexity: float, context: float
    ) -> float:
        """Combine individual scores into a total score."""
        # Weighted combination
        return (
            (urgency * 0.4) + (importance * 0.3) + (complexity * 0.2) + (context * 0.1)
        )

    def cleanup(self):
        """Clean up resources."""

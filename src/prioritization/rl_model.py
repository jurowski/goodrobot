import json
import os
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from sklearn.linear_model import SGDRegressor

from config.settings import Settings
from src.prioritization.state_encoder import StateEncoder
from src.prioritization.task_evaluator import TaskEvaluator


class RLModel:
    """Reinforcement learning model for task prioritization."""

    def __init__(self, settings: Settings):
        """Initialize the RL model."""
        self.settings = settings
        self.task_evaluator = TaskEvaluator(settings)
        self.state_encoder = StateEncoder(settings)
        self._ensure_data_directory()
        
        # Initialize Q-learning model
        self.q_model = SGDRegressor(
            learning_rate='constant',
            eta0=0.01,
            max_iter=1000,
            tol=1e-3
        )
        self.feature_dim = 10  # Number of features in state representation
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Q-learning model with random weights."""
        # Create initial weights
        self.q_model.coef_ = np.random.randn(self.feature_dim)
        self.q_model.intercept_ = np.random.randn(1)

    def _ensure_data_directory(self):
        """Ensure the data directory exists."""
        os.makedirs(self.settings.data_dir, exist_ok=True)

    def train(self, training_data: List[Dict[str, Any]], epochs: int = 100):
        """Train the RL model on historical task data."""
        # Convert training data to state-action pairs
        states = []
        actions = []
        rewards = []

        for task in training_data:
            state = self.state_encoder.encode(task)
            action = task.get("priority", 1)
            reward = self._calculate_reward(task)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

        # Train the model
        self._train_model(
            np.array(states), np.array(actions), np.array(rewards), epochs
        )

    def _train_model(
        self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, epochs: int
    ):
        """Train the model using Q-learning."""
        # Convert states to feature vectors
        X = self._prepare_features(states)
        
        # Train for specified epochs
        for epoch in range(epochs):
            # Calculate Q-values for current state
            current_q_values = self.q_model.predict(X)
            
            # Update Q-values using Bellman equation
            target_q_values = rewards + 0.9 * np.max(current_q_values)  # Discount factor of 0.9
            
            # Update model
            self.q_model.partial_fit(X, target_q_values)

    def predict_priority(self, task: Dict[str, Any]) -> int:
        """Predict the priority for a given task."""
        state = self.state_encoder.encode(task)
        priority_score = self._predict(state)
        return self._convert_score_to_priority(priority_score)

    def _predict(self, state: np.ndarray) -> float:
        """Make a prediction for the given state."""
        # Prepare features
        X = self._prepare_features(state.reshape(1, -1))
        
        # Predict Q-value
        q_value = self.q_model.predict(X)[0]
        
        return q_value

    def _prepare_features(self, states: np.ndarray) -> np.ndarray:
        """Prepare features for the model."""
        # Normalize features
        states = (states - np.mean(states, axis=0)) / (np.std(states, axis=0) + 1e-8)
        return states

    def _calculate_reward(self, task: Dict[str, Any]) -> float:
        """Calculate the reward for a task based on its outcome."""
        base_reward = 0.0
        
        # Reward for completion
        if task.get("completed", False):
            base_reward += 1.0
            
            # Additional reward for early completion
            if "due_date" in task and "completion_time" in task:
                due_date = datetime.fromisoformat(task["due_date"])
                completion_time = datetime.fromisoformat(task["completion_time"])
                if completion_time < due_date:
                    base_reward += 0.5
                    
        # Penalty for missing deadline
        elif "due_date" in task:
            due_date = datetime.fromisoformat(task["due_date"])
            if datetime.now() > due_date:
                base_reward -= 0.5
                
        # Adjust reward based on priority
        priority = task.get("priority", 1)
        base_reward *= (priority / 5.0)  # Normalize priority to 0-1 range
        
        return base_reward

    def _convert_score_to_priority(self, score: float) -> int:
        """Convert a prediction score to a priority level."""
        if score >= 0.8:
            return 5
        elif score >= 0.6:
            return 4
        elif score >= 0.4:
            return 3
        elif score >= 0.2:
            return 2
        else:
            return 1

    def save_model(self, path: str):
        """Save the trained model to disk."""
        model_data = {
            "weights": self._get_model_weights(),
            "metadata": {"created_at": datetime.now().isoformat(), "version": "1.0"},
        }

        with open(path, "w") as f:
            json.dump(model_data, f, indent=2)

    def load_model(self, path: str):
        """Load a trained model from disk."""
        with open(path, "r") as f:
            model_data = json.load(f)

        self._set_model_weights(model_data["weights"])

    def _get_model_weights(self) -> Dict[str, Any]:
        """Get the current model weights."""
        return {
            "coef": self.q_model.coef_.tolist(),
            "intercept": self.q_model.intercept_.tolist()
        }

    def _set_model_weights(self, weights: Dict[str, Any]):
        """Set the model weights."""
        self.q_model.coef_ = np.array(weights["coef"])
        self.q_model.intercept_ = np.array(weights["intercept"])

    def cleanup(self):
        """Clean up resources."""
        self.task_evaluator.cleanup()
        self.state_encoder.cleanup()


if __name__ == "__main__":
    # Example usage
    settings = Settings()
    model = RLModel(settings)

    # Sample training data
    training_data = [
        {
            "id": "task1",
            "title": "Complete project documentation",
            "description": "Write detailed documentation",
            "priority": 5,
            "completed": True,
            "completion_time": datetime.now().isoformat(),
        },
        {
            "id": "task2",
            "title": "Review code changes",
            "description": "Review recent pull requests",
            "priority": 3,
            "completed": True,
            "completion_time": datetime.now().isoformat(),
        },
    ]

    # Train the model
    model.train(training_data, epochs=100)

    # Save the model
    model.save_model("models/rl_model.json")

    # Load the model
    model.load_model("models/rl_model.json")

    # Predict priority for a new task
    new_task = {
        "id": "task3",
        "title": "Implement new feature",
        "description": "Add user authentication system",
    }

    priority = model.predict_priority(new_task)
    print(f"Predicted priority: {priority}")

    # Cleanup
    model.cleanup()

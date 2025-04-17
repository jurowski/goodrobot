import json
import os
import random
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from config.settings import Settings


class BaseStrategy:
    """Base class for exploration strategies."""

    def __init__(self, settings: Settings):
        """Initialize the strategy."""
        self.settings = settings
        self._ensure_data_directory()

    def _ensure_data_directory(self):
        """Ensure the data directory exists."""
        os.makedirs(self.settings.data_dir, exist_ok=True)

    def select_tasks(
        self, state: Dict[str, Any], possible_tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Select tasks using the strategy."""
        raise NotImplementedError

    def calculate_state_compatibility(self, state: Dict[str, Any]) -> float:
        """Calculate how compatible the strategy is with the current state."""
        raise NotImplementedError

    def cleanup(self):
        """Clean up resources."""


class EpsilonGreedyStrategy(BaseStrategy):
    """Epsilon-greedy exploration strategy."""

    def __init__(self, settings: Settings):
        """Initialize the epsilon-greedy strategy."""
        super().__init__(settings)
        self.epsilon = 0.1  # Exploration rate

    def select_tasks(
        self, state: Dict[str, Any], possible_tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Select tasks using epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            # Explore: randomly select tasks
            return random.sample(possible_tasks, min(3, len(possible_tasks)))
        else:
            # Exploit: select tasks with highest priority
            return sorted(
                possible_tasks, key=lambda x: x.get("priority", 1), reverse=True
            )[:3]

    def calculate_state_compatibility(self, state: Dict[str, Any]) -> float:
        """Calculate state compatibility for epsilon-greedy."""
        # More compatible when we have many tasks to explore
        total_tasks = state.get("task_stats", {}).get("total_tasks", 0)
        return min(total_tasks / 10.0, 1.0)


class ThompsonSamplingStrategy(BaseStrategy):
    """Thompson sampling exploration strategy."""

    def __init__(self, settings: Settings):
        """Initialize the Thompson sampling strategy."""
        super().__init__(settings)
        self.min_alpha = 1.0  # Minimum alpha value for prior
        self.min_beta = 1.0   # Minimum beta value for prior
        self.decay_rate = 0.99  # Decay rate for old observations
        self._load_prior_distributions()

    def _load_prior_distributions(self):
        """Load prior distributions for tasks."""
        dist_file = os.path.join(self.settings.data_dir, "thompson_priors.json")
        if os.path.exists(dist_file):
            with open(dist_file, "r") as f:
                data = json.load(f)
                self.prior_distributions = data.get("priors", {})
                self.task_features = data.get("features", {})
                self.last_update = datetime.fromisoformat(data.get("last_update", datetime.now().isoformat()))
        else:
            self.prior_distributions = {}
            self.task_features = {}
            self.last_update = datetime.now()

    def _save_prior_distributions(self):
        """Save prior distributions for tasks."""
        dist_file = os.path.join(self.settings.data_dir, "thompson_priors.json")
        data = {
            "priors": self.prior_distributions,
            "features": self.task_features,
            "last_update": datetime.now().isoformat()
        }
        with open(dist_file, "w") as f:
            json.dump(data, f, indent=2)

    def _decay_priors(self):
        """Apply decay to old observations."""
        current_time = datetime.now()
        hours_since_update = (current_time - self.last_update).total_seconds() / 3600
        
        if hours_since_update > 24:  # Only decay if more than 24 hours have passed
            for task_id in self.prior_distributions:
                # Apply exponential decay
                self.prior_distributions[task_id]["alpha"] = max(
                    self.min_alpha,
                    self.prior_distributions[task_id]["alpha"] * (self.decay_rate ** hours_since_update)
                )
                self.prior_distributions[task_id]["beta"] = max(
                    self.min_beta,
                    self.prior_distributions[task_id]["beta"] * (self.decay_rate ** hours_since_update)
                )
            
            self.last_update = current_time
            self._save_prior_distributions()

    def _extract_task_features(self, task: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from task for similarity comparison."""
        features = {
            "priority": task.get("priority", 1) / 5.0,
            "description_length": len(task.get("description", "")) / 100.0,
            "has_due_date": 1.0 if "due_date" in task else 0.0,
            "is_completed": 1.0 if task.get("completed", False) else 0.0
        }
        
        if "due_date" in task:
            due_date = datetime.fromisoformat(task["due_date"])
            features["time_until_due"] = min(
                (due_date - datetime.now()).total_seconds() / 86400.0,
                1.0
            )
        else:
            features["time_until_due"] = 1.0
            
        return features

    def _calculate_similarity(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """Calculate similarity between two sets of features."""
        # Cosine similarity
        dot_product = sum(f1 * f2 for f1, f2 in zip(features1.values(), features2.values()))
        norm1 = np.sqrt(sum(f ** 2 for f in features1.values()))
        norm2 = np.sqrt(sum(f ** 2 for f in features2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)

    def _find_similar_task(self, task: Dict[str, Any]) -> Optional[str]:
        """Find a similar task in the prior distributions."""
        current_features = self._extract_task_features(task)
        best_similarity = 0.7  # Minimum similarity threshold
        similar_task_id = None
        
        for task_id, features in self.task_features.items():
            similarity = self._calculate_similarity(current_features, features)
            if similarity > best_similarity:
                best_similarity = similarity
                similar_task_id = task_id
                
        return similar_task_id

    def select_tasks(
        self, state: Dict[str, Any], possible_tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Select tasks using Thompson sampling."""
        self._decay_priors()  # Apply decay to old observations
        sampled_tasks = []
        
        for task in possible_tasks:
            task_id = task.get("id", str(hash(str(task))))
            
            # Store task features
            self.task_features[task_id] = self._extract_task_features(task)
            
            # Find similar task if exists
            similar_task_id = self._find_similar_task(task)
            
            if similar_task_id and similar_task_id in self.prior_distributions:
                # Use similar task's distribution with slight randomization
                similar_prior = self.prior_distributions[similar_task_id]
                alpha = similar_prior["alpha"] * 0.9 + self.min_alpha * 0.1
                beta = similar_prior["beta"] * 0.9 + self.min_beta * 0.1
            else:
                # Initialize with default priors
                alpha = self.min_alpha
                beta = self.min_beta
            
            # Sample from beta distribution
            sample = random.betavariate(alpha, beta)
            
            # Adjust sample based on task features
            feature_bonus = sum(self.task_features[task_id].values()) / len(self.task_features[task_id])
            adjusted_sample = (0.7 * sample) + (0.3 * feature_bonus)
            
            sampled_tasks.append((task, adjusted_sample))
            
            # Initialize or update prior distribution
            if task_id not in self.prior_distributions:
                self.prior_distributions[task_id] = {"alpha": alpha, "beta": beta}

        # Select top tasks based on samples
        selected_tasks = [
            task
            for task, _ in sorted(sampled_tasks, key=lambda x: x[1], reverse=True)[:3]
        ]

        self._save_prior_distributions()
        return selected_tasks

    def update_prior(self, task: Dict[str, Any], success: bool):
        """Update the prior distribution for a task."""
        task_id = task.get("id", str(hash(str(task))))
        
        if task_id in self.prior_distributions:
            if success:
                self.prior_distributions[task_id]["alpha"] += 1
            else:
                self.prior_distributions[task_id]["beta"] += 1
                
            # Ensure minimum values
            self.prior_distributions[task_id]["alpha"] = max(
                self.min_alpha,
                self.prior_distributions[task_id]["alpha"]
            )
            self.prior_distributions[task_id]["beta"] = max(
                self.min_beta,
                self.prior_distributions[task_id]["beta"]
            )
            
            self._save_prior_distributions()

    def calculate_state_compatibility(self, state: Dict[str, Any]) -> float:
        """Calculate state compatibility for Thompson sampling."""
        # More compatible when we have historical data and similar tasks
        total_priors = len(self.prior_distributions)
        avg_observations = sum(
            (p["alpha"] + p["beta"] - 2)  # Subtract initial priors
            for p in self.prior_distributions.values()
        ) / max(1, total_priors)
        
        return min((total_priors * avg_observations) / 100.0, 1.0)

    def cleanup(self):
        """Clean up resources."""
        self._save_prior_distributions()


class LinUCBStrategy(BaseStrategy):
    """Linear Upper Confidence Bound exploration strategy."""

    def __init__(self, settings: Settings):
        """Initialize the LinUCB strategy."""
        super().__init__(settings)
        self.alpha = 1.0  # Exploration parameter
        self.feature_dim = 10  # Number of features
        self._load_model_parameters()

    def _load_model_parameters(self):
        """Load model parameters."""
        param_file = os.path.join(self.settings.data_dir, "linucb_params.json")
        if os.path.exists(param_file):
            with open(param_file, "r") as f:
                self.model_params = json.load(f)
        else:
            # Initialize with identity matrix for A and zero vectors for b and theta
            self.model_params = {
                "A": np.identity(self.feature_dim).tolist(),
                "b": np.zeros(self.feature_dim).tolist(),
                "theta": np.zeros(self.feature_dim).tolist(),
                "feature_counts": {}  # Track how many times each feature has been seen
            }

    def _save_model_parameters(self):
        """Save model parameters."""
        param_file = os.path.join(self.settings.data_dir, "linucb_params.json")
        with open(param_file, "w") as f:
            json.dump(self.model_params, f, indent=2)

    def select_tasks(
        self, state: Dict[str, Any], possible_tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Select tasks using LinUCB."""
        scored_tasks = []
        
        # Convert model parameters to numpy arrays
        A = np.array(self.model_params["A"])
        b = np.array(self.model_params["b"])
        theta = np.array(self.model_params["theta"])
        
        for task in possible_tasks:
            # Extract features
            features = self._extract_features(task, state)
            x = np.array(features).reshape(-1, 1)  # Convert to column vector
            
            # Calculate UCB score
            expected_reward = np.dot(theta.T, x)[0][0]
            uncertainty = self.alpha * np.sqrt(np.dot(x.T, np.linalg.inv(A).dot(x)))[0][0]
            ucb_score = expected_reward + uncertainty
            
            scored_tasks.append((task, ucb_score))
            
            # Update feature counts
            for i, feature in enumerate(features):
                feature_key = f"feature_{i}"
                self.model_params["feature_counts"][feature_key] = self.model_params["feature_counts"].get(feature_key, 0) + 1

        # Select top tasks based on UCB scores
        selected_tasks = [
            task
            for task, _ in sorted(scored_tasks, key=lambda x: x[1], reverse=True)[:3]
        ]

        self._save_model_parameters()
        return selected_tasks

    def _extract_features(
        self, task: Dict[str, Any], state: Dict[str, Any]
    ) -> List[float]:
        """Extract features from task and state."""
        features = []

        # Task features
        features.append(task.get("priority", 1) / 5.0)  # Normalized priority
        features.append(len(task.get("description", "")) / 100.0)  # Description length
        
        # Temporal features
        if "due_date" in task:
            due_date = datetime.fromisoformat(task["due_date"])
            time_until_due = (due_date - datetime.now()).total_seconds()
            features.append(min(time_until_due / 86400.0, 1.0))  # Days until due
        else:
            features.append(1.0)  # No due date
        
        # State features
        task_stats = state.get("task_stats", {})
        features.append(task_stats.get("pending_tasks", 0) / 10.0)
        features.append(task_stats.get("completed_tasks", 0) / 10.0)
        
        # Temporal state features
        temporal_features = state.get("temporal_features", {})
        features.append(temporal_features.get("avg_time_until_due", 0) / 86400.0)
        features.append(temporal_features.get("avg_completion_time", 0) / 3600.0)
        
        # Task history features
        task_id = task.get("id", str(hash(str(task))))
        feature_key = f"task_{task_id}"
        features.append(self.model_params["feature_counts"].get(feature_key, 0) / 100.0)
        
        # Add bias term
        features.append(1.0)
        
        return features

    def _calculate_ucb_score(self, features: List[float]) -> float:
        """Calculate UCB score for features."""
        # Convert features to numpy array
        x = np.array(features).reshape(-1, 1)
        
        # Get model parameters
        A = np.array(self.model_params["A"])
        b = np.array(self.model_params["b"])
        theta = np.array(self.model_params["theta"])
        
        # Calculate expected reward
        expected_reward = np.dot(theta.T, x)[0][0]
        
        # Calculate uncertainty
        uncertainty = self.alpha * np.sqrt(np.dot(x.T, np.linalg.inv(A).dot(x)))[0][0]
        
        # Return UCB score
        return expected_reward + uncertainty

    def update_model(self, task: Dict[str, Any], reward: float):
        """Update the model with new observation."""
        # Extract features
        features = self._extract_features(task, {})
        x = np.array(features).reshape(-1, 1)
        
        # Convert model parameters to numpy arrays
        A = np.array(self.model_params["A"])
        b = np.array(self.model_params["b"])
        
        # Update A and b
        A += np.dot(x, x.T)
        b += reward * x.flatten()
        
        # Update theta
        theta = np.linalg.inv(A).dot(b)
        
        # Save updated parameters
        self.model_params["A"] = A.tolist()
        self.model_params["b"] = b.tolist()
        self.model_params["theta"] = theta.tolist()
        
        self._save_model_parameters()

    def calculate_state_compatibility(self, state: Dict[str, Any]) -> float:
        """Calculate state compatibility for LinUCB."""
        # More compatible when we have many features and sufficient data
        feature_count = len(self.model_params["feature_counts"])
        total_observations = sum(self.model_params["feature_counts"].values())
        return min((feature_count * total_observations) / 1000.0, 1.0)

    def cleanup(self):
        """Clean up resources."""
        self._save_model_parameters()


class ProgressiveValidationStrategy(BaseStrategy):
    """Progressive validation exploration strategy."""

    def __init__(self, settings: Settings):
        """Initialize the progressive validation strategy."""
        super().__init__(settings)
        self._load_validation_data()

    def _load_validation_data(self):
        """Load validation data."""
        val_file = os.path.join(self.settings.data_dir, "validation_data.json")
        if os.path.exists(val_file):
            with open(val_file, "r") as f:
                self.validation_data = json.load(f)
        else:
            self.validation_data = {"task_performance": {}, "validation_scores": []}

    def _save_validation_data(self):
        """Save validation data."""
        val_file = os.path.join(self.settings.data_dir, "validation_data.json")
        with open(val_file, "w") as f:
            json.dump(self.validation_data, f, indent=2)

    def select_tasks(
        self, state: Dict[str, Any], possible_tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Select tasks using progressive validation."""
        scored_tasks = []
        for task in possible_tasks:
            task_id = task.get("id", str(hash(str(task))))

            # Calculate validation score
            if task_id in self.validation_data["task_performance"]:
                perf = self.validation_data["task_performance"][task_id]
                score = perf["success_count"] / perf["total_count"]
            else:
                score = 0.5

            scored_tasks.append((task, score))

        # Select top tasks based on validation scores
        selected_tasks = [
            task
            for task, _ in sorted(scored_tasks, key=lambda x: x[1], reverse=True)[:3]
        ]

        self._save_validation_data()
        return selected_tasks

    def calculate_state_compatibility(self, state: Dict[str, Any]) -> float:
        """Calculate state compatibility for progressive validation."""
        # More compatible when we have validation data
        return min(len(self.validation_data["task_performance"]) / 50.0, 1.0)

    def cleanup(self):
        """Clean up resources."""
        self._save_validation_data()


class DomainGuidedExploration(BaseStrategy):
    """Domain-guided exploration strategy."""

    def __init__(self, settings: Settings):
        """Initialize the domain-guided exploration strategy."""
        super().__init__(settings)
        self.min_pattern_score = 0.1
        self.min_rule_score = 0.1
        self.sequence_decay = 0.95
        self._load_domain_knowledge()
        self._initialize_default_rules()
        self._initialize_default_patterns()

    def _initialize_default_rules(self):
        """Initialize default domain rules if none exist."""
        if not self.domain_knowledge["domain_rules"]:
            self.domain_knowledge["domain_rules"] = [
                # Peak Productivity Rules
                {
                    "id": "peak_productivity_high_priority",
                    "score": 0.9,
                    "conditions": [
                        {
                            "field": "temporal.is_peak_productivity",
                            "operator": "equals",
                            "value": True
                        },
                        {
                            "field": "task.priority",
                            "operator": "greater_than",
                            "value": 3
                        },
                        {
                            "field": "task.description_length",
                            "operator": "less_than",
                            "value": 200
                        }
                    ]
                },
                {
                    "id": "low_energy_quick_tasks",
                    "score": 0.7,
                    "conditions": [
                        {
                            "field": "temporal.is_low_energy",
                            "operator": "equals",
                            "value": True
                        },
                        {
                            "field": "task.description_length",
                            "operator": "less_than",
                            "value": 100
                        },
                        {
                            "field": "task.has_dependencies",
                            "operator": "equals",
                            "value": False
                        }
                    ]
                },
                
                # Time of Day Rules
                {
                    "id": "early_morning_planning",
                    "score": 0.8,
                    "conditions": [
                        {
                            "field": "temporal.time_period",
                            "operator": "equals",
                            "value": "early_morning"
                        },
                        {
                            "field": "task.description",
                            "operator": "contains_any",
                            "value": ["plan", "schedule", "organize"]
                        }
                    ]
                },
                {
                    "id": "late_afternoon_review",
                    "score": 0.75,
                    "conditions": [
                        {
                            "field": "temporal.time_period",
                            "operator": "equals",
                            "value": "late_afternoon"
                        },
                        {
                            "field": "task.description",
                            "operator": "contains_any",
                            "value": ["review", "check", "verify"]
                        }
                    ]
                },
                
                # Business Calendar Rules
                {
                    "id": "first_business_day_high_priority",
                    "score": 0.85,
                    "conditions": [
                        {
                            "field": "temporal.is_first_business_day",
                            "operator": "equals",
                            "value": True
                        },
                        {
                            "field": "task.priority",
                            "operator": "greater_than",
                            "value": 3
                        }
                    ]
                },
                {
                    "id": "last_business_day_urgent",
                    "score": 0.9,
                    "conditions": [
                        {
                            "field": "temporal.is_last_business_day",
                            "operator": "equals",
                            "value": True
                        },
                        {
                            "field": "task.urgency_score",
                            "operator": "greater_than",
                            "value": 0.7
                        }
                    ]
                },
                
                # Seasonal Rules
                {
                    "id": "holiday_season_high_priority",
                    "score": 0.8,
                    "conditions": [
                        {
                            "field": "temporal.is_holiday_season",
                            "operator": "equals",
                            "value": True
                        },
                        {
                            "field": "task.priority",
                            "operator": "greater_than",
                            "value": 3
                        },
                        {
                            "field": "task.is_due_this_month",
                            "operator": "equals",
                            "value": True
                        }
                    ]
                },
                {
                    "id": "summer_break_quick_tasks",
                    "score": 0.7,
                    "conditions": [
                        {
                            "field": "temporal.is_summer_break",
                            "operator": "equals",
                            "value": True
                        },
                        {
                            "field": "task.description_length",
                            "operator": "less_than",
                            "value": 150
                        },
                        {
                            "field": "task.has_dependencies",
                            "operator": "equals",
                            "value": False
                        }
                    ]
                },
                
                # Work Schedule Rules
                {
                    "id": "work_hours_complex_tasks",
                    "score": 0.8,
                    "conditions": [
                        {
                            "field": "temporal.is_typical_work_hours",
                            "operator": "equals",
                            "value": True
                        },
                        {
                            "field": "task.description_length",
                            "operator": "greater_than",
                            "value": 200
                        },
                        {
                            "field": "task.has_dependencies",
                            "operator": "equals",
                            "value": True
                        }
                    ]
                },
                {
                    "id": "global_collaboration_communication",
                    "score": 0.75,
                    "conditions": [
                        {
                            "field": "temporal.is_global_collaboration_hours",
                            "operator": "equals",
                            "value": True
                        },
                        {
                            "field": "task.description",
                            "operator": "contains_any",
                            "value": ["meet", "discuss", "coordinate", "sync"]
                        }
                    ]
                },
                
                # Due Date Rules
                {
                    "id": "due_today_high_priority",
                    "score": 0.95,
                    "conditions": [
                        {
                            "field": "task.is_due_today",
                            "operator": "equals",
                            "value": True
                        },
                        {
                            "field": "task.priority",
                            "operator": "greater_than",
                            "value": 2
                        }
                    ]
                },
                {
                    "id": "due_weekend_urgent",
                    "score": 0.9,
                    "conditions": [
                        {
                            "field": "task.is_due_on_weekend",
                            "operator": "equals",
                            "value": True
                        },
                        {
                            "field": "task.urgency_score",
                            "operator": "greater_than",
                            "value": 0.8
                        }
                    ]
                },
                {
                    "id": "due_business_day_complex",
                    "score": 0.85,
                    "conditions": [
                        {
                            "field": "task.is_due_on_business_day",
                            "operator": "equals",
                            "value": True
                        },
                        {
                            "field": "task.description_length",
                            "operator": "greater_than",
                            "value": 150
                        },
                        {
                            "field": "task.has_dependencies",
                            "operator": "equals",
                            "value": True
                        }
                    ]
                },
                
                # Time-Based Priority Rules
                {
                    "id": "high_time_priority_quick",
                    "score": 0.9,
                    "conditions": [
                        {
                            "field": "task.time_based_priority",
                            "operator": "greater_than",
                            "value": 0.8
                        },
                        {
                            "field": "task.description_length",
                            "operator": "less_than",
                            "value": 100
                        }
                    ]
                },
                {
                    "id": "low_time_priority_complex",
                    "score": 0.6,
                    "conditions": [
                        {
                            "field": "task.time_based_priority",
                            "operator": "less_than",
                            "value": 0.4
                        },
                        {
                            "field": "task.description_length",
                            "operator": "greater_than",
                            "value": 200
                        }
                    ]
                }
            ]
            self._save_domain_knowledge()

    def _initialize_default_patterns(self):
        """Initialize default task patterns if none exist."""
        if not self.domain_knowledge["task_patterns"]:
            self.domain_knowledge["task_patterns"] = {
                "urgent": {
                    "score": 0.9,
                    "keywords": ["urgent", "asap", "immediate", "critical", "emergency", "pressing"],
                    "context": ["high priority", "top priority", "must do", "cannot wait"]
                },
                "review": {
                    "score": 0.7,
                    "keywords": ["review", "check", "verify", "validate", "inspect", "examine"],
                    "context": ["quality", "accuracy", "completeness", "correctness"]
                },
                "follow_up": {
                    "score": 0.8,
                    "keywords": ["follow up", "check in", "status update", "progress", "track"],
                    "context": ["previous", "ongoing", "pending", "waiting"]
                },
                "planning": {
                    "score": 0.6,
                    "keywords": ["plan", "schedule", "organize", "prepare", "arrange", "coordinate"],
                    "context": ["future", "upcoming", "next", "later"]
                },
                "documentation": {
                    "score": 0.5,
                    "keywords": ["document", "write", "record", "note", "log", "capture"],
                    "context": ["process", "procedure", "guide", "manual"]
                },
                "analysis": {
                    "score": 0.7,
                    "keywords": ["analyze", "evaluate", "assess", "study", "research"],
                    "context": ["data", "results", "findings", "trends"]
                },
                "collaboration": {
                    "score": 0.6,
                    "keywords": ["collaborate", "coordinate", "team", "partner", "work with"],
                    "context": ["team", "group", "department", "stakeholder"]
                },
                "improvement": {
                    "score": 0.65,
                    "keywords": ["improve", "enhance", "optimize", "refine", "better"],
                    "context": ["process", "system", "performance", "efficiency"]
                },
                "communication": {
                    "score": 0.6,
                    "keywords": ["communicate", "discuss", "meet", "present", "share"],
                    "context": ["team", "stakeholder", "client", "customer"]
                },
                "development": {
                    "score": 0.7,
                    "keywords": ["develop", "create", "build", "implement", "code"],
                    "context": ["feature", "function", "system", "application"]
                },
                "testing": {
                    "score": 0.65,
                    "keywords": ["test", "verify", "validate", "check", "debug"],
                    "context": ["quality", "functionality", "performance", "reliability"]
                },
                "deployment": {
                    "score": 0.8,
                    "keywords": ["deploy", "release", "publish", "launch", "rollout"],
                    "context": ["production", "environment", "system", "application"]
                }
            }
            self._save_domain_knowledge()

    def _load_domain_knowledge(self):
        """Load domain knowledge."""
        domain_file = os.path.join(self.settings.data_dir, "domain_knowledge.json")
        if os.path.exists(domain_file):
            with open(domain_file, "r") as f:
                self.domain_knowledge = json.load(f)
        else:
            self.domain_knowledge = {
                "task_patterns": {},
                "successful_sequences": [],
                "domain_rules": [],
                "rule_performance": {},
                "pattern_performance": {},
                "sequence_performance": {}
            }
            self.last_update = datetime.now()

    def _save_domain_knowledge(self):
        """Save domain knowledge."""
        domain_file = os.path.join(self.settings.data_dir, "domain_knowledge.json")
        data = {
            "knowledge": self.domain_knowledge,
            "last_update": datetime.now().isoformat()
        }
        with open(domain_file, "w") as f:
            json.dump(data, f, indent=2)

    def _extract_task_context(self, task: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context from task and state."""
        current_time = datetime.now()
        context = {
            "task": {
                "priority": task.get("priority", 1),
                "has_due_date": "due_date" in task,
                "description_length": len(task.get("description", "")),
                "is_completed": task.get("completed", False),
                "has_dependencies": len(task.get("dependencies", [])) > 0,
                "has_subtasks": len(task.get("subtasks", [])) > 0,
                "has_attachments": len(task.get("attachments", [])) > 0,
                "has_labels": len(task.get("labels", [])) > 0,
                "has_comments": len(task.get("comments", [])) > 0,
                "has_assignee": "assignee" in task,
                "has_project": "project" in task,
                "time_based_priority": self._get_time_based_priority(task, state)
            },
            "state": {
                "pending_tasks": state.get("task_stats", {}).get("pending_tasks", 0),
                "completed_tasks": state.get("task_stats", {}).get("completed_tasks", 0),
                "avg_priority": state.get("task_stats", {}).get("avg_priority", 1),
                "recent_completion_rate": state.get("task_stats", {}).get("recent_completion_rate", 0),
                "task_distribution": state.get("task_stats", {}).get("task_distribution", {}),
                "label_distribution": state.get("label_stats", {}).get("distribution", {}),
                "project_distribution": state.get("project_stats", {}).get("distribution", {})
            },
            "temporal": {
                # Basic time features
                "is_weekend": current_time.weekday() >= 5,
                "is_morning": current_time.hour < 12,
                "is_evening": current_time.hour >= 18,
                "is_work_hours": 9 <= current_time.hour < 17,
                "day_of_week": current_time.weekday(),
                "hour_of_day": current_time.hour,
                
                # Enhanced time period features
                "time_period": self._get_time_period(current_time),
                "time_of_day_category": self._get_time_of_day_category(current_time),
                "is_early_morning": 5 <= current_time.hour < 9,
                "is_late_evening": 20 <= current_time.hour < 24,
                "is_night": current_time.hour < 5 or current_time.hour >= 24,
                "is_afternoon": 12 <= current_time.hour < 17,
                "is_late_afternoon": 15 <= current_time.hour < 17,
                
                # Calendar period features
                "is_month_start": current_time.day == 1,
                "is_month_end": current_time.day >= 28,
                "is_quarter_start": current_time.month in [1, 4, 7, 10] and current_time.day == 1,
                "is_quarter_end": current_time.month in [3, 6, 9, 12] and current_time.day >= 28,
                "is_year_start": current_time.month == 1 and current_time.day == 1,
                "is_year_end": current_time.month == 12 and current_time.day >= 28,
                
                # Business calendar features
                "is_holiday": self._is_holiday(current_time),
                "is_business_day": self._is_business_day(current_time),
                "is_first_business_day": self._is_first_business_day(current_time),
                "is_last_business_day": self._is_last_business_day(current_time),
                
                # Productivity features
                "productivity_score": self._get_productivity_score(current_time),
                "is_peak_productivity": self._is_peak_productivity_hours(current_time),
                "is_low_energy": self._is_low_energy_hours(current_time),
                
                # Season and weather features
                "season": self._get_season(current_time),
                "is_holiday_season": self._is_holiday_season(current_time),
                "is_summer_break": self._is_summer_break(current_time),
                
                # Work schedule features
                "is_typical_work_hours": self._is_typical_work_hours(current_time),
                "is_local_business_hours": self._is_local_business_hours(current_time),
                "is_global_collaboration_hours": self._is_global_collaboration_hours(current_time)
            }
        }
        
        if "due_date" in task:
            due_date = datetime.fromisoformat(task["due_date"])
            time_until_due = (due_date - current_time).total_seconds()
            context["task"].update({
                "hours_until_due": time_until_due / 3600,
                "days_until_due": time_until_due / 86400,
                "is_overdue": time_until_due < 0,
                "is_due_today": 0 <= time_until_due < 86400,
                "is_due_this_week": 0 <= time_until_due < 604800,
                "is_due_this_month": 0 <= time_until_due < 2592000,
                "is_due_next_month": 2592000 <= time_until_due < 5184000,
                "due_time_period": self._get_time_period(due_date),
                "due_time_of_day": self._get_time_of_day(due_date),
                "due_day_of_week": due_date.weekday(),
                "is_due_on_weekend": due_date.weekday() >= 5,
                "is_due_on_holiday": self._is_holiday(due_date),
                "is_due_on_business_day": self._is_business_day(due_date),
                "urgency_score": self._get_urgency_score(task, current_time)
            })
            
        return context

    def _is_holiday(self, date: datetime) -> bool:
        """Check if the given date is a holiday."""
        # This is a simplified implementation. In a real system, you would
        # want to use a proper holiday calendar or API.
        holidays = {
            (1, 1): "New Year's Day",
            (7, 4): "Independence Day",
            (12, 25): "Christmas Day"
        }
        return (date.month, date.day) in holidays

    def _is_business_day(self, date: datetime) -> bool:
        """Check if the given date is a business day."""
        return date.weekday() < 5 and not self._is_holiday(date)

    def _get_time_of_day(self, date: datetime) -> str:
        """Get the time of day category."""
        hour = date.hour
        if hour < 6:
            return "early_morning"
        elif hour < 12:
            return "morning"
        elif hour < 17:
            return "afternoon"
        elif hour < 21:
            return "evening"
        else:
            return "night"

    def _get_season(self, date: datetime) -> str:
        """Get the season for the given date."""
        month = date.month
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "fall"

    def _find_matching_patterns(self, task_text: str, temporal_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find patterns that match the task text and temporal context."""
        matches = []
        task_text = task_text.lower()
        
        for pattern, pattern_data in self.domain_knowledge["task_patterns"].items():
            # Check for keyword matches
            keyword_matches = sum(
                1 for keyword in pattern_data.get("keywords", [])
                if keyword in task_text
            )
            
            # Check for context matches
            context_matches = sum(
                1 for context in pattern_data.get("context", [])
                if context in task_text
            )
            
            # Calculate temporal relevance
            temporal_relevance = self._calculate_temporal_relevance(pattern, temporal_context)
            
            # Get temporal pattern performance
            temporal_performance = self._get_temporal_pattern_performance(pattern, temporal_context)
            
            if keyword_matches > 0 or context_matches > 0 or temporal_relevance > 0:
                # Calculate pattern score based on matches and temporal relevance
                base_score = pattern_data.get("score", self.min_pattern_score)
                
                # Weight different components
                keyword_weight = 0.3
                context_weight = 0.2
                temporal_weight = 0.3
                performance_weight = 0.2
                
                # Calculate weighted match score
                match_score = base_score * (
                    1 + 
                    (keyword_matches * 0.1 * keyword_weight) + 
                    (context_matches * 0.1 * context_weight) +
                    (temporal_relevance * temporal_weight) +
                    (temporal_performance * performance_weight)
                )
                
                matches.append({
                    "pattern": pattern,
                    "score": min(match_score, 1.0),
                    "performance": temporal_performance,
                    "keyword_matches": keyword_matches,
                    "context_matches": context_matches,
                    "temporal_relevance": temporal_relevance
                })
                
        return matches

    def _calculate_temporal_relevance(self, pattern: str, temporal_context: Dict[str, Any]) -> float:
        """Calculate temporal relevance for a pattern."""
        relevance = 0.0
        
        # Define temporal patterns for different task types
        temporal_patterns = {
            "urgent": {
                "peak_hours": 0.8,
                "work_hours": 0.6,
                "low_energy": 0.3,
                "holiday": 0.9,
                "weekend": 0.7
            },
            "review": {
                "late_afternoon": 0.8,
                "early_morning": 0.6,
                "work_hours": 0.7,
                "low_energy": 0.4
            },
            "planning": {
                "early_morning": 0.9,
                "first_business_day": 0.8,
                "work_hours": 0.7,
                "peak_hours": 0.6
            },
            "documentation": {
                "low_energy": 0.7,
                "work_hours": 0.6,
                "late_afternoon": 0.5
            },
            "analysis": {
                "peak_hours": 0.8,
                "work_hours": 0.7,
                "early_morning": 0.6
            },
            "collaboration": {
                "work_hours": 0.8,
                "global_collaboration": 0.9,
                "peak_hours": 0.7
            },
            "improvement": {
                "work_hours": 0.7,
                "peak_hours": 0.6,
                "low_energy": 0.5
            },
            "communication": {
                "work_hours": 0.8,
                "global_collaboration": 0.9,
                "peak_hours": 0.7
            },
            "development": {
                "peak_hours": 0.8,
                "work_hours": 0.7,
                "early_morning": 0.6
            },
            "testing": {
                "work_hours": 0.7,
                "late_afternoon": 0.6,
                "low_energy": 0.5
            },
            "deployment": {
                "work_hours": 0.8,
                "low_energy": 0.4,
                "last_business_day": 0.9
            }
        }
        
        # Get pattern's temporal preferences
        pattern_preferences = temporal_patterns.get(pattern, {})
        
        # Calculate relevance based on current temporal context
        for temporal_feature, weight in pattern_preferences.items():
            if temporal_context.get(temporal_feature, False):
                relevance += weight
        
        # Normalize relevance
        if pattern_preferences:
            relevance /= len(pattern_preferences)
        
        return relevance

    def _get_time_based_pattern_score(self, pattern: str, temporal_context: Dict[str, Any]) -> float:
        """Calculate time-based score for a pattern."""
        base_score = 0.5
        
        # Time of day adjustments
        time_period = temporal_context.get("time_period", "")
        if time_period == "early_morning":
            if pattern in ["planning", "development", "analysis"]:
                base_score += 0.2
        elif time_period == "late_afternoon":
            if pattern in ["review", "testing", "documentation"]:
                base_score += 0.2
        elif time_period == "peak_hours":
            if pattern in ["development", "analysis", "communication"]:
                base_score += 0.3
        elif time_period == "low_energy":
            if pattern in ["documentation", "review", "testing"]:
                base_score += 0.2
        
        # Business calendar adjustments
        if temporal_context.get("is_first_business_day", False):
            if pattern in ["planning", "analysis"]:
                base_score += 0.2
        if temporal_context.get("is_last_business_day", False):
            if pattern in ["review", "deployment"]:
                base_score += 0.2
        
        # Season adjustments
        if temporal_context.get("is_holiday_season", False):
            if pattern in ["urgent", "communication"]:
                base_score += 0.2
        if temporal_context.get("is_summer_break", False):
            if pattern in ["documentation", "improvement"]:
                base_score += 0.2
        
        return min(max(base_score, 0.0), 1.0)

    def _calculate_pattern_urgency(self, pattern: str, temporal_context: Dict[str, Any]) -> float:
        """Calculate urgency score for a pattern based on temporal context."""
        urgency = 0.5
        
        # Time-based urgency
        if temporal_context.get("is_peak_productivity", False):
            if pattern in ["urgent", "development", "communication"]:
                urgency += 0.2
        if temporal_context.get("is_low_energy", False):
            if pattern in ["urgent", "communication"]:
                urgency += 0.1
        
        # Calendar-based urgency
        if temporal_context.get("is_last_business_day", False):
            if pattern in ["urgent", "deployment"]:
                urgency += 0.3
        if temporal_context.get("is_month_end", False):
            if pattern in ["urgent", "review"]:
                urgency += 0.2
        
        return min(max(urgency, 0.0), 1.0)

    def _find_matching_sequences(self, task_id: str) -> List[Dict[str, Any]]:
        """Find sequences that include this task."""
        matches = []
        for sequence in self.domain_knowledge["successful_sequences"]:
            if task_id in sequence["tasks"]:
                # Calculate sequence relevance based on position and age
                position = sequence["tasks"].index(task_id)
                age = (datetime.now() - datetime.fromisoformat(sequence["timestamp"])).days
                relevance = (1.0 / (position + 1)) * (self.sequence_decay ** age)
                
                # Get temporal performance for the sequence
                temporal_performance = self._get_sequence_temporal_performance(sequence, sequence.get("temporal_context", {}))
                
                # Get transition probability if this is not the first task in the sequence
                transition_probability = 0.5
                if position > 0:
                    current_pattern = sequence["patterns"][position - 1]
                    next_pattern = sequence["patterns"][position]
                    transition_probability = self._get_transition_probability(
                        current_pattern,
                        next_pattern,
                        sequence.get("temporal_context", {})
                    )
                
                matches.append({
                    "sequence": sequence,
                    "relevance": relevance,
                    "performance": temporal_performance,
                    "transition_probability": transition_probability,
                    "position": position,
                    "age": age
                })
        return matches

    def _get_sequence_temporal_performance(self, sequence: Dict[str, Any], temporal_context: Dict[str, Any]) -> float:
        """Get the performance score for a sequence in the current temporal context."""
        sequence_id = sequence.get("id", str(hash(str(sequence))))
        performance_scores = []
        
        # Get time period performance
        time_period = temporal_context.get("time_period", "")
        if time_period:
            perf = self.domain_knowledge["pattern_sequences"]["temporal_performance"]["time_periods"][time_period].get(sequence_id, 0.5)
            performance_scores.append(perf)
        
        # Get business calendar performance
        if temporal_context.get("is_first_business_day", False):
            perf = self.domain_knowledge["pattern_sequences"]["temporal_performance"]["business_calendar"]["first_business_day"].get(sequence_id, 0.5)
            performance_scores.append(perf)
        
        if temporal_context.get("is_last_business_day", False):
            perf = self.domain_knowledge["pattern_sequences"]["temporal_performance"]["business_calendar"]["last_business_day"].get(sequence_id, 0.5)
            performance_scores.append(perf)
        
        if temporal_context.get("is_holiday", False):
            perf = self.domain_knowledge["pattern_sequences"]["temporal_performance"]["business_calendar"]["holiday"].get(sequence_id, 0.5)
            performance_scores.append(perf)
        
        if temporal_context.get("is_weekend", False):
            perf = self.domain_knowledge["pattern_sequences"]["temporal_performance"]["business_calendar"]["weekend"].get(sequence_id, 0.5)
            performance_scores.append(perf)
        
        # Get season performance
        season = temporal_context.get("season", "")
        if season:
            perf = self.domain_knowledge["pattern_sequences"]["temporal_performance"]["seasons"][season].get(sequence_id, 0.5)
            performance_scores.append(perf)
        
        # Get work schedule performance
        if temporal_context.get("is_typical_work_hours", False):
            perf = self.domain_knowledge["pattern_sequences"]["temporal_performance"]["work_schedule"]["work_hours"].get(sequence_id, 0.5)
            performance_scores.append(perf)
        
        if temporal_context.get("is_peak_productivity", False):
            perf = self.domain_knowledge["pattern_sequences"]["temporal_performance"]["work_schedule"]["peak_hours"].get(sequence_id, 0.5)
            performance_scores.append(perf)
        
        if temporal_context.get("is_low_energy", False):
            perf = self.domain_knowledge["pattern_sequences"]["temporal_performance"]["work_schedule"]["low_energy"].get(sequence_id, 0.5)
            performance_scores.append(perf)
        
        if temporal_context.get("is_global_collaboration_hours", False):
            perf = self.domain_knowledge["pattern_sequences"]["temporal_performance"]["work_schedule"]["global_collaboration"].get(sequence_id, 0.5)
            performance_scores.append(perf)
        
        # Calculate weighted average of performance scores
        if performance_scores:
            weights = [1.0] * len(performance_scores)  # Equal weights for now
            return np.average(performance_scores, weights=weights)
        
        return 0.5  # Default performance if no temporal context matches

    def update_sequence_performance(self, sequence: Dict[str, Any], success: bool, temporal_context: Dict[str, Any]):
        """Update sequence performance based on temporal context."""
        sequence_id = sequence.get("id", str(hash(str(sequence))))
        
        # Update time period performance
        time_period = temporal_context.get("time_period", "")
        if time_period:
            current_perf = self.domain_knowledge["pattern_sequences"]["temporal_performance"]["time_periods"][time_period].get(sequence_id, 0.5)
            new_perf = (0.9 * current_perf) + (0.1 * (1.0 if success else 0.0))
            self.domain_knowledge["pattern_sequences"]["temporal_performance"]["time_periods"][time_period][sequence_id] = new_perf
        
        # Update business calendar performance
        if temporal_context.get("is_first_business_day", False):
            current_perf = self.domain_knowledge["pattern_sequences"]["temporal_performance"]["business_calendar"]["first_business_day"].get(sequence_id, 0.5)
            new_perf = (0.9 * current_perf) + (0.1 * (1.0 if success else 0.0))
            self.domain_knowledge["pattern_sequences"]["temporal_performance"]["business_calendar"]["first_business_day"][sequence_id] = new_perf
        
        if temporal_context.get("is_last_business_day", False):
            current_perf = self.domain_knowledge["pattern_sequences"]["temporal_performance"]["business_calendar"]["last_business_day"].get(sequence_id, 0.5)
            new_perf = (0.9 * current_perf) + (0.1 * (1.0 if success else 0.0))
            self.domain_knowledge["pattern_sequences"]["temporal_performance"]["business_calendar"]["last_business_day"][sequence_id] = new_perf
        
        if temporal_context.get("is_holiday", False):
            current_perf = self.domain_knowledge["pattern_sequences"]["temporal_performance"]["business_calendar"]["holiday"].get(sequence_id, 0.5)
            new_perf = (0.9 * current_perf) + (0.1 * (1.0 if success else 0.0))
            self.domain_knowledge["pattern_sequences"]["temporal_performance"]["business_calendar"]["holiday"][sequence_id] = new_perf
        
        if temporal_context.get("is_weekend", False):
            current_perf = self.domain_knowledge["pattern_sequences"]["temporal_performance"]["business_calendar"]["weekend"].get(sequence_id, 0.5)
            new_perf = (0.9 * current_perf) + (0.1 * (1.0 if success else 0.0))
            self.domain_knowledge["pattern_sequences"]["temporal_performance"]["business_calendar"]["weekend"][sequence_id] = new_perf
        
        # Update season performance
        season = temporal_context.get("season", "")
        if season:
            current_perf = self.domain_knowledge["pattern_sequences"]["temporal_performance"]["seasons"][season].get(sequence_id, 0.5)
            new_perf = (0.9 * current_perf) + (0.1 * (1.0 if success else 0.0))
            self.domain_knowledge["pattern_sequences"]["temporal_performance"]["seasons"][season][sequence_id] = new_perf
        
        # Update work schedule performance
        if temporal_context.get("is_typical_work_hours", False):
            current_perf = self.domain_knowledge["pattern_sequences"]["temporal_performance"]["work_schedule"]["work_hours"].get(sequence_id, 0.5)
            new_perf = (0.9 * current_perf) + (0.1 * (1.0 if success else 0.0))
            self.domain_knowledge["pattern_sequences"]["temporal_performance"]["work_schedule"]["work_hours"][sequence_id] = new_perf
        
        if temporal_context.get("is_peak_productivity", False):
            current_perf = self.domain_knowledge["pattern_sequences"]["temporal_performance"]["work_schedule"]["peak_hours"].get(sequence_id, 0.5)
            new_perf = (0.9 * current_perf) + (0.1 * (1.0 if success else 0.0))
            self.domain_knowledge["pattern_sequences"]["temporal_performance"]["work_schedule"]["peak_hours"][sequence_id] = new_perf
        
        if temporal_context.get("is_low_energy", False):
            current_perf = self.domain_knowledge["pattern_sequences"]["temporal_performance"]["work_schedule"]["low_energy"].get(sequence_id, 0.5)
            new_perf = (0.9 * current_perf) + (0.1 * (1.0 if success else 0.0))
            self.domain_knowledge["pattern_sequences"]["temporal_performance"]["work_schedule"]["low_energy"][sequence_id] = new_perf
        
        if temporal_context.get("is_global_collaboration_hours", False):
            current_perf = self.domain_knowledge["pattern_sequences"]["temporal_performance"]["work_schedule"]["global_collaboration"].get(sequence_id, 0.5)
            new_perf = (0.9 * current_perf) + (0.1 * (1.0 if success else 0.0))
            self.domain_knowledge["pattern_sequences"]["temporal_performance"]["work_schedule"]["global_collaboration"][sequence_id] = new_perf
        
        self._save_domain_knowledge()

    def _calculate_domain_score(
        self, task: Dict[str, Any], state: Dict[str, Any]
    ) -> float:
        """Calculate domain score for a task."""
        score = 0.0
        weights = {"pattern": 0.3, "rule": 0.3, "sequence": 0.4}

        # Check task patterns
        task_text = f"{task.get('title', '')} {task.get('description', '')}".lower()
        pattern_matches = self._find_matching_patterns(task_text, state.get("temporal", {}))
        if pattern_matches:
            pattern_score = max(
                match["score"] * match["performance"]
                for match in pattern_matches
            )
            score += weights["pattern"] * pattern_score

        # Check domain rules
        rule_matches = []
        for rule in self.domain_knowledge["domain_rules"]:
            if self._evaluate_rule(rule, task, state):
                rule_score = rule.get("score", self.min_rule_score)
                rule_performance = self.domain_knowledge["rule_performance"].get(rule["id"], 0.5)
                rule_matches.append(rule_score * rule_performance)
        
        if rule_matches:
            score += weights["rule"] * max(rule_matches)

        # Check successful sequences
        task_id = task.get("id", str(hash(str(task))))
        sequence_matches = self._find_matching_sequences(task_id)
        if sequence_matches:
            sequence_score = max(
                match["relevance"] * match["performance"] * match["transition_probability"]
                for match in sequence_matches
            )
            score += weights["sequence"] * sequence_score

        return min(score, 1.0)

    def update_knowledge(self, task: Dict[str, Any], success: bool):
        """Update domain knowledge based on task outcome."""
        task_id = task.get("id", str(hash(str(task))))
        task_text = f"{task.get('title', '')} {task.get('description', '')}".lower()
        
        # Update pattern performance
        for pattern in self.domain_knowledge["task_patterns"]:
            if pattern in task_text:
                current_perf = self.domain_knowledge["pattern_performance"].get(pattern, 0.5)
                new_perf = (0.9 * current_perf) + (0.1 * (1.0 if success else 0.0))
                self.domain_knowledge["pattern_performance"][pattern] = new_perf
        
        # Update rule performance
        for rule in self.domain_knowledge["domain_rules"]:
            if self._evaluate_rule(rule, task, {}):  # Empty state for rule evaluation
                current_perf = self.domain_knowledge["rule_performance"].get(rule["id"], 0.5)
                new_perf = (0.9 * current_perf) + (0.1 * (1.0 if success else 0.0))
                self.domain_knowledge["rule_performance"][rule["id"]] = new_perf
        
        # Update sequence performance
        for sequence in self.domain_knowledge["successful_sequences"]:
            if task_id in sequence["tasks"]:
                current_perf = self.domain_knowledge["sequence_performance"].get(str(sequence["id"]), 0.5)
                new_perf = (0.9 * current_perf) + (0.1 * (1.0 if success else 0.0))
                self.domain_knowledge["sequence_performance"][str(sequence["id"])] = new_perf
        
        self._save_domain_knowledge()

    def calculate_state_compatibility(self, state: Dict[str, Any]) -> float:
        """Calculate state compatibility for domain-guided exploration."""
        # Consider both knowledge base size and performance
        pattern_count = len(self.domain_knowledge["task_patterns"])
        rule_count = len(self.domain_knowledge["domain_rules"])
        sequence_count = len(self.domain_knowledge["successful_sequences"])
        
        # Calculate average performance
        avg_pattern_perf = np.mean(list(self.domain_knowledge["pattern_performance"].values())) if self.domain_knowledge["pattern_performance"] else 0.5
        avg_rule_perf = np.mean(list(self.domain_knowledge["rule_performance"].values())) if self.domain_knowledge["rule_performance"] else 0.5
        avg_sequence_perf = np.mean(list(self.domain_knowledge["sequence_performance"].values())) if self.domain_knowledge["sequence_performance"] else 0.5
        
        # Combine metrics
        knowledge_size = (pattern_count + rule_count + sequence_count) / 30.0
        avg_performance = (avg_pattern_perf + avg_rule_perf + avg_sequence_perf) / 3.0
        
        return min(knowledge_size * avg_performance, 1.0)

    def cleanup(self):
        """Clean up resources."""
        self._save_domain_knowledge()

    def _get_time_period(self, date: datetime) -> str:
        """Get the time period category with enhanced granularity."""
        hour = date.hour
        if hour < 5:
            return "very_early_morning"
        elif hour < 8:
            return "early_morning"
        elif hour < 11:
            return "morning"
        elif hour < 14:
            return "early_afternoon"
        elif hour < 17:
            return "late_afternoon"
        elif hour < 20:
            return "evening"
        elif hour < 23:
            return "late_evening"
        else:
            return "night"

    def _get_time_of_day_category(self, date: datetime) -> str:
        """Get the time of day category with more granularity."""
        hour = date.hour
        if hour < 5:
            return "very_early_morning"
        elif hour < 8:
            return "early_morning"
        elif hour < 11:
            return "morning"
        elif hour < 14:
            return "early_afternoon"
        elif hour < 17:
            return "late_afternoon"
        elif hour < 20:
            return "evening"
        elif hour < 23:
            return "late_evening"
        else:
            return "night"

    def _is_first_business_day(self, date: datetime) -> bool:
        """Check if this is the first business day of the month."""
        if date.day == 1:
            return True
        if date.day == 2 and date.weekday() == 0:  # Monday after weekend
            return True
        if date.day == 3 and date.weekday() == 0 and self._is_holiday(date.replace(day=1)):
            return True
        return False

    def _is_last_business_day(self, date: datetime) -> bool:
        """Check if this is the last business day of the month."""
        next_day = date + timedelta(days=1)
        if next_day.month != date.month:
            return True
        if next_day.weekday() == 0 and date.weekday() == 4:  # Friday before weekend
            return True
        return False

    def _is_holiday_season(self, date: datetime) -> bool:
        """Check if current date is during holiday season."""
        month = date.month
        if month == 12:  # December
            return True
        if month == 11 and date.day >= 20:  # Late November
            return True
        if month == 1 and date.day <= 7:  # Early January
            return True
        return False

    def _is_summer_break(self, date: datetime) -> bool:
        """Check if current date is during summer break."""
        month = date.month
        if month in [6, 7, 8]:  # Summer months
            return True
        if month == 5 and date.day >= 25:  # Late May
            return True
        if month == 9 and date.day <= 5:  # Early September
            return True
        return False

    def _is_typical_work_hours(self, date: datetime) -> bool:
        """Check if current time is during typical work hours."""
        return 9 <= date.hour < 17 and date.weekday() < 5

    def _is_peak_productivity_hours(self, date: datetime) -> bool:
        """Check if current time is during peak productivity hours."""
        hour = date.hour
        return (9 <= hour < 11) or (14 <= hour < 16)

    def _is_low_energy_hours(self, date: datetime) -> bool:
        """Check if current time is during low energy hours."""
        hour = date.hour
        return (12 <= hour < 14) or (16 <= hour < 17)

    def _is_local_business_hours(self, date: datetime) -> bool:
        """Check if current time is during local business hours."""
        return 9 <= date.hour < 17 and date.weekday() < 5

    def _is_global_collaboration_hours(self, date: datetime) -> bool:
        """Check if current time is during global collaboration hours."""
        hour = date.hour
        return (8 <= hour < 10) or (15 <= hour < 17)  # Overlap with other time zones

    def _get_productivity_score(self, date: datetime) -> float:
        """Calculate a productivity score based on time of day."""
        hour = date.hour
        if 9 <= hour < 11:
            return 0.9  # Morning peak
        elif 14 <= hour < 16:
            return 0.8  # Afternoon peak
        elif 11 <= hour < 12:
            return 0.7  # Late morning
        elif 16 <= hour < 17:
            return 0.6  # Late afternoon
        elif 8 <= hour < 9:
            return 0.5  # Early morning
        elif 17 <= hour < 18:
            return 0.4  # Early evening
        else:
            return 0.3  # Other times

    def _get_urgency_score(self, task: Dict[str, Any], current_time: datetime) -> float:
        """Calculate an urgency score based on due date and current time."""
        if "due_date" not in task:
            return 0.5  # No due date, neutral urgency
            
        due_date = datetime.fromisoformat(task["due_date"])
        time_until_due = (due_date - current_time).total_seconds()
        
        if time_until_due < 0:
            return 1.0  # Overdue
        elif time_until_due < 3600:
            return 0.9  # Due within an hour
        elif time_until_due < 86400:
            return 0.8  # Due within a day
        elif time_until_due < 604800:
            return 0.7  # Due within a week
        elif time_until_due < 2592000:
            return 0.6  # Due within a month
        else:
            return 0.5  # Due in more than a month

    def _get_time_based_priority(self, task: Dict[str, Any], state: Dict[str, Any]) -> float:
        """Calculate a time-based priority score for the task."""
        current_time = datetime.now()
        
        # Base priority from task
        base_priority = task.get("priority", 1) / 5.0
        
        # Time-based factors
        productivity_score = self._get_productivity_score(current_time)
        urgency_score = self._get_urgency_score(task, current_time)
        
        # Temporal context factors
        is_work_hours = self._is_typical_work_hours(current_time)
        is_peak_hours = self._is_peak_productivity_hours(current_time)
        is_low_energy = self._is_low_energy_hours(current_time)
        
        # Calculate weighted score
        weights = {
            "base_priority": 0.4,
            "productivity": 0.2,
            "urgency": 0.3,
            "time_context": 0.1
        }
        
        time_context_bonus = 0.0
        if is_peak_hours:
            time_context_bonus += 0.2
        if is_work_hours:
            time_context_bonus += 0.1
        if is_low_energy:
            time_context_bonus -= 0.1
            
        final_score = (
            weights["base_priority"] * base_priority +
            weights["productivity"] * productivity_score +
            weights["urgency"] * urgency_score +
            weights["time_context"] * time_context_bonus
        )
        
        return min(max(final_score, 0.0), 1.0)

    def _initialize_pattern_performance(self):
        """Initialize pattern performance tracking."""
        if "pattern_performance" not in self.domain_knowledge:
            self.domain_knowledge["pattern_performance"] = {}
        
        if "temporal_pattern_performance" not in self.domain_knowledge:
            self.domain_knowledge["temporal_pattern_performance"] = {
                "time_periods": {},
                "business_calendar": {},
                "seasons": {},
                "work_schedule": {}
            }
        
        if "pattern_sequences" not in self.domain_knowledge:
            self.domain_knowledge["pattern_sequences"] = {
                "sequences": [],
                "temporal_performance": {
                    "time_periods": {},
                    "business_calendar": {},
                    "seasons": {},
                    "work_schedule": {}
                },
                "transition_probabilities": {
                    "time_periods": {},
                    "business_calendar": {},
                    "seasons": {},
                    "work_schedule": {}
                },
                "recommendations": {
                    "successful_patterns": {},
                    "temporal_patterns": {},
                    "sequence_templates": [],
                    "performance_history": {}
                }
            }
        
        # Initialize time period performance and transitions
        for period in ["early_morning", "morning", "afternoon", "evening", "night"]:
            if period not in self.domain_knowledge["temporal_pattern_performance"]["time_periods"]:
                self.domain_knowledge["temporal_pattern_performance"]["time_periods"][period] = {}
            if period not in self.domain_knowledge["pattern_sequences"]["temporal_performance"]["time_periods"]:
                self.domain_knowledge["pattern_sequences"]["temporal_performance"]["time_periods"][period] = {}
            if period not in self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["time_periods"]:
                self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["time_periods"][period] = {}
            if period not in self.domain_knowledge["pattern_sequences"]["recommendations"]["temporal_patterns"]:
                self.domain_knowledge["pattern_sequences"]["recommendations"]["temporal_patterns"][period] = {}
        
        # Initialize business calendar performance and transitions
        for event in ["first_business_day", "last_business_day", "holiday", "weekend"]:
            if event not in self.domain_knowledge["temporal_pattern_performance"]["business_calendar"]:
                self.domain_knowledge["temporal_pattern_performance"]["business_calendar"][event] = {}
            if event not in self.domain_knowledge["pattern_sequences"]["temporal_performance"]["business_calendar"]:
                self.domain_knowledge["pattern_sequences"]["temporal_performance"]["business_calendar"][event] = {}
            if event not in self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["business_calendar"]:
                self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["business_calendar"][event] = {}
            if event not in self.domain_knowledge["pattern_sequences"]["recommendations"]["temporal_patterns"]:
                self.domain_knowledge["pattern_sequences"]["recommendations"]["temporal_patterns"][event] = {}
        
        # Initialize season performance and transitions
        for season in ["spring", "summer", "fall", "winter"]:
            if season not in self.domain_knowledge["temporal_pattern_performance"]["seasons"]:
                self.domain_knowledge["temporal_pattern_performance"]["seasons"][season] = {}
            if season not in self.domain_knowledge["pattern_sequences"]["temporal_performance"]["seasons"]:
                self.domain_knowledge["pattern_sequences"]["temporal_performance"]["seasons"][season] = {}
            if season not in self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["seasons"]:
                self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["seasons"][season] = {}
            if season not in self.domain_knowledge["pattern_sequences"]["recommendations"]["temporal_patterns"]:
                self.domain_knowledge["pattern_sequences"]["recommendations"]["temporal_patterns"][season] = {}
        
        # Initialize work schedule performance and transitions
        for schedule in ["work_hours", "peak_hours", "low_energy", "global_collaboration"]:
            if schedule not in self.domain_knowledge["temporal_pattern_performance"]["work_schedule"]:
                self.domain_knowledge["temporal_pattern_performance"]["work_schedule"][schedule] = {}
            if schedule not in self.domain_knowledge["pattern_sequences"]["temporal_performance"]["work_schedule"]:
                self.domain_knowledge["pattern_sequences"]["temporal_performance"]["work_schedule"][schedule] = {}
            if schedule not in self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["work_schedule"]:
                self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["work_schedule"][schedule] = {}
            if schedule not in self.domain_knowledge["pattern_sequences"]["recommendations"]["temporal_patterns"]:
                self.domain_knowledge["pattern_sequences"]["recommendations"]["temporal_patterns"][schedule] = {}
        
        self._save_domain_knowledge()

    def _update_transition_probabilities(self, current_pattern: str, next_pattern: str, success: bool, temporal_context: Dict[str, Any]):
        """Update transition probabilities between patterns based on temporal context."""
        # Update time period transitions
        time_period = temporal_context.get("time_period", "")
        if time_period:
            transitions = self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["time_periods"][time_period]
            if current_pattern not in transitions:
                transitions[current_pattern] = {}
            if next_pattern not in transitions[current_pattern]:
                transitions[current_pattern][next_pattern] = {"success": 0, "total": 0}
            transitions[current_pattern][next_pattern]["total"] += 1
            if success:
                transitions[current_pattern][next_pattern]["success"] += 1
        
        # Update business calendar transitions
        if temporal_context.get("is_first_business_day", False):
            transitions = self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["business_calendar"]["first_business_day"]
            self._update_transition(transitions, current_pattern, next_pattern, success)
        
        if temporal_context.get("is_last_business_day", False):
            transitions = self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["business_calendar"]["last_business_day"]
            self._update_transition(transitions, current_pattern, next_pattern, success)
        
        if temporal_context.get("is_holiday", False):
            transitions = self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["business_calendar"]["holiday"]
            self._update_transition(transitions, current_pattern, next_pattern, success)
        
        if temporal_context.get("is_weekend", False):
            transitions = self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["business_calendar"]["weekend"]
            self._update_transition(transitions, current_pattern, next_pattern, success)
        
        # Update season transitions
        season = temporal_context.get("season", "")
        if season:
            transitions = self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["seasons"][season]
            self._update_transition(transitions, current_pattern, next_pattern, success)
        
        # Update work schedule transitions
        if temporal_context.get("is_typical_work_hours", False):
            transitions = self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["work_schedule"]["work_hours"]
            self._update_transition(transitions, current_pattern, next_pattern, success)
        
        if temporal_context.get("is_peak_productivity", False):
            transitions = self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["work_schedule"]["peak_hours"]
            self._update_transition(transitions, current_pattern, next_pattern, success)
        
        if temporal_context.get("is_low_energy", False):
            transitions = self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["work_schedule"]["low_energy"]
            self._update_transition(transitions, current_pattern, next_pattern, success)
        
        if temporal_context.get("is_global_collaboration_hours", False):
            transitions = self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["work_schedule"]["global_collaboration"]
            self._update_transition(transitions, current_pattern, next_pattern, success)
        
        self._save_domain_knowledge()

    def _update_transition(self, transitions: Dict[str, Any], current_pattern: str, next_pattern: str, success: bool):
        """Update a single transition probability."""
        if current_pattern not in transitions:
            transitions[current_pattern] = {}
        if next_pattern not in transitions[current_pattern]:
            transitions[current_pattern][next_pattern] = {"success": 0, "total": 0}
        transitions[current_pattern][next_pattern]["total"] += 1
        if success:
            transitions[current_pattern][next_pattern]["success"] += 1

    def _get_transition_probability(self, current_pattern: str, next_pattern: str, temporal_context: Dict[str, Any]) -> float:
        """Get the probability of transitioning from current_pattern to next_pattern in the given temporal context."""
        probabilities = []
        
        # Get time period transition probability
        time_period = temporal_context.get("time_period", "")
        if time_period:
            transitions = self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["time_periods"][time_period]
            prob = self._calculate_transition_probability(transitions, current_pattern, next_pattern)
            probabilities.append(prob)
        
        # Get business calendar transition probabilities
        if temporal_context.get("is_first_business_day", False):
            transitions = self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["business_calendar"]["first_business_day"]
            prob = self._calculate_transition_probability(transitions, current_pattern, next_pattern)
            probabilities.append(prob)
        
        if temporal_context.get("is_last_business_day", False):
            transitions = self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["business_calendar"]["last_business_day"]
            prob = self._calculate_transition_probability(transitions, current_pattern, next_pattern)
            probabilities.append(prob)
        
        if temporal_context.get("is_holiday", False):
            transitions = self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["business_calendar"]["holiday"]
            prob = self._calculate_transition_probability(transitions, current_pattern, next_pattern)
            probabilities.append(prob)
        
        if temporal_context.get("is_weekend", False):
            transitions = self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["business_calendar"]["weekend"]
            prob = self._calculate_transition_probability(transitions, current_pattern, next_pattern)
            probabilities.append(prob)
        
        # Get season transition probability
        season = temporal_context.get("season", "")
        if season:
            transitions = self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["seasons"][season]
            prob = self._calculate_transition_probability(transitions, current_pattern, next_pattern)
            probabilities.append(prob)
        
        # Get work schedule transition probabilities
        if temporal_context.get("is_typical_work_hours", False):
            transitions = self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["work_schedule"]["work_hours"]
            prob = self._calculate_transition_probability(transitions, current_pattern, next_pattern)
            probabilities.append(prob)
        
        if temporal_context.get("is_peak_productivity", False):
            transitions = self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["work_schedule"]["peak_hours"]
            prob = self._calculate_transition_probability(transitions, current_pattern, next_pattern)
            probabilities.append(prob)
        
        if temporal_context.get("is_low_energy", False):
            transitions = self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["work_schedule"]["low_energy"]
            prob = self._calculate_transition_probability(transitions, current_pattern, next_pattern)
            probabilities.append(prob)
        
        if temporal_context.get("is_global_collaboration_hours", False):
            transitions = self.domain_knowledge["pattern_sequences"]["transition_probabilities"]["work_schedule"]["global_collaboration"]
            prob = self._calculate_transition_probability(transitions, current_pattern, next_pattern)
            probabilities.append(prob)
        
        # Calculate weighted average of probabilities
        if probabilities:
            weights = [1.0] * len(probabilities)  # Equal weights for now
            return np.average(probabilities, weights=weights)
        
        return 0.5  # Default probability if no temporal context matches

    def _calculate_transition_probability(self, transitions: Dict[str, Any], current_pattern: str, next_pattern: str) -> float:
        """Calculate the probability of transitioning from current_pattern to next_pattern."""
        if current_pattern not in transitions or next_pattern not in transitions[current_pattern]:
            return 0.5  # Default probability if no transition data
        
        transition = transitions[current_pattern][next_pattern]
        if transition["total"] == 0:
            return 0.5
        
        return transition["success"] / transition["total"]

    def _find_matching_sequences(self, task_id: str) -> List[Dict[str, Any]]:
        """Find sequences that include this task."""
        matches = []
        for sequence in self.domain_knowledge["successful_sequences"]:
            if task_id in sequence["tasks"]:
                # Calculate sequence relevance based on position and age
                position = sequence["tasks"].index(task_id)
                age = (datetime.now() - datetime.fromisoformat(sequence["timestamp"])).days
                relevance = (1.0 / (position + 1)) * (self.sequence_decay ** age)
                
                # Get temporal performance for the sequence
                temporal_performance = self._get_sequence_temporal_performance(sequence, sequence.get("temporal_context", {}))
                
                # Get transition probability if this is not the first task in the sequence
                transition_probability = 0.5
                if position > 0:
                    current_pattern = sequence["patterns"][position - 1]
                    next_pattern = sequence["patterns"][position]
                    transition_probability = self._get_transition_probability(
                        current_pattern,
                        next_pattern,
                        sequence.get("temporal_context", {})
                    )
                
                matches.append({
                    "sequence": sequence,
                    "relevance": relevance,
                    "performance": temporal_performance,
                    "transition_probability": transition_probability,
                    "position": position,
                    "age": age
                })
        return matches

    def _analyze_sequence_patterns(self, sequence: List[Tuple[Dict[str, Any], str]], temporal_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sequence patterns with enhanced metrics and visualization."""
        # Initialize analysis dictionary
        analysis = {
            "sequence_metrics": {},
            "pattern_analysis": {},
            "temporal_analysis": {},
            "relationship_analysis": {},
            "visualization_data": {},
            "recommendations": {}
        }

        # Extract tasks and patterns
        tasks = [task for task, _ in sequence]
        patterns = [pattern for _, pattern in sequence]

        # Calculate sequence metrics
        analysis["sequence_metrics"] = {
            "pattern_diversity": self._calculate_pattern_diversity(sequence),
            "temporal_consistency": self._calculate_temporal_flow(sequence, temporal_context),
            "resource_utilization": self._calculate_resource_efficiency(sequence),
            "cognitive_load": sum(self._calculate_cognitive_complexity(task) for task in tasks) / len(tasks),
            "context_switching": self._calculate_context_switching(sequence),
            "dependency_satisfaction": self._calculate_dependency_satisfaction([], tasks)
        }

        # Analyze pattern transitions
        transitions = []
        for i in range(len(sequence) - 1):
            curr_task, curr_pattern = sequence[i]
            next_task, next_pattern = sequence[i + 1]
            
            transition = {
                "transition_quality": self._calculate_transition_quality(
                    curr_task, next_task, curr_pattern, next_pattern, temporal_context
                ),
                "knowledge_transfer": self._calculate_knowledge_transfer(
                    curr_task, next_task, curr_pattern, next_pattern
                ),
                "adaptability": self._calculate_adaptability(next_task, temporal_context)
            }
            transitions.append(transition)

        analysis["pattern_analysis"] = {
            "transitions": transitions,
            "avg_transition_quality": sum(t["transition_quality"] for t in transitions) / len(transitions),
            "avg_knowledge_transfer": sum(t["knowledge_transfer"] for t in transitions) / len(transitions),
            "avg_adaptability": sum(t["adaptability"] for t in transitions) / len(transitions)
        }

        # Analyze temporal patterns
        analysis["temporal_analysis"] = {
            "consistency": self._calculate_temporal_flow(sequence, temporal_context),
            "efficiency": sum(self._calculate_task_flow_efficiency(
                sequence[i][0], sequence[i+1][0], temporal_context
            ) for i in range(len(sequence)-1)) / (len(sequence)-1),
            "adaptability": sum(self._calculate_temporal_resilience(
                sequence[i][0], sequence[i+1][0], temporal_context
            ) for i in range(len(sequence)-1)) / (len(sequence)-1)
        }

        # Analyze task relationships
        relationships = []
        for i in range(len(sequence) - 1):
            curr_task, _ = sequence[i]
            next_task, _ = sequence[i + 1]
            relationship = self._analyze_task_relationship(curr_task, next_task)
            relationships.append(relationship)

        analysis["relationship_analysis"] = {
            "relationships": relationships,
            "avg_strength": sum(r["strength"] for r in relationships) / len(relationships),
            "avg_quality": sum(r["quality"] for r in relationships) / len(relationships),
            "avg_adaptability": sum(r["adaptability"] for r in relationships) / len(relationships)
        }

        # Generate visualization data
        analysis["visualization_data"] = {
            "pattern_flow": {
                "nodes": [{"id": p, "type": "pattern"} for p in set(patterns)],
                "edges": [{"source": patterns[i], "target": patterns[i+1], 
                          "weight": transitions[i]["transition_quality"]} 
                         for i in range(len(patterns)-1)]
            },
            "temporal_heatmap": {
                "times": [task.get("timestamp", "") for task, _ in sequence],
                "patterns": patterns,
                "values": [self._calculate_temporal_flow([sequence[i]], temporal_context) 
                          for i in range(len(sequence))]
            },
            "resource_distribution": {
                "resources": list(set(task.get("resources", []) for task, _ in sequence)),
                "utilization": [self._calculate_resource_efficiency([sequence[i]]) 
                              for i in range(len(sequence))]
            },
            "cognitive_flow": {
                "tasks": [task["id"] for task, _ in sequence],
                "cognitive_load": [self._calculate_cognitive_complexity(task) for task, _ in sequence],
                "context_switches": [self._calculate_context_switch(sequence[i][0], sequence[i+1][0]) 
                                   for i in range(len(sequence)-1)]
            }
        }

        # Generate pattern recommendations
        recommended_sequences = self._get_sequence_recommendations(sequence, temporal_context)
        analysis["recommendations"] = {
            "next_patterns": [p for p, _ in recommended_sequences[0]] if recommended_sequences else [],
            "alternative_sequences": recommended_sequences[1:],
            "optimization_suggestions": [
                {
                    "type": "transition_optimization",
                    "suggestion": "Optimize transitions between patterns",
                    "score": analysis["pattern_analysis"]["avg_transition_quality"]
                },
                {
                    "type": "cognitive_load_balancing",
                    "suggestion": "Balance cognitive load across sequence",
                    "score": analysis["sequence_metrics"]["cognitive_load"]
                },
                {
                    "type": "resource_allocation",
                    "suggestion": "Optimize resource allocation",
                    "score": analysis["sequence_metrics"]["resource_utilization"]
                }
            ]
        }

        return analysis

    def _calculate_pattern_diversity(self, sequence: List[Tuple[Dict[str, Any], str]]) -> float:
        """Calculate the diversity of patterns in a sequence."""
        if not sequence:
            return 0.0

        # Get unique patterns and their frequencies
        patterns = [pattern for _, pattern in sequence]
        unique_patterns = set(patterns)
        pattern_counts = {p: patterns.count(p) for p in unique_patterns}
        
        # Calculate Shannon diversity index
        total_patterns = len(patterns)
        diversity = 0.0
        for count in pattern_counts.values():
            p = count / total_patterns
            diversity -= p * np.log(p)
        
        # Normalize to 0-1 range
        max_diversity = np.log(len(unique_patterns)) if len(unique_patterns) > 1 else 1.0
        normalized_diversity = diversity / max_diversity if max_diversity > 0 else 0.0
        
        return normalized_diversity

    def _calculate_context_switching(self, sequence: List[Tuple[Dict[str, Any], str]]) -> float:
        """Calculate the context switching cost for a sequence."""
        if len(sequence) < 2:
            return 0.0

        total_cost = 0.0
        for i in range(len(sequence) - 1):
            curr_task, curr_pattern = sequence[i]
            next_task, next_pattern = sequence[i + 1]
            
            # Calculate base switching cost
            base_cost = 0.0 if curr_pattern == next_pattern else 0.5
            
            # Adjust based on task properties
            cognitive_diff = abs(
                self._calculate_cognitive_complexity(curr_task) -
                self._calculate_cognitive_complexity(next_task)
            )
            
            resource_diff = 1.0 - self._calculate_resource_overlap(curr_task, next_task)
            context_diff = 1.0 - self._calculate_context_continuity(curr_task, next_task)
            
            # Combine costs with weights
            switch_cost = (
                0.3 * base_cost +
                0.3 * cognitive_diff +
                0.2 * resource_diff +
                0.2 * context_diff
            )
            
            total_cost += switch_cost
        
        # Normalize to 0-1 range
        avg_cost = total_cost / (len(sequence) - 1)
        return avg_cost

    def _calculate_transition_quality(
        self,
        current_task: Dict[str, Any],
        next_task: Dict[str, Any],
        current_pattern: str,
        next_pattern: str,
        temporal_context: Dict[str, Any]
    ) -> float:
        """Calculate the quality of a transition between tasks."""
        # Calculate pattern transition probability
        pattern_prob = self._get_transition_probability(current_pattern, next_pattern, temporal_context)
        
        # Calculate task relationship strength
        relationship = self._analyze_task_relationship(current_task, next_task)
        relationship_strength = relationship["relationship_strength"]
        
        # Calculate temporal compatibility
        temporal_compatibility = self._calculate_temporal_compatibility(
            current_task, next_task, temporal_context
        )
        
        # Calculate resource continuity
        resource_continuity = self._calculate_resource_continuity(current_task, next_task)
        
        # Calculate cognitive load change
        current_load = self._calculate_cognitive_load(current_task)
        next_load = self._calculate_cognitive_load(next_task)
        load_change = abs(current_load - next_load)
        
        # Combine factors with weights
        quality = (
            pattern_prob * 0.3 +
            relationship_strength * 0.2 +
            temporal_compatibility * 0.2 +
            resource_continuity * 0.2 +
            (1 - load_change) * 0.1
        )
        
        return min(quality, 1.0)

    def _calculate_pattern_complexity(self, pattern: str) -> float:
        """Calculate the complexity of a pattern."""
        # Count number of conditions and actions
        conditions = pattern.count("if") + pattern.count("when")
        actions = pattern.count("then") + pattern.count("do")
        
        # Count number of temporal constraints
        temporal_constraints = sum(1 for word in pattern.split() if word in [
            "before", "after", "during", "until", "while"
        ])
        
        # Calculate complexity score
        complexity = (conditions * 0.4 + actions * 0.4 + temporal_constraints * 0.2) / 5
        return min(complexity, 1.0)

    def _calculate_energy_level(self, task: Dict[str, Any], temporal_context: Dict[str, Any]) -> float:
        """Calculate the energy level required for a task."""
        # Base energy from cognitive load
        base_energy = self._calculate_cognitive_load(task)
        
        # Adjust for time of day
        time_of_day = temporal_context.get("time_of_day", "")
        if time_of_day == "morning":
            energy_multiplier = 1.2
        elif time_of_day == "afternoon":
            energy_multiplier = 1.0
        else:
            energy_multiplier = 0.8
        
        # Adjust for task type
        task_type = task.get("type", "")
        if task_type in ["creative", "strategic"]:
            energy_multiplier *= 1.2
        elif task_type in ["routine", "administrative"]:
            energy_multiplier *= 0.8
        
        return min(base_energy * energy_multiplier, 1.0)

    def _calculate_focus_score(self, task: Dict[str, Any], temporal_context: Dict[str, Any]) -> float:
        """Calculate the focus score for a task."""
        # Base focus from cognitive load
        base_focus = self._calculate_cognitive_load(task)
        
        # Adjust for task complexity
        complexity = task.get("complexity", 0)
        focus_multiplier = 1 + (complexity * 0.2)
        
        # Adjust for time of day
        time_of_day = temporal_context.get("time_of_day", "")
        if time_of_day in ["morning", "afternoon"]:
            focus_multiplier *= 1.1
        else:
            focus_multiplier *= 0.9
        
        return min(base_focus * focus_multiplier, 1.0)

    def _calculate_temporal_variance(self, temporal_patterns: List[Dict[str, Any]]) -> float:
        """Calculate the variance in temporal scores."""
        scores = [p["temporal_score"] for p in temporal_patterns]
        if not scores:
            return 0.0
        
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        return min(variance, 1.0)

    def _calculate_skill_utilization(self, prev_task: Dict[str, Any], current_task: Dict[str, Any]) -> float:
        """Calculate skill utilization between tasks."""
        prev_skills = set(prev_task.get("required_skills", []))
        current_skills = set(current_task.get("required_skills", []))
        
        if not prev_skills or not current_skills:
            return 0.0
        
        # Calculate skill overlap
        overlap = len(prev_skills.intersection(current_skills))
        utilization = overlap / max(len(prev_skills), len(current_skills))
        
        return utilization

    def _calculate_learning_curve(self, prev_task: Dict[str, Any], current_task: Dict[str, Any]) -> float:
        """Calculate learning curve between tasks."""
        prev_complexity = prev_task.get("complexity", 0)
        current_complexity = current_task.get("complexity", 0)
        
        if prev_complexity == 0:
            return 0.0
        
        # Calculate complexity increase
        complexity_increase = (current_complexity - prev_complexity) / prev_complexity
        
        # Normalize to 0-1 range
        learning_curve = (complexity_increase + 1) / 2
        return min(max(learning_curve, 0.0), 1.0)

    def _calculate_stress_level(self, task: Dict[str, Any], temporal_context: Dict[str, Any], position: int) -> float:
        """Calculate stress level for a task."""
        # Base stress from cognitive load
        base_stress = self._calculate_cognitive_load(task)
        
        # Adjust for urgency
        urgency = task.get("urgency", 0)
        stress_multiplier = 1 + (urgency * 0.3)
        
        # Adjust for position in sequence
        position_multiplier = 1 + (position * 0.1)
        
        # Adjust for time of day
        time_of_day = temporal_context.get("time_of_day", "")
        if time_of_day in ["morning", "afternoon"]:
            time_multiplier = 0.9
        else:
            time_multiplier = 1.1
        
        return min(base_stress * stress_multiplier * position_multiplier * time_multiplier, 1.0)

    def _calculate_focus_continuity(self, prev_task: Dict[str, Any], current_task: Dict[str, Any]) -> float:
        """Calculate focus continuity between tasks."""
        prev_focus = self._calculate_focus_score(prev_task, {})
        current_focus = self._calculate_focus_score(current_task, {})
        
        # Calculate focus change
        focus_change = abs(prev_focus - current_focus)
        
        # Calculate continuity as inverse of change
        continuity = 1 - focus_change
        return max(continuity, 0.0)

    def _calculate_temporal_compatibility(
        self,
        current_task: Dict[str, Any],
        next_task: Dict[str, Any],
        temporal_context: Dict[str, Any]
    ) -> float:
        """Calculate temporal compatibility between tasks."""
        current_temporal = self._calculate_temporal_relevance(current_task.get("type", ""), temporal_context)
        next_temporal = self._calculate_temporal_relevance(next_task.get("type", ""), temporal_context)
        
        # Calculate compatibility as similarity of temporal scores
        compatibility = 1 - abs(current_temporal - next_temporal)
        return max(compatibility, 0.0)

    def _calculate_resource_continuity(self, current_task: Dict[str, Any], next_task: Dict[str, Any]) -> float:
        """Calculate resource continuity between tasks."""
        current_resources = set(current_task.get("resources", []))
        next_resources = set(next_task.get("resources", []))
        
        if not current_resources or not next_resources:
            return 0.0
        
        # Calculate resource overlap
        overlap = len(current_resources.intersection(next_resources))
        continuity = overlap / max(len(current_resources), len(next_resources))
        
        return continuity

    def _calculate_energy_optimization(self, energy_levels: List[float]) -> float:
        """Calculate energy level optimization for a sequence."""
        if not energy_levels:
            return 0.0
        
        # Calculate energy variance
        mean_energy = sum(energy_levels) / len(energy_levels)
        variance = sum((energy - mean_energy) ** 2 for energy in energy_levels) / len(energy_levels)
        
        # Calculate optimization as inverse of variance
        optimization = 1 - min(variance, 1.0)
        return max(optimization, 0.0)

    def _calculate_sequence_quality(
        self,
        transition_quality_scores: List[float],
        temporal_patterns: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        cognitive_loads: List[float],
        context_switches: List[float],
        dependencies: List[float]
    ) -> float:
        """Calculate overall sequence quality."""
        # Calculate average transition quality
        avg_transition_quality = sum(transition_quality_scores) / len(transition_quality_scores) if transition_quality_scores else 0.0
        
        # Calculate average temporal score
        avg_temporal_score = sum(p["temporal_score"] for p in temporal_patterns) / len(temporal_patterns) if temporal_patterns else 0.0
        
        # Calculate average relationship strength
        avg_relationship_strength = sum(r["relationship_strength"] for r in relationships) / len(relationships) if relationships else 0.0
        
        # Calculate average cognitive load
        avg_cognitive_load = sum(cognitive_loads) / len(cognitive_loads) if cognitive_loads else 0.0
        
        # Calculate average context switching
        avg_context_switching = sum(context_switches) / len(context_switches) if context_switches else 0.0
        
        # Calculate average dependency satisfaction
        avg_dependency_satisfaction = sum(dependencies) / len(dependencies) if dependencies else 1.0
        
        # Combine factors with weights
        quality = (
            avg_transition_quality * 0.3 +
            avg_temporal_score * 0.2 +
            avg_relationship_strength * 0.2 +
            (1 - avg_cognitive_load) * 0.1 +
            (1 - avg_context_switching) * 0.1 +
            avg_dependency_satisfaction * 0.1
        )
        
        return min(quality, 1.0)

    def _calculate_productivity_potential(
        self,
        sequence_metrics: Dict[str, float],
        temporal_context: Dict[str, Any]
    ) -> float:
        """Calculate productivity potential for a sequence."""
        # Get relevant metrics
        temporal_consistency = sequence_metrics["temporal_consistency"]
        cognitive_load = sequence_metrics["cognitive_load"]
        context_switching = sequence_metrics["context_switching"]
        energy_optimization = sequence_metrics["energy_level_optimization"]
        focus_continuity = sequence_metrics["focus_continuity"]
        
        # Adjust for time of day
        time_of_day = temporal_context.get("time_of_day", "")
        if time_of_day == "morning":
            time_multiplier = 1.2
        elif time_of_day == "afternoon":
            time_multiplier = 1.0
        else:
            time_multiplier = 0.8
        
        # Calculate productivity potential
        potential = (
            temporal_consistency * 0.3 +
            (1 - cognitive_load) * 0.2 +
            (1 - context_switching) * 0.2 +
            energy_optimization * 0.2 +
            focus_continuity * 0.1
        ) * time_multiplier
        
        return min(potential, 1.0)

    def _calculate_cognitive_load(self, task: Dict[str, Any]) -> float:
        """Calculate cognitive load for a task."""
        complexity = task.get("complexity", 0)
        duration = task.get("duration", 0)
        dependencies = len(task.get("dependencies", []))
        resources = len(task.get("resources", []))
        
        # Calculate base cognitive load
        base_load = (complexity * 0.4 + duration * 0.3 + dependencies * 0.2 + resources * 0.1)
        
        # Adjust for task type
        task_type = task.get("type", "")
        if task_type in ["planning", "analysis"]:
            base_load *= 1.2
        elif task_type in ["routine", "maintenance"]:
            base_load *= 0.8
        
        return min(base_load, 1.0)

    def _calculate_context_switch(self, prev_task: Dict[str, Any], curr_task: Dict[str, Any]) -> float:
        """Calculate the context switch cost between two tasks."""
        # Base cost for different domains
        base_cost = 0.0 if prev_task.get("domain") == curr_task.get("domain") else 0.3
        
        # Calculate cognitive load difference
        cognitive_diff = abs(
            self._calculate_cognitive_complexity(prev_task) -
            self._calculate_cognitive_complexity(curr_task)
        )
        
        # Calculate resource and context differences
        resource_diff = 1.0 - self._calculate_resource_overlap(prev_task, curr_task)
        context_diff = 1.0 - self._calculate_context_continuity(prev_task, curr_task)
        
        # Calculate skill set difference
        prev_skills = set(prev_task.get("required_skills", []))
        curr_skills = set(curr_task.get("required_skills", []))
        skill_diff = 1.0 - len(prev_skills & curr_skills) / len(prev_skills | curr_skills) if prev_skills or curr_skills else 0.0
        
        # Combine costs with weights
        switch_cost = (
            0.2 * base_cost +
            0.3 * cognitive_diff +
            0.2 * resource_diff +
            0.2 * context_diff +
            0.1 * skill_diff
        )
        
        return min(1.0, switch_cost)  # Ensure cost is in [0,1] range

    def _calculate_resource_utilization(self, sequence: List[Tuple[Dict[str, Any], str]]) -> Dict[str, Any]:
        """Calculate resource utilization for a sequence."""
        resource_usage = {}
        total_resources = set()
        
        for task, _ in sequence:
            resources = task.get("resources", [])
            total_resources.update(resources)
            for resource in resources:
                if resource not in resource_usage:
                    resource_usage[resource] = 0
                resource_usage[resource] += 1
        
        # Calculate efficiency
        efficiency = sum(resource_usage.values()) / (len(total_resources) * len(sequence)) if total_resources else 1.0
        
        return {
            "efficiency": efficiency,
            "allocation": resource_usage
        }

    def _calculate_pattern_coherence(self, sequence: List[Tuple[Dict[str, Any], str]]) -> float:
        """Calculate pattern coherence for a sequence."""
        patterns = [pattern for _, pattern in sequence]
        transitions = []
        
        for i in range(len(patterns) - 1):
            current = patterns[i]
            next_pattern = patterns[i + 1]
            transitions.append((current, next_pattern))
        
        # Calculate transition consistency
        consistent_transitions = sum(1 for t in transitions if t[0] == t[1])
        coherence = consistent_transitions / len(transitions) if transitions else 1.0
        
        return coherence

    def _calculate_temporal_flow(self, sequence: List[Tuple[Dict[str, Any], str]], temporal_context: Dict[str, Any]) -> float:
        """Calculate temporal flow for a sequence."""
        temporal_scores = []
        
        for task, pattern in sequence:
            temporal_score = self._calculate_temporal_relevance(pattern, temporal_context)
            temporal_scores.append(temporal_score)
        
        # Calculate flow as the consistency of temporal scores
        flow = 1 - (max(temporal_scores) - min(temporal_scores)) if temporal_scores else 1.0
        
        return flow

    def _calculate_resource_efficiency(self, sequence: List[Tuple[Dict[str, Any], str]]) -> float:
        """Calculate resource efficiency for a sequence."""
        resource_usage = {}
        total_tasks = len(sequence)
        
        for task, _ in sequence:
            resources = task.get("resources", [])
            for resource in resources:
                if resource not in resource_usage:
                    resource_usage[resource] = 0
                resource_usage[resource] += 1
        
        # Calculate efficiency as the ratio of tasks to unique resources
        efficiency = total_tasks / len(resource_usage) if resource_usage else 1.0
        
        return min(efficiency, 1.0)

    def _calculate_cognitive_continuity(self, sequence: List[Tuple[Dict[str, Any], str]]) -> float:
        """Calculate cognitive continuity for a sequence."""
        cognitive_loads = []
        
        for task, _ in sequence:
            cognitive_load = self._calculate_cognitive_load(task)
            cognitive_loads.append(cognitive_load)
        
        # Calculate continuity as the inverse of cognitive load variance
        if len(cognitive_loads) > 1:
            mean_load = sum(cognitive_loads) / len(cognitive_loads)
            variance = sum((load - mean_load) ** 2 for load in cognitive_loads) / len(cognitive_loads)
            continuity = 1 - min(variance, 1.0)
        else:
            continuity = 1.0
        
        return continuity

    def _calculate_dependency_satisfaction(self, dependencies: List[str], previous_tasks: List[Tuple[Dict[str, Any], str]]) -> float:
        """Calculate how well dependencies are satisfied by previous tasks."""
        if not dependencies:
            return 1.0
            
        satisfied = 0
        for dep in dependencies:
            for task, _ in previous_tasks:
                if task.get("id") == dep:
                    satisfied += 1
                    break
        
        return satisfied / len(dependencies)

    def _analyze_task_relationship(self, prev_task: Dict[str, Any], current_task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the relationship between two consecutive tasks."""
        relationship = {
            "type": "unknown",
            "strength": 0.0,
            "factors": [],
            "semantic_similarity": 0.0,
            "temporal_proximity": 0.0,
            "resource_overlap": 0.0,
            "context_continuity": 0.0
        }
        
        # Check for direct dependencies
        if current_task.get("id") in prev_task.get("dependencies", []):
            relationship["type"] = "dependency"
            relationship["strength"] = 1.0
            relationship["factors"].append("direct_dependency")
        
        # Check for shared attributes
        shared_attributes = self._find_shared_attributes(prev_task, current_task)
        if shared_attributes:
            relationship["type"] = "related"
            relationship["strength"] = len(shared_attributes) / 5.0  # Normalize by max possible shared attributes
            relationship["factors"].extend(shared_attributes)
        
        # Calculate semantic similarity
        relationship["semantic_similarity"] = self._calculate_semantic_similarity(prev_task, current_task)
        
        # Calculate temporal proximity
        if "due_date" in prev_task and "due_date" in current_task:
            prev_due = datetime.fromisoformat(prev_task["due_date"])
            curr_due = datetime.fromisoformat(current_task["due_date"])
            time_diff = abs((curr_due - prev_due).total_seconds())
            relationship["temporal_proximity"] = 1.0 - min(time_diff / 86400.0, 1.0)  # Normalize to 0-1
            if relationship["temporal_proximity"] > 0.8:
                relationship["type"] = "temporal"
                relationship["strength"] = max(relationship["strength"], 0.8)
                relationship["factors"].append("temporal_proximity")
        
        # Calculate resource overlap
        relationship["resource_overlap"] = self._calculate_resource_overlap(prev_task, current_task)
        
        # Calculate context continuity
        relationship["context_continuity"] = self._calculate_context_continuity(prev_task, current_task)
        
        # Update overall strength based on all factors
        relationship["strength"] = max(relationship["strength"], 
            (relationship["semantic_similarity"] * 0.3 +
             relationship["temporal_proximity"] * 0.2 +
             relationship["resource_overlap"] * 0.2 +
             relationship["context_continuity"] * 0.3))
        
        return relationship

    def _calculate_semantic_similarity(self, task1: Dict[str, Any], task2: Dict[str, Any]) -> float:
        """Calculate semantic similarity between two tasks."""
        # Extract text content
        text1 = f"{task1.get('title', '')} {task1.get('description', '')}".lower()
        text2 = f"{task2.get('title', '')} {task2.get('description', '')}".lower()
        
        # Split into words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        if union == 0:
            return 0.0
        
        return intersection / union

    def _calculate_resource_overlap(self, task1: Dict[str, Any], task2: Dict[str, Any]) -> float:
        """Calculate resource overlap between two tasks."""
        overlap = 0.0
        
        # Check assignee
        if task1.get("assignee") == task2.get("assignee"):
            overlap += 0.4
        
        # Check project
        if task1.get("project") == task2.get("project"):
            overlap += 0.3
        
        # Check labels
        labels1 = set(task1.get("labels", []))
        labels2 = set(task2.get("labels", []))
        if labels1 & labels2:
            overlap += 0.3
        
        return overlap

    def _calculate_context_continuity(self, task1: Dict[str, Any], task2: Dict[str, Any]) -> float:
        """Calculate context continuity between two tasks."""
        continuity = 0.0
        
        # Check for similar patterns
        pattern1 = self._extract_task_pattern(task1)
        pattern2 = self._extract_task_pattern(task2)
        if pattern1 == pattern2:
            continuity += 0.4
        
        # Check for similar complexity
        complexity1 = self._calculate_task_complexity(task1)
        complexity2 = self._calculate_task_complexity(task2)
        complexity_diff = abs(complexity1 - complexity2)
        continuity += 0.3 * (1.0 - complexity_diff)
        
        # Check for similar priority
        priority1 = task1.get("priority", 1)
        priority2 = task2.get("priority", 1)
        if priority1 == priority2:
            continuity += 0.3
        
        return continuity

    def _extract_task_pattern(self, task: Dict[str, Any]) -> str:
        """Extract the dominant pattern from a task."""
        task_text = f"{task.get('title', '')} {task.get('description', '')}".lower()
        pattern_matches = self._find_matching_patterns(task_text, {})
        if pattern_matches:
            return max(pattern_matches, key=lambda x: x["score"])["pattern"]
        return "unknown"

    def _optimize_sequence(self, tasks: List[Dict[str, Any]], temporal_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Optimize task sequence based on multiple factors including innovation, resilience, and flow.
        
        Args:
            tasks: List of tasks to optimize
            temporal_context: Current temporal context
            
        Returns:
            Optimized sequence of tasks
        """
        # Extract task patterns
        task_patterns = [(task, self._extract_task_pattern(task)) for task in tasks]
        
        # Generate all possible sequences
        sequences = self._generate_sequences(task_patterns)
        
        # Score each sequence
        scored_sequences = []
        for sequence in sequences:
            # Calculate base sequence score
            sequence_score = self._calculate_sequence_score(sequence, temporal_context)
            
            # Calculate innovation potential
            innovation_score = self._calculate_innovation_potential(sequence[0][0], temporal_context)
            
            # Calculate resilience score
            resilience_score = 0
            for i in range(len(sequence) - 1):
                resilience_score += self._calculate_resilience(
                    sequence[i][0],
                    sequence[i + 1][0],
                    temporal_context
                )
            resilience_score /= max(1, len(sequence) - 1)
            
            # Calculate flow efficiency
            flow_score = 0
            for i in range(len(sequence) - 1):
                flow_score += self._calculate_task_flow_efficiency(
                    sequence[i][0],
                    sequence[i + 1][0],
                    temporal_context
                )
            flow_score /= max(1, len(sequence) - 1)
            
            # Calculate weighted total score
            weights = {
                "sequence": 0.4,
                "innovation": 0.2,
                "resilience": 0.2,
                "flow": 0.2
            }
            
            total_score = (
                weights["sequence"] * sequence_score +
                weights["innovation"] * innovation_score +
                weights["resilience"] * resilience_score +
                weights["flow"] * flow_score
            )
            
            scored_sequences.append((sequence, total_score))
        
        # Return the best sequence
        best_sequence = max(scored_sequences, key=lambda x: x[1])[0]
        return [task for task, _ in best_sequence]
    
    def _calculate_innovation_potential(self, task: Dict[str, Any], temporal_context: Dict[str, Any]) -> float:
        """
        Calculate the innovation potential of a task based on its characteristics and temporal context.
        
        This method evaluates how conducive a task is to innovative thinking and creative problem-solving.
        It considers both intrinsic task properties and temporal factors that influence innovation.
        
        Args:
            task: The task to evaluate, containing properties like complexity, novelty, and creativity
            temporal_context: Current temporal context including time period, business calendar, etc.
            
        Returns:
            A score between 0 and 1 representing the task's innovation potential, where:
            - 0.0: No innovation potential
            - 0.5: Moderate innovation potential
            - 1.0: High innovation potential
            
        The score is calculated based on:
        1. Task complexity: How challenging the task is
        2. Novelty: How new or unique the task is
        3. Creativity: How much creative thinking is required
        4. Time factor: Adjusts score based on time of day (higher in morning)
        """
        # Extract task features
        complexity = task.get("complexity", 0)
        novelty = task.get("novelty", 0)
        creativity = task.get("creativity", 0)
        
        # Calculate time-based innovation factors
        time_factor = 1.0
        if temporal_context.get("time_period") == "morning":
            time_factor = 1.2  # Higher innovation potential in morning
        elif temporal_context.get("time_period") == "afternoon":
            time_factor = 0.8  # Lower innovation potential in afternoon
        
        # Calculate innovation score
        innovation_score = (complexity + novelty + creativity) / 3
        return min(1.0, innovation_score * time_factor)
    
    def _calculate_resilience(
        self,
        prev_task: Dict[str, Any],
        current_task: Dict[str, Any],
        temporal_context: Dict[str, Any]
    ) -> float:
        """
        Calculate the resilience of a task transition based on multiple factors.
        
        This method evaluates how well a sequence of tasks can maintain performance under
        varying conditions and potential disruptions. It considers temporal, resource, and
        task-specific factors that contribute to overall resilience.
        
        Args:
            prev_task: The previous task in the sequence
            current_task: The current task being evaluated
            temporal_context: Current temporal context including time period, business calendar, etc.
            
        Returns:
            A score between 0 and 1 representing the transition's resilience, where:
            - 0.0: Low resilience (high risk of disruption)
            - 0.5: Moderate resilience
            - 1.0: High resilience (can withstand disruptions)
            
        The score is calculated based on:
        1. Temporal resilience: How well the tasks align with temporal patterns
        2. Resource resilience: How efficiently resources are utilized
        3. Task stability: How stable and predictable the task is
        """
        # Calculate temporal resilience
        temporal_resilience = self._calculate_temporal_resilience(
            prev_task,
            current_task,
            temporal_context
        )
        
        # Calculate resource resilience
        resource_resilience = self._calculate_resource_resilience(
            prev_task,
            current_task
        )
        
        # Calculate task stability
        task_stability = self._calculate_task_stability(current_task)
        
        # Calculate weighted resilience score
        weights = {
            "temporal": 0.4,
            "resource": 0.3,
            "stability": 0.3
        }
        
        return (
            weights["temporal"] * temporal_resilience +
            weights["resource"] * resource_resilience +
            weights["stability"] * task_stability
        )
    
    def _calculate_task_flow_efficiency(
        self,
        prev_task: Dict[str, Any],
        current_task: Dict[str, Any],
        temporal_context: Dict[str, Any]
    ) -> float:
        """
        Calculate the flow efficiency of a task transition based on cognitive and temporal factors.
        
        This method evaluates how smoothly a person can transition between tasks while
        maintaining focus and productivity. It considers factors that contribute to
        maintaining a state of flow during task execution.
        
        Args:
            prev_task: The previous task in the sequence
            current_task: The current task being evaluated
            temporal_context: Current temporal context including time period, business calendar, etc.
            
        Returns:
            A score between 0 and 1 representing the flow efficiency, where:
            - 0.0: Poor flow (frequent context switches)
            - 0.5: Moderate flow
            - 1.0: Excellent flow (seamless transitions)
            
        The score is calculated based on:
        1. Skill utilization: How well skills are leveraged between tasks
        2. Learning curve: How tasks build on previous knowledge
        3. Focus continuity: How well focus is maintained
        4. Temporal compatibility: How well tasks align with temporal patterns
        """
        # Calculate skill utilization
        skill_utilization = self._calculate_skill_utilization(prev_task, current_task)
        
        # Calculate learning curve
        learning_curve = self._calculate_learning_curve(prev_task, current_task)
        
        # Calculate focus continuity
        focus_continuity = self._calculate_focus_continuity(prev_task, current_task)
        
        # Calculate temporal compatibility
        temporal_compatibility = self._calculate_temporal_compatibility(
            prev_task,
            current_task,
            temporal_context
        )
        
        # Calculate weighted flow efficiency
        weights = {
            "skill": 0.3,
            "learning": 0.2,
            "focus": 0.3,
            "temporal": 0.2
        }
        
        return (
            weights["skill"] * skill_utilization +
            weights["learning"] * learning_curve +
            weights["focus"] * focus_continuity +
            weights["temporal"] * temporal_compatibility
        )

    def select_tasks(
        self, state: Dict[str, Any], possible_tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Select tasks using domain-guided exploration."""
        if not possible_tasks:
            return []
            
        # Extract temporal context
        temporal_context = state.get("temporal", {})
        
        # Optimize task sequence
        optimized_sequence = self._optimize_sequence(possible_tasks, temporal_context)
        
        # Return top 3 tasks from optimized sequence
        return optimized_sequence[:3]

    def recommend_sequence(self, tasks: List[Dict[str, Any]], temporal_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Recommend an optimal sequence of tasks based on learned patterns and temporal context.
        
        Args:
            tasks: List of tasks to sequence
            temporal_context: Current temporal context
            
        Returns:
            List of tasks in recommended order
        """
        # Extract task patterns
        task_patterns = [(task, self._extract_task_pattern(task)) for task in tasks]
        
        # Get sequence recommendations
        recommended_sequences = self._get_sequence_recommendations(task_patterns, temporal_context)
        
        if not recommended_sequences:
            # If no recommendations, use default optimization
            return self._optimize_sequence(tasks, temporal_context)
        
        # Get the best sequence
        best_sequence = max(recommended_sequences, key=lambda seq: self._calculate_recommendation_score(seq, self._analyze_sequence_patterns(seq, temporal_context), temporal_context))
        
        # Return the tasks in the recommended order
        return [task for task, _ in best_sequence]
    
    def _get_sequence_recommendations(
        self,
        task_patterns: List[Tuple[Dict[str, Any], str]],
        temporal_context: Dict[str, Any]
    ) -> List[List[Tuple[Dict[str, Any], str]]]:
        """Generate recommendations for task sequences."""
        recommendations = []
        
        # Get successful patterns for current context
        successful_patterns = self._get_successful_patterns(temporal_context)
        
        # Get sequence templates
        templates = self._get_sequence_templates(temporal_context)
        
        # Generate recommendations from templates
        for template in templates:
            sequence = self._apply_template(template, task_patterns, successful_patterns)
            if sequence:
                recommendations.append(sequence)
        
        # Generate recommendations from transition probabilities
        transition_recommendations = self._generate_transition_recommendations(
            task_patterns, temporal_context
        )
        recommendations.extend(transition_recommendations)
        
        # Sort recommendations by score
        scored_recommendations = []
        for sequence in recommendations:
            pattern_analysis = self._analyze_sequence_patterns(sequence, temporal_context)
            score = self._calculate_recommendation_score(sequence, pattern_analysis, temporal_context)
            scored_recommendations.append((sequence, score))
        
        # Return top recommendations sorted by score
        sorted_recommendations = [
            seq for seq, _ in sorted(scored_recommendations, key=lambda x: x[1], reverse=True)
        ]
        return sorted_recommendations[:5]  # Return top 5 recommendations
    
    def _get_successful_patterns(self, temporal_context: Dict[str, Any]) -> Dict[str, float]:
        """Get patterns that have been successful in similar temporal contexts."""
        successful_patterns = {}
        
        # Get historical pattern performance
        for pattern, stats in self.pattern_performance.items():
            # Check temporal context match
            context_matches = []
            for context, performance in stats["temporal_performance"].items():
                if self._is_temporal_match(json.loads(context), temporal_context):
                    context_matches.append(performance["success_rate"])
            
            # Calculate success rate for matching contexts
            if context_matches:
                success_rate = sum(context_matches) / len(context_matches)
                successful_patterns[pattern] = success_rate
        
        return successful_patterns
    
    def _get_sequence_templates(self, temporal_context: Dict[str, Any]) -> List[List[str]]:
        """Get sequence templates that have been successful."""
        templates = []
        
        # Get successful sequences from history
        for sequence in self.sequence_history:
            if sequence["success"] and self._is_temporal_match(
                sequence["temporal_context"],
                temporal_context
            ):
                templates.append([p for _, p in sequence["patterns"]])
        
        # Add default templates for different contexts
        if self._is_peak_productivity_hours(temporal_context.get("timestamp", datetime.now())):
            templates.append(["focus", "deep_work", "review"])
        elif self._is_low_energy_hours(temporal_context.get("timestamp", datetime.now())):
            templates.append(["routine", "maintenance", "planning"])
        
        return templates
    
    def _is_temporal_match(
        self,
        sequence_context: Dict[str, Any],
        current_context: Dict[str, Any]
    ) -> bool:
        """Check if two temporal contexts match."""
        # Check time of day
        if (sequence_context.get("time_of_day") ==
            current_context.get("time_of_day")):
            return True
        
        # Check day type (workday, weekend)
        if (sequence_context.get("is_workday") ==
            current_context.get("is_workday")):
            return True
        
        # Check energy level
        seq_energy = sequence_context.get("energy_level", 0.5)
        curr_energy = current_context.get("energy_level", 0.5)
        if abs(seq_energy - curr_energy) < 0.2:
            return True
        
        return False
    
    def _apply_template(
        self,
        template: List[str],
        task_patterns: List[Tuple[Dict[str, Any], str]],
        successful_patterns: Dict[str, float]
    ) -> List[Tuple[Dict[str, Any], str]]:
        """Apply a sequence template to current tasks."""
        result = []
        used_tasks = set()
        
        # Try to fill each template position
        for template_pattern in template:
            best_match = None
            best_score = -1
            
            # Find best matching task for this pattern
            for task, pattern in task_patterns:
                if task["id"] in used_tasks:
                    continue
                    
                # Calculate match score
                pattern_score = successful_patterns.get(pattern, 0.3)
                template_match = 1.0 if pattern == template_pattern else 0.0
                
                # Adjust score based on task properties
                task_score = (
                    0.4 * pattern_score +
                    0.4 * template_match +
                    0.2 * self._calculate_task_flow_efficiency(
                        result[-1][0] if result else None,
                        task,
                        {}  # Empty temporal context as it's not needed here
                    )
                )
                
                if task_score > best_score:
                    best_score = task_score
                    best_match = (task, pattern)
            
            if best_match:
                result.append(best_match)
                used_tasks.add(best_match[0]["id"])
        
        return result if len(result) == len(template) else None
    
    def _generate_transition_recommendations(
        self,
        task_patterns: List[Tuple[Dict[str, Any], str]],
        temporal_context: Dict[str, Any]
    ) -> List[List[Tuple[Dict[str, Any], str]]]:
        """Generate recommendations based on transition probabilities."""
        recommendations = []
        
        # Generate different sequence permutations
        tasks_remaining = set(task["id"] for task, _ in task_patterns)
        current_sequence = []
        
        def backtrack():
            if not tasks_remaining:
                recommendations.append(current_sequence[:])
                return
            
            for task, pattern in task_patterns:
                if task["id"] not in tasks_remaining:
                    continue
                    
                # Calculate transition probability
                transition_prob = 1.0
                if current_sequence:
                    prev_task, prev_pattern = current_sequence[-1]
                    transition_prob = self._get_transition_probability(
                        prev_pattern,
                        pattern,
                        temporal_context
                    )
                
                # Only consider transitions with good probability
                if transition_prob > 0.3:
                    current_sequence.append((task, pattern))
                    tasks_remaining.remove(task["id"])
                    backtrack()
                    tasks_remaining.add(task["id"])
                    current_sequence.pop()
        
        backtrack()
        return recommendations
    
    def _calculate_recommendation_score(
        self,
        sequence: List[Tuple[Dict[str, Any], str]],
        pattern_analysis: Dict[str, Any],
        temporal_context: Dict[str, Any]
    ) -> float:
        """Calculate a score for a recommended sequence."""
        if not sequence:
            return 0.0
        
        # Get metrics from pattern analysis
        transition_quality = pattern_analysis["pattern_analysis"]["avg_transition_quality"]
        knowledge_transfer = pattern_analysis["pattern_analysis"]["avg_knowledge_transfer"]
        adaptability = pattern_analysis["pattern_analysis"]["avg_adaptability"]
        
        # Get sequence metrics
        pattern_diversity = pattern_analysis["sequence_metrics"]["pattern_diversity"]
        temporal_consistency = pattern_analysis["sequence_metrics"]["temporal_consistency"]
        resource_utilization = pattern_analysis["sequence_metrics"]["resource_utilization"]
        cognitive_load = pattern_analysis["sequence_metrics"]["cognitive_load"]
        context_switching = pattern_analysis["sequence_metrics"]["context_switching"]
        
        # Calculate temporal relevance
        temporal_relevance = sum(
            self._calculate_temporal_relevance(pattern, temporal_context)
            for _, pattern in sequence
        ) / len(sequence)
        
        # Calculate final score with weights
        score = (
            0.2 * transition_quality +
            0.15 * knowledge_transfer +
            0.15 * adaptability +
            0.1 * pattern_diversity +
            0.1 * temporal_consistency +
            0.1 * resource_utilization +
            0.1 * (1.0 - cognitive_load) +  # Lower cognitive load is better
            0.1 * (1.0 - context_switching) +  # Lower context switching is better
            0.1 * temporal_relevance
        )
        
        return score

    def update_recommendations(self, sequence: List[Dict[str, Any]], success: bool, temporal_context: Dict[str, Any]):
        """Update recommendation data based on sequence outcome."""
        # Extract patterns from sequence
        patterns = []
        for task in sequence:
            task_text = f"{task.get('title', '')} {task.get('description', '')}".lower()
            pattern_matches = self._find_matching_patterns(task_text, temporal_context)
            if pattern_matches:
                best_pattern = max(pattern_matches, key=lambda x: x["score"])
                patterns.append(best_pattern["pattern"])
        
        # Update successful patterns
        for pattern in patterns:
            self._update_successful_pattern(pattern, success, temporal_context)
        
        # Update sequence templates
        if success:
            self._update_sequence_template(patterns, temporal_context)
        
        # Update performance history
        self._update_performance_history(sequence, success, temporal_context)
        
        self._save_domain_knowledge()

    def _update_successful_pattern(self, pattern: str, success: bool, temporal_context: Dict[str, Any]):
        """Update successful pattern data."""
        # Update time period patterns
        time_period = temporal_context.get("time_period", "")
        if time_period:
            patterns = self.domain_knowledge["pattern_sequences"]["recommendations"]["temporal_patterns"][time_period]
            current_score = patterns.get(pattern, 0.5)
            new_score = (0.9 * current_score) + (0.1 * (1.0 if success else 0.0))
            patterns[pattern] = new_score
        
        # Update business calendar patterns
        if temporal_context.get("is_first_business_day", False):
            patterns = self.domain_knowledge["pattern_sequences"]["recommendations"]["temporal_patterns"]["first_business_day"]
            self._update_pattern_score(patterns, pattern, success)
        
        if temporal_context.get("is_last_business_day", False):
            patterns = self.domain_knowledge["pattern_sequences"]["recommendations"]["temporal_patterns"]["last_business_day"]
            self._update_pattern_score(patterns, pattern, success)
        
        if temporal_context.get("is_holiday", False):
            patterns = self.domain_knowledge["pattern_sequences"]["recommendations"]["temporal_patterns"]["holiday"]
            self._update_pattern_score(patterns, pattern, success)
        
        if temporal_context.get("is_weekend", False):
            patterns = self.domain_knowledge["pattern_sequences"]["recommendations"]["temporal_patterns"]["weekend"]
            self._update_pattern_score(patterns, pattern, success)
        
        # Update season patterns
        season = temporal_context.get("season", "")
        if season:
            patterns = self.domain_knowledge["pattern_sequences"]["recommendations"]["temporal_patterns"][season]
            self._update_pattern_score(patterns, pattern, success)
        
        # Update work schedule patterns
        if temporal_context.get("is_typical_work_hours", False):
            patterns = self.domain_knowledge["pattern_sequences"]["recommendations"]["temporal_patterns"]["work_hours"]
            self._update_pattern_score(patterns, pattern, success)
        
        if temporal_context.get("is_peak_productivity", False):
            patterns = self.domain_knowledge["pattern_sequences"]["recommendations"]["temporal_patterns"]["peak_hours"]
            self._update_pattern_score(patterns, pattern, success)
        
        if temporal_context.get("is_low_energy", False):
            patterns = self.domain_knowledge["pattern_sequences"]["recommendations"]["temporal_patterns"]["low_energy"]
            self._update_pattern_score(patterns, pattern, success)
        
        if temporal_context.get("is_global_collaboration_hours", False):
            patterns = self.domain_knowledge["pattern_sequences"]["recommendations"]["temporal_patterns"]["global_collaboration"]
            self._update_pattern_score(patterns, pattern, success)

    def _update_pattern_score(self, patterns: Dict[str, float], pattern: str, success: bool):
        """Update a pattern's success score."""
        current_score = patterns.get(pattern, 0.5)
        new_score = (0.9 * current_score) + (0.1 * (1.0 if success else 0.0))
        patterns[pattern] = new_score

    def _update_sequence_template(self, patterns: List[str], temporal_context: Dict[str, Any]):
        """Update sequence templates with successful patterns."""
        template = {
            "patterns": patterns,
            "temporal_context": temporal_context,
            "timestamp": datetime.now().isoformat(),
            "success_count": 1
        }
        
        # Check if similar template exists
        for existing in self.domain_knowledge["pattern_sequences"]["recommendations"]["sequence_templates"]:
            if self._is_temporal_match(existing["temporal_context"], temporal_context) and existing["patterns"] == patterns:
                existing["success_count"] += 1
                return
        
        # Add new template
        self.domain_knowledge["pattern_sequences"]["recommendations"]["sequence_templates"].append(template)

    def _update_performance_history(self, sequence: List[Dict[str, Any]], success: bool, temporal_context: Dict[str, Any]):
        """Update performance history for sequence recommendations."""
        sequence_id = str(hash(str(sequence)))
        
        if sequence_id not in self.domain_knowledge["pattern_sequences"]["recommendations"]["performance_history"]:
            self.domain_knowledge["pattern_sequences"]["recommendations"]["performance_history"][sequence_id] = {
                "success_count": 0,
                "total_count": 0,
                "temporal_context": temporal_context,
                "last_updated": datetime.now().isoformat()
            }
        
        history = self.domain_knowledge["pattern_sequences"]["recommendations"]["performance_history"][sequence_id]
        history["total_count"] += 1
        if success:
            history["success_count"] += 1
        history["last_updated"] = datetime.now().isoformat()

    def _calculate_knowledge_transfer(
        self,
        current_task: Dict[str, Any],
        next_task: Dict[str, Any],
        current_pattern: str,
        next_pattern: str
    ) -> float:
        """Calculate knowledge transfer between tasks."""
        # Calculate skill overlap
        current_skills = set(current_task.get("required_skills", []))
        next_skills = set(next_task.get("required_skills", []))
        skill_overlap = len(current_skills.intersection(next_skills)) / max(len(current_skills), len(next_skills)) if current_skills and next_skills else 0
        
        # Calculate pattern similarity
        pattern_similarity = 1.0 if current_pattern == next_pattern else 0.5
        
        # Calculate context overlap
        current_context = set(current_task.get("context", []))
        next_context = set(next_task.get("context", []))
        context_overlap = len(current_context.intersection(next_context)) / max(len(current_context), len(next_context)) if current_context and next_context else 0
        
        # Combine factors with weights
        knowledge_transfer = (
            skill_overlap * 0.4 +
            pattern_similarity * 0.3 +
            context_overlap * 0.3
        )
        
        return min(knowledge_transfer, 1.0)

    def _calculate_adaptability(self, task: Dict[str, Any], temporal_context: Dict[str, Any]) -> float:
        """Calculate task adaptability."""
        # Base adaptability from task properties
        complexity = task.get("complexity", 0)
        dependencies = len(task.get("dependencies", []))
        resources = len(task.get("resources", []))
        
        # Calculate base adaptability
        base_adaptability = 1 - (complexity * 0.4 + dependencies * 0.3 + resources * 0.3)
        
        # Adjust for temporal context
        time_of_day = temporal_context.get("time_of_day", "")
        if time_of_day in ["morning", "afternoon"]:
            time_multiplier = 1.1
        else:
            time_multiplier = 0.9
        
        return min(base_adaptability * time_multiplier, 1.0)

    def _calculate_resilience(
        self,
        prev_task: Dict[str, Any],
        current_task: Dict[str, Any],
        temporal_context: Dict[str, Any]
    ) -> float:
        """Calculate resilience between tasks."""
        # Calculate task stability
        prev_stability = self._calculate_task_stability(prev_task)
        current_stability = self._calculate_task_stability(current_task)
        
        # Calculate temporal resilience
        temporal_resilience = self._calculate_temporal_resilience(prev_task, current_task, temporal_context)
        
        # Calculate resource resilience
        resource_resilience = self._calculate_resource_resilience(prev_task, current_task)
        
        # Combine factors with weights
        resilience = (
            (prev_stability + current_stability) * 0.4 +
            temporal_resilience * 0.3 +
            resource_resilience * 0.3
        ) / 2
        
        return min(resilience, 1.0)

    def _calculate_task_stability(self, task: Dict[str, Any]) -> float:
        """Calculate task stability."""
        # Factors affecting stability
        complexity = task.get("complexity", 0)
        dependencies = len(task.get("dependencies", []))
        resources = len(task.get("resources", []))
        duration = task.get("duration", 0)
        
        # Calculate stability
        stability = 1 - (complexity * 0.3 + dependencies * 0.2 + resources * 0.2 + duration * 0.3)
        return max(stability, 0.0)

    def _calculate_temporal_resilience(
        self,
        prev_task: Dict[str, Any],
        current_task: Dict[str, Any],
        temporal_context: Dict[str, Any]
    ) -> float:
        """Calculate temporal resilience between tasks."""
        # Calculate temporal compatibility
        temporal_compatibility = self._calculate_temporal_compatibility(prev_task, current_task, temporal_context)
        
        # Calculate time buffer
        prev_buffer = prev_task.get("buffer_time", 0)
        current_buffer = current_task.get("buffer_time", 0)
        buffer_score = min(prev_buffer + current_buffer, 1.0)
        
        # Combine factors
        resilience = temporal_compatibility * 0.6 + buffer_score * 0.4
        return min(resilience, 1.0)

    def _calculate_resource_resilience(self, prev_task: Dict[str, Any], current_task: Dict[str, Any]) -> float:
        """Calculate resource resilience between tasks."""
        # Calculate resource overlap
        prev_resources = set(prev_task.get("resources", []))
        current_resources = set(current_task.get("resources", []))
        
        if not prev_resources or not current_resources:
            return 0.0
        
        # Calculate resource flexibility
        overlap = len(prev_resources.intersection(current_resources))
        flexibility = overlap / max(len(prev_resources), len(current_resources))
        
        return flexibility

    def _calculate_innovation_potential(self, task: Dict[str, Any], temporal_context: Dict[str, Any]) -> float:
        """Calculate innovation potential for a task."""
        # Base innovation from task properties
        complexity = task.get("complexity", 0)
        creativity = task.get("creativity_required", 0)
        novelty = task.get("novelty", 0)
        
        # Calculate base innovation
        base_innovation = (complexity * 0.3 + creativity * 0.4 + novelty * 0.3)
        
        # Adjust for temporal context
        time_of_day = temporal_context.get("time_of_day", "")
        if time_of_day in ["morning", "afternoon"]:
            time_multiplier = 1.2
        else:
            time_multiplier = 0.8
        
        return min(base_innovation * time_multiplier, 1.0)

    def _calculate_task_flow_efficiency(
        self,
        prev_task: Dict[str, Any],
        current_task: Dict[str, Any],
        temporal_context: Dict[str, Any]
    ) -> float:
        """Calculate task flow efficiency."""
        # Calculate transition quality
        transition_quality = self._calculate_transition_quality(
            prev_task, current_task,
            prev_task.get("type", ""), current_task.get("type", ""),
            temporal_context
        )
        
        # Calculate resource continuity
        resource_continuity = self._calculate_resource_continuity(prev_task, current_task)
        
        # Calculate temporal flow
        temporal_flow = self._calculate_temporal_flow([(prev_task, ""), (current_task, "")], temporal_context)
        
        # Combine factors with weights
        efficiency = (
            transition_quality * 0.4 +
            resource_continuity * 0.3 +
            temporal_flow * 0.3
        )
        
        return min(efficiency, 1.0)

    def _analyze_pattern_evolution(self, sequence: List[Tuple[Dict[str, Any], str]]) -> Dict[str, Any]:
        """Analyze the evolution of patterns in a sequence."""
        if len(sequence) < 2:
            return {
                "pattern_changes": [],
                "complexity_trend": [],
                "adaptation_rate": [],
                "innovation_points": []
            }

        evolution = {
            "pattern_changes": [],
            "complexity_trend": [],
            "adaptation_rate": [],
            "innovation_points": []
        }

        # Analyze pattern changes
        for i in range(len(sequence) - 1):
            curr_task, curr_pattern = sequence[i]
            next_task, next_pattern = sequence[i + 1]

            # Calculate complexity change
            curr_complexity = self._calculate_pattern_complexity(curr_pattern)
            next_complexity = self._calculate_pattern_complexity(next_pattern)
            complexity_change = next_complexity - curr_complexity

            # Calculate adaptation rate
            adaptation = self._calculate_adaptability(next_task, {})

            # Determine if this is an innovation point
            is_innovation = (
                complexity_change > 0.3 and  # Significant complexity increase
                adaptation > 0.7 and  # High adaptability
                self._calculate_pattern_novelty(next_pattern) > 0.6  # Novel pattern
            )

            evolution["pattern_changes"].append({
                "from_pattern": curr_pattern,
                "to_pattern": next_pattern,
                "complexity_change": complexity_change,
                "adaptation_rate": adaptation,
                "is_innovation": is_innovation
            })

            evolution["complexity_trend"].append(next_complexity)
            evolution["adaptation_rate"].append(adaptation)
            
            if is_innovation:
                evolution["innovation_points"].append({
                    "task": next_task,
                    "pattern": next_pattern,
                    "innovation_score": self._calculate_innovation_potential(next_task, {})
                })

        return evolution

    def _analyze_pattern_clusters(self, sequence: List[Tuple[Dict[str, Any], str]]) -> Dict[str, Any]:
        """Analyze clusters of patterns in a sequence."""
        if not sequence:
            return {
                "pattern_groups": {},
                "cluster_metrics": {},
                "transition_networks": {}
            }

        # Group tasks by pattern
        pattern_groups = {}
        for task, pattern in sequence:
            if pattern not in pattern_groups:
                pattern_groups[pattern] = []
            pattern_groups[pattern].append(task)

        # Calculate cluster metrics
        cluster_metrics = {}
        for pattern, tasks in pattern_groups.items():
            metrics = {
                "size": len(tasks),
                "complexity": self._calculate_pattern_complexity(pattern),
                "adaptability": sum(
                    self._calculate_adaptability(task, {})
                    for task in tasks
                ) / len(tasks),
                "innovation": self._calculate_pattern_novelty(pattern),
                "cognitive_load": sum(
                    self._calculate_cognitive_complexity(task)
                    for task in tasks
                ) / len(tasks),
                "resource_utilization": sum(
                    self._calculate_resource_efficiency([(task, pattern)])
                    for task in tasks
                ) / len(tasks)
            }
            cluster_metrics[pattern] = metrics

        # Build transition network
        transition_networks = {}
        for i in range(len(sequence) - 1):
            curr_task, curr_pattern = sequence[i]
            next_task, next_pattern = sequence[i + 1]

            if curr_pattern not in transition_networks:
                transition_networks[curr_pattern] = {}

            if next_pattern not in transition_networks[curr_pattern]:
                transition_networks[curr_pattern][next_pattern] = {
                    "count": 0,
                    "quality": 0.0,
                    "knowledge_transfer": 0.0
                }

            transition = transition_networks[curr_pattern][next_pattern]
            transition["count"] += 1
            transition["quality"] += self._calculate_transition_quality(
                curr_task, next_task, curr_pattern, next_pattern, {}
            )
            transition["knowledge_transfer"] += self._calculate_knowledge_transfer(
                curr_task, next_task, curr_pattern, next_pattern
            )

        # Average transition metrics
        for source in transition_networks:
            for target in transition_networks[source]:
                transition = transition_networks[source][target]
                transition["quality"] /= transition["count"]
                transition["knowledge_transfer"] /= transition["count"]

        return {
            "pattern_groups": pattern_groups,
            "cluster_metrics": cluster_metrics,
            "transition_networks": transition_networks
        }

    def _analyze_pattern_sequences(self, sequence: List[Tuple[Dict[str, Any], str]]) -> Dict[str, Any]:
        """Analyze recurring pattern sequences."""
        if len(sequence) < 2:
            return {
                "recurring_sequences": [],
                "sequence_frequency": {},
                "sequence_performance": {},
                "sequence_variations": []
            }

        # Extract pattern sequences of different lengths
        pattern_sequences = {}
        for length in range(2, min(len(sequence), 5)):  # Look for sequences up to length 4
            for i in range(len(sequence) - length + 1):
                seq = tuple(pattern for _, pattern in sequence[i:i+length])
                if seq not in pattern_sequences:
                    pattern_sequences[seq] = {
                        "count": 0,
                        "performance": 0.0,
                        "variations": []
                    }
                pattern_sequences[seq]["count"] += 1
                
                # Calculate sequence performance
                tasks = [task for task, _ in sequence[i:i+length]]
                performance = sum(
                    self._calculate_task_flow_efficiency(
                        tasks[j], tasks[j+1], {}
                    ) for j in range(len(tasks)-1)
                ) / (len(tasks)-1)
                pattern_sequences[seq]["performance"] += performance

                # Record task sequence as variation
                variation = [{
                    "task": task["id"],
                    "pattern": pattern
                } for task, pattern in sequence[i:i+length]]
                pattern_sequences[seq]["variations"].append(variation)

        # Calculate average performance for each sequence
        for seq_data in pattern_sequences.values():
            seq_data["performance"] /= seq_data["count"]

        # Find significant recurring sequences
        recurring_sequences = []
        for seq, data in pattern_sequences.items():
            if data["count"] > 1:  # Only include sequences that appear multiple times
                recurring_sequences.append({
                    "patterns": list(seq),
                    "frequency": data["count"],
                    "avg_performance": data["performance"],
                    "variations": data["variations"]
                })

        # Sort by frequency and performance
        recurring_sequences.sort(
            key=lambda x: (x["frequency"], x["avg_performance"]),
            reverse=True
        )

        return {
            "recurring_sequences": recurring_sequences,
            "sequence_frequency": {
                seq: data["count"]
                for seq, data in pattern_sequences.items()
            },
            "sequence_performance": {
                seq: data["performance"]
                for seq, data in pattern_sequences.items()
            },
            "sequence_variations": {
                seq: data["variations"]
                for seq, data in pattern_sequences.items()
            }
        }

    def _calculate_pattern_stability(self, pattern1: str, pattern2: str) -> float:
        """
        Calculate the stability between two patterns.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            Stability score between 0 and 1
        """
        # Calculate pattern similarity
        similarity = self._calculate_pattern_similarity(pattern1, pattern2)
        
        # Calculate complexity difference
        complexity_diff = abs(self._calculate_pattern_complexity(pattern1) - self._calculate_pattern_complexity(pattern2))
        
        # Calculate stability score
        stability = (similarity + (1 - complexity_diff)) / 2
        
        return max(0.0, min(stability, 1.0))
    
    def _calculate_pattern_novelty(self, pattern: str) -> float:
        """
        Calculate the novelty of a pattern.
        
        Args:
            pattern: Pattern to analyze
            
        Returns:
            Novelty score between 0 and 1
        """
        # Calculate pattern frequency
        frequency = self._get_pattern_frequency(pattern)
        
        # Calculate pattern complexity
        complexity = self._calculate_pattern_complexity(pattern)
        
        # Calculate novelty score
        novelty = (1 - frequency) * complexity
        
        return max(0.0, min(novelty, 1.0))
    
    def _recommend_pattern_sequence(self, current_sequence: List[Tuple[Dict[str, Any], str]], temporal_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Recommend optimal pattern sequences based on analysis.
        
        Args:
            current_sequence: Current sequence of (task, pattern) tuples
            temporal_context: Current temporal context
            
        Returns:
            List of recommended pattern sequences
        """
        recommendations = []
        
        # Get successful patterns
        successful_patterns = self._get_successful_patterns(temporal_context)
        
        # Get sequence templates
        templates = self._get_sequence_templates(temporal_context)
        
        # Generate recommendations
        for template in templates:
            # Apply template to current sequence
            recommended_sequence = self._apply_template(template, current_sequence, successful_patterns)
            
            # Calculate sequence score
            sequence_score = self._calculate_sequence_score(recommended_sequence, temporal_context)
            
            # Add to recommendations
            recommendations.append({
                "sequence": recommended_sequence,
                "score": sequence_score,
                "template": template,
                "analysis": self._analyze_sequence_patterns(recommended_sequence, temporal_context)
            })
        
        # Sort recommendations by score
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        
        return recommendations
    
    def _calculate_sequence_score(self, sequence: List[Tuple[Dict[str, Any], str]], temporal_context: Dict[str, Any]) -> float:
        """
        Calculate the overall score for a sequence.
        
        Args:
            sequence: Sequence of (task, pattern) tuples
            temporal_context: Current temporal context
            
        Returns:
            Sequence score between 0 and 1
        """
        # Get pattern analysis
        analysis = self._analyze_sequence_patterns(sequence, temporal_context)
        
        # Calculate weighted score
        weights = {
            "pattern_diversity": 0.2,
            "temporal_consistency": 0.2,
            "resource_utilization": 0.15,
            "cognitive_load": 0.15,
            "context_switching": 0.1,
            "dependency_satisfaction": 0.1,
            "innovation": 0.1
        }
        
        score = 0.0
        for metric, weight in weights.items():
            if metric in analysis["sequence_metrics"]:
                score += analysis["sequence_metrics"][metric] * weight
        
        return max(0.0, min(score, 1.0))

    def _calculate_cognitive_complexity(self, task: Dict[str, Any]) -> float:
        """
        Calculate the cognitive complexity of a task.
        
        Args:
            task: Task to analyze
            
        Returns:
            Cognitive complexity score between 0 and 1
        """
        # Calculate cognitive load
        cognitive_load = self._calculate_cognitive_load(task)
        
        # Calculate focus requirements
        focus_requirements = task.get("focus_requirements", 0.5)
        
        # Calculate mental effort
        mental_effort = task.get("mental_effort", 0.5)
        
        # Calculate cognitive complexity
        complexity = (cognitive_load + focus_requirements + mental_effort) / 3
        
        return max(0.0, min(complexity, 1.0))
    
    def _calculate_structural_complexity(self, pattern: str) -> float:
        """
        Calculate the structural complexity of a pattern.
        
        Args:
            pattern: Pattern to analyze
            
        Returns:
            Structural complexity score between 0 and 1
        """
        # Calculate pattern length
        pattern_length = len(pattern.split())
        
        # Calculate pattern depth
        pattern_depth = pattern.count("->")
        
        # Calculate pattern breadth
        pattern_breadth = len(set(pattern.split()))
        
        # Calculate structural complexity
        complexity = (pattern_length + pattern_depth + pattern_breadth) / 3
        
        return max(0.0, min(complexity, 1.0))
    
    def _calculate_temporal_complexity(self, task: Dict[str, Any]) -> float:
        """
        Calculate the temporal complexity of a task.
        
        Args:
            task: Task to analyze
            
        Returns:
            Temporal complexity score between 0 and 1
        """
        # Calculate time requirements
        time_requirements = task.get("time_requirements", 0.5)
        
        # Calculate temporal dependencies
        temporal_dependencies = len(task.get("temporal_dependencies", []))
        
        # Calculate schedule constraints
        schedule_constraints = len(task.get("schedule_constraints", []))
        
        # Calculate temporal complexity
        complexity = (time_requirements + temporal_dependencies + schedule_constraints) / 3
        
        return max(0.0, min(complexity, 1.0))
    
    def _calculate_resource_stability(self, task1: Dict[str, Any], task2: Dict[str, Any]) -> float:
        """
        Calculate the resource stability between two tasks.
        
        Args:
            task1: First task
            task2: Second task
            
        Returns:
            Resource stability score between 0 and 1
        """
        # Calculate resource overlap
        resource_overlap = self._calculate_resource_overlap(task1, task2)
        
        # Calculate resource continuity
        resource_continuity = self._calculate_resource_continuity(task1, task2)
        
        # Calculate resource stability
        stability = (resource_overlap + resource_continuity) / 2
        
        return max(0.0, min(stability, 1.0))
    
    def _calculate_temporal_stability(self, task1: Dict[str, Any], task2: Dict[str, Any]) -> float:
        """
        Calculate the temporal stability between two tasks.
        
        Args:
            task1: First task
            task2: Second task
            
        Returns:
            Temporal stability score between 0 and 1
        """
        # Calculate temporal compatibility
        temporal_compatibility = self._calculate_temporal_compatibility(task1, task2, {})
        
        # Calculate temporal continuity
        temporal_continuity = self._calculate_temporal_flow([(task1, ""), (task2, "")], {})
        
        # Calculate temporal stability
        stability = (temporal_compatibility + temporal_continuity) / 2
        
        return max(0.0, min(stability, 1.0))
    
    def _calculate_cognitive_stability(self, task1: Dict[str, Any], task2: Dict[str, Any]) -> float:
        """
        Calculate the cognitive stability between two tasks.
        
        Args:
            task1: First task
            task2: Second task
            
        Returns:
            Cognitive stability score between 0 and 1
        """
        # Calculate cognitive load difference
        load_diff = abs(self._calculate_cognitive_load(task1) - self._calculate_cognitive_load(task2))
        
        # Calculate focus continuity
        focus_continuity = self._calculate_focus_continuity(task1, task2)
        
        # Calculate cognitive stability
        stability = (1 - load_diff) * focus_continuity
        
        return max(0.0, min(stability, 1.0))
    
    def _calculate_creative_potential(self, task: Dict[str, Any]) -> float:
        """
        Calculate the creative potential of a task.
        
        Args:
            task: Task to analyze
            
        Returns:
            Creative potential score between 0 and 1
        """
        # Calculate innovation potential
        innovation = self._calculate_innovation_potential(task, {})
        
        # Calculate exploration potential
        exploration = self._calculate_exploration_score(task)
        
        # Calculate creative potential
        potential = (innovation + exploration) / 2
        
        return max(0.0, min(potential, 1.0))
    
    def _calculate_exploration_score(self, task: Dict[str, Any]) -> float:
        """
        Calculate the exploration score of a task.
        
        Args:
            task: Task to analyze
            
        Returns:
            Exploration score between 0 and 1
        """
        # Calculate novelty
        novelty = task.get("novelty", 0.5)
        
        # Calculate uncertainty
        uncertainty = task.get("uncertainty", 0.5)
        
        # Calculate exploration score
        score = (novelty + uncertainty) / 2
        
        return max(0.0, min(score, 1.0))
    
    def _calculate_adaptation_capacity(self, task: Dict[str, Any]) -> float:
        """
        Calculate the adaptation capacity of a task.
        
        Args:
            task: Task to analyze
            
        Returns:
            Adaptation capacity score between 0 and 1
        """
        # Calculate adaptability
        adaptability = self._calculate_adaptability(task, {})
        
        # Calculate resilience
        resilience = self._calculate_resilience(task, task, {})
        
        # Calculate adaptation capacity
        capacity = (adaptability + resilience) / 2
        
        return max(0.0, min(capacity, 1.0))

    def _calculate_semantic_complexity(self, task: Dict[str, Any]) -> float:
        """
        Calculate the semantic complexity of a task.
        
        Args:
            task: Task to analyze
            
        Returns:
            Semantic complexity score between 0 and 1
        """
        # Calculate semantic similarity
        semantic_similarity = self._calculate_semantic_similarity(task, task)
        
        # Calculate semantic diversity
        semantic_diversity = len(set(task.get("keywords", [])))
        
        # Calculate semantic complexity
        complexity = (semantic_similarity + semantic_diversity) / 2
        
        return max(0.0, min(complexity, 1.0))
    
    def _calculate_dependency_complexity(self, task: Dict[str, Any]) -> float:
        """
        Calculate the dependency complexity of a task.
        
        Args:
            task: Task to analyze
            
        Returns:
            Dependency complexity score between 0 and 1
        """
        # Calculate dependency count
        dependency_count = len(task.get("dependencies", []))
        
        # Calculate dependency depth
        dependency_depth = task.get("dependency_depth", 0)
        
        # Calculate dependency complexity
        complexity = (dependency_count + dependency_depth) / 2
        
        return max(0.0, min(complexity, 1.0))
    
    def _calculate_context_complexity(self, task: Dict[str, Any]) -> float:
        """
        Calculate the context complexity of a task.
        
        Args:
            task: Task to analyze
            
        Returns:
            Context complexity score between 0 and 1
        """
        # Calculate context size
        context_size = len(task.get("context", {}))
        
        # Calculate context diversity
        context_diversity = len(set(task.get("context", {}).values()))
        
        # Calculate context complexity
        complexity = (context_size + context_diversity) / 2
        
        return max(0.0, min(complexity, 1.0))
    
    def _calculate_semantic_stability(self, task1: Dict[str, Any], task2: Dict[str, Any]) -> float:
        """
        Calculate the semantic stability between two tasks.
        
        Args:
            task1: First task
            task2: Second task
            
        Returns:
            Semantic stability score between 0 and 1
        """
        # Calculate semantic similarity
        semantic_similarity = self._calculate_semantic_similarity(task1, task2)
        
        # Calculate semantic continuity
        semantic_continuity = self._calculate_semantic_continuity(task1, task2)
        
        # Calculate semantic stability
        stability = (semantic_similarity + semantic_continuity) / 2
        
        return max(0.0, min(stability, 1.0))
    
    def _calculate_dependency_stability(self, task1: Dict[str, Any], task2: Dict[str, Any]) -> float:
        """
        Calculate the dependency stability between two tasks.
        
        Args:
            task1: First task
            task2: Second task
            
        Returns:
            Dependency stability score between 0 and 1
        """
        # Calculate dependency overlap
        dependency_overlap = len(set(task1.get("dependencies", [])) & set(task2.get("dependencies", [])))
        
        # Calculate dependency continuity
        dependency_continuity = self._calculate_dependency_continuity(task1, task2)
        
        # Calculate dependency stability
        stability = (dependency_overlap + dependency_continuity) / 2
        
        return max(0.0, min(stability, 1.0))
    
    def _calculate_context_stability(self, task1: Dict[str, Any], task2: Dict[str, Any]) -> float:
        """
        Calculate the context stability between two tasks.
        
        Args:
            task1: First task
            task2: Second task
            
        Returns:
            Context stability score between 0 and 1
        """
        # Calculate context overlap
        context_overlap = len(set(task1.get("context", {}).items()) & set(task2.get("context", {}).items()))
        
        # Calculate context continuity
        context_continuity = self._calculate_context_continuity(task1, task2)
        
        # Calculate context stability
        stability = (context_overlap + context_continuity) / 2
        
        return max(0.0, min(stability, 1.0))
    
    def _calculate_semantic_novelty(self, task: Dict[str, Any]) -> float:
        """
        Calculate the semantic novelty of a task.
        
        Args:
            task: Task to analyze
            
        Returns:
            Semantic novelty score between 0 and 1
        """
        # Calculate semantic diversity
        semantic_diversity = len(set(task.get("keywords", [])))
        
        # Calculate semantic uniqueness
        semantic_uniqueness = 1 - self._calculate_semantic_similarity(task, task)
        
        # Calculate semantic novelty
        novelty = (semantic_diversity + semantic_uniqueness) / 2
        
        return max(0.0, min(novelty, 1.0))
    
    def _calculate_context_novelty(self, task: Dict[str, Any]) -> float:
        """
        Calculate the context novelty of a task.
        
        Args:
            task: Task to analyze
            
        Returns:
            Context novelty score between 0 and 1
        """
        # Calculate context diversity
        context_diversity = len(set(task.get("context", {}).values()))
        
        # Calculate context uniqueness
        context_uniqueness = 1 - self._calculate_context_similarity(task, task)
        
        # Calculate context novelty
        novelty = (context_diversity + context_uniqueness) / 2
        
        return max(0.0, min(novelty, 1.0))
    
    def _calculate_evolution_potential(self, task: Dict[str, Any]) -> float:
        """
        Calculate the evolution potential of a task.
        
        Args:
            task: Task to analyze
            
        Returns:
            Evolution potential score between 0 and 1
        """
        # Calculate adaptability
        adaptability = self._calculate_adaptability(task, {})
        
        # Calculate innovation potential
        innovation = self._calculate_innovation_potential(task, {})
        
        # Calculate evolution potential
        potential = (adaptability + innovation) / 2
        
        return max(0.0, min(potential, 1.0))

    def _calculate_interaction_complexity(self, task: Dict[str, Any]) -> float:
        """
        Calculate the interaction complexity of a task.
        
        Args:
            task: Task to analyze
            
        Returns:
            Interaction complexity score between 0 and 1
        """
        # Calculate interaction requirements
        interaction_requirements = len(task.get("interaction_requirements", []))
        
        # Calculate interaction depth
        interaction_depth = task.get("interaction_depth", 0)
        
        # Calculate interaction complexity
        complexity = (interaction_requirements + interaction_depth) / 2
        
        return max(0.0, min(complexity, 1.0))
    
    def _calculate_adaptation_complexity(self, task: Dict[str, Any]) -> float:
        """
        Calculate the adaptation complexity of a task.
        
        Args:
            task: Task to analyze
            
        Returns:
            Adaptation complexity score between 0 and 1
        """
        # Calculate adaptation requirements
        adaptation_requirements = len(task.get("adaptation_requirements", []))
        
        # Calculate adaptation depth
        adaptation_depth = task.get("adaptation_depth", 0)
        
        # Calculate adaptation complexity
        complexity = (adaptation_requirements + adaptation_depth) / 2
        
        return max(0.0, min(complexity, 1.0))
    
    def _calculate_evolution_complexity(self, task: Dict[str, Any]) -> float:
        """
        Calculate the evolution complexity of a task.
        
        Args:
            task: Task to analyze
            
        Returns:
            Evolution complexity score between 0 and 1
        """
        # Calculate evolution requirements
        evolution_requirements = len(task.get("evolution_requirements", []))
        
        # Calculate evolution depth
        evolution_depth = task.get("evolution_depth", 0)
        
        # Calculate evolution complexity
        complexity = (evolution_requirements + evolution_depth) / 2
        
        return max(0.0, min(complexity, 1.0))
    
    def _calculate_interaction_stability(self, task1: Dict[str, Any], task2: Dict[str, Any]) -> float:
        """
        Calculate the interaction stability between two tasks.
        
        Args:
            task1: First task
            task2: Second task
            
        Returns:
            Interaction stability score between 0 and 1
        """
        # Calculate interaction overlap
        interaction_overlap = len(set(task1.get("interaction_requirements", [])) & set(task2.get("interaction_requirements", [])))
        
        # Calculate interaction continuity
        interaction_continuity = self._calculate_interaction_continuity(task1, task2)
        
        # Calculate interaction stability
        stability = (interaction_overlap + interaction_continuity) / 2
        
        return max(0.0, min(stability, 1.0))
    
    def _calculate_adaptation_stability(self, task1: Dict[str, Any], task2: Dict[str, Any]) -> float:
        """
        Calculate the adaptation stability between two tasks.
        
        Args:
            task1: First task
            task2: Second task
            
        Returns:
            Adaptation stability score between 0 and 1
        """
        # Calculate adaptation overlap
        adaptation_overlap = len(set(task1.get("adaptation_requirements", [])) & set(task2.get("adaptation_requirements", [])))
        
        # Calculate adaptation continuity
        adaptation_continuity = self._calculate_adaptation_continuity(task1, task2)
        
        # Calculate adaptation stability
        stability = (adaptation_overlap + adaptation_continuity) / 2
        
        return max(0.0, min(stability, 1.0))
    
    def _calculate_evolution_stability(self, task1: Dict[str, Any], task2: Dict[str, Any]) -> float:
        """
        Calculate the evolution stability between two tasks.
        
        Args:
            task1: First task
            task2: Second task
            
        Returns:
            Evolution stability score between 0 and 1
        """
        # Calculate evolution overlap
        evolution_overlap = len(set(task1.get("evolution_requirements", [])) & set(task2.get("evolution_requirements", [])))
        
        # Calculate evolution continuity
        evolution_continuity = self._calculate_evolution_continuity(task1, task2)
        
        # Calculate evolution stability
        stability = (evolution_overlap + evolution_continuity) / 2
        
        return max(0.0, min(stability, 1.0))
    
    def _calculate_interaction_novelty(self, task: Dict[str, Any]) -> float:
        """
        Calculate the interaction novelty of a task.
        
        Args:
            task: Task to analyze
            
        Returns:
            Interaction novelty score between 0 and 1
        """
        # Calculate interaction diversity
        interaction_diversity = len(set(task.get("interaction_requirements", [])))
        
        # Calculate interaction uniqueness
        interaction_uniqueness = 1 - self._calculate_interaction_similarity(task, task)
        
        # Calculate interaction novelty
        novelty = (interaction_diversity + interaction_uniqueness) / 2
        
        return max(0.0, min(novelty, 1.0))
    
    def _calculate_adaptation_novelty(self, task: Dict[str, Any]) -> float:
        """
        Calculate the adaptation novelty of a task.
        
        Args:
            task: Task to analyze
            
        Returns:
            Adaptation novelty score between 0 and 1
        """
        # Calculate adaptation diversity
        adaptation_diversity = len(set(task.get("adaptation_requirements", [])))
        
        # Calculate adaptation uniqueness
        adaptation_uniqueness = 1 - self._calculate_adaptation_similarity(task, task)
        
        # Calculate adaptation novelty
        novelty = (adaptation_diversity + adaptation_uniqueness) / 2
        
        return max(0.0, min(novelty, 1.0))
    
    def _calculate_evolution_novelty(self, task: Dict[str, Any]) -> float:
        """
        Calculate the evolution novelty of a task.
        
        Args:
            task: Task to analyze
            
        Returns:
            Evolution novelty score between 0 and 1
        """
        # Calculate evolution diversity
        evolution_diversity = len(set(task.get("evolution_requirements", [])))
        
        # Calculate evolution uniqueness
        evolution_uniqueness = 1 - self._calculate_evolution_similarity(task, task)
        
        # Calculate evolution novelty
        novelty = (evolution_diversity + evolution_uniqueness) / 2
        
        return max(0.0, min(novelty, 1.0))


class StrategyEnsemble:
    """Ensemble of exploration strategies."""

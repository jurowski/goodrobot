import json
import os
import random
from typing import Any, Dict, List

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
        self._load_prior_distributions()

    def _load_prior_distributions(self):
        """Load prior distributions for tasks."""
        dist_file = os.path.join(self.settings.data_dir, "thompson_priors.json")
        if os.path.exists(dist_file):
            with open(dist_file, "r") as f:
                self.prior_distributions = json.load(f)
        else:
            self.prior_distributions = {}

    def _save_prior_distributions(self):
        """Save prior distributions for tasks."""
        dist_file = os.path.join(self.settings.data_dir, "thompson_priors.json")
        with open(dist_file, "w") as f:
            json.dump(self.prior_distributions, f, indent=2)

    def select_tasks(
        self, state: Dict[str, Any], possible_tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Select tasks using Thompson sampling."""
        sampled_tasks = []
        for task in possible_tasks:
            task_id = task.get("id", str(hash(str(task))))
            if task_id not in self.prior_distributions:
                self.prior_distributions[task_id] = {"alpha": 1, "beta": 1}

            # Sample from beta distribution
            alpha = self.prior_distributions[task_id]["alpha"]
            beta = self.prior_distributions[task_id]["beta"]
            sample = random.betavariate(alpha, beta)

            sampled_tasks.append((task, sample))

        # Select top tasks based on samples
        selected_tasks = [
            task
            for task, _ in sorted(sampled_tasks, key=lambda x: x[1], reverse=True)[:3]
        ]

        self._save_prior_distributions()
        return selected_tasks

    def calculate_state_compatibility(self, state: Dict[str, Any]) -> float:
        """Calculate state compatibility for Thompson sampling."""
        # More compatible when we have historical data
        return min(len(self.prior_distributions) / 100.0, 1.0)

    def cleanup(self):
        """Clean up resources."""
        self._save_prior_distributions()


class LinUCBStrategy(BaseStrategy):
    """Linear Upper Confidence Bound exploration strategy."""

    def __init__(self, settings: Settings):
        """Initialize the LinUCB strategy."""
        super().__init__(settings)
        self.alpha = 1.0  # Exploration parameter
        self._load_model_parameters()

    def _load_model_parameters(self):
        """Load model parameters."""
        param_file = os.path.join(self.settings.data_dir, "linucb_params.json")
        if os.path.exists(param_file):
            with open(param_file, "r") as f:
                self.model_params = json.load(f)
        else:
            self.model_params = {
                "A": {},  # Feature covariance matrix
                "b": {},  # Feature-reward vector
                "theta": {},  # Model parameters
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
        for task in possible_tasks:
            # Extract features
            features = self._extract_features(task, state)

            # Calculate UCB score
            score = self._calculate_ucb_score(features)

            scored_tasks.append((task, score))

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

        # State features
        features.append(state.get("task_stats", {}).get("pending_tasks", 0) / 10.0)
        features.append(
            state.get("temporal_features", {}).get("avg_time_until_due", 0) / 86400.0
        )

        return features

    def _calculate_ucb_score(self, features: List[float]) -> float:
        """Calculate UCB score for features."""
        # Implementation of LinUCB algorithm
        return sum(features)  # Simplified for now

    def calculate_state_compatibility(self, state: Dict[str, Any]) -> float:
        """Calculate state compatibility for LinUCB."""
        # More compatible when we have many features
        return min(len(self.model_params["theta"]) / 50.0, 1.0)

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
        self._load_domain_knowledge()

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
            }

    def _save_domain_knowledge(self):
        """Save domain knowledge."""
        domain_file = os.path.join(self.settings.data_dir, "domain_knowledge.json")
        with open(domain_file, "w") as f:
            json.dump(self.domain_knowledge, f, indent=2)

    def select_tasks(
        self, state: Dict[str, Any], possible_tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Select tasks using domain-guided exploration."""
        scored_tasks = []
        for task in possible_tasks:
            # Calculate domain score
            domain_score = self._calculate_domain_score(task, state)

            scored_tasks.append((task, domain_score))

        # Select top tasks based on domain scores
        selected_tasks = [
            task
            for task, _ in sorted(scored_tasks, key=lambda x: x[1], reverse=True)[:3]
        ]

        self._save_domain_knowledge()
        return selected_tasks

    def _calculate_domain_score(
        self, task: Dict[str, Any], state: Dict[str, Any]
    ) -> float:
        """Calculate domain score for a task."""
        score = 0.0

        # Check task patterns
        task_text = f"{task.get('title', '')} {task.get('description', '')}".lower()
        for pattern, pattern_score in self.domain_knowledge["task_patterns"].items():
            if pattern in task_text:
                score += pattern_score

        # Check successful sequences
        for sequence in self.domain_knowledge["successful_sequences"]:
            if task.get("id") in sequence:
                score += 0.2

        # Apply domain rules
        for rule in self.domain_knowledge["domain_rules"]:
            if self._evaluate_rule(rule, task, state):
                score += rule.get("score", 0.1)

        return min(score, 1.0)

    def _evaluate_rule(
        self, rule: Dict[str, Any], task: Dict[str, Any], state: Dict[str, Any]
    ) -> bool:
        """Evaluate a domain rule."""
        # Implementation of rule evaluation
        return True  # Simplified for now

    def calculate_state_compatibility(self, state: Dict[str, Any]) -> float:
        """Calculate state compatibility for domain-guided exploration."""
        # More compatible when we have domain knowledge
        return min(
            (
                len(self.domain_knowledge["task_patterns"])
                + len(self.domain_knowledge["successful_sequences"])
                + len(self.domain_knowledge["domain_rules"])
            )
            / 30.0,
            1.0,
        )

    def cleanup(self):
        """Clean up resources."""
        self._save_domain_knowledge()


class StrategyEnsemble:
    """Ensemble of exploration strategies."""

    def __init__(self, settings: Settings):
        """Initialize the strategy ensemble."""
        self.settings = settings
        self.strategies = [
            EpsilonGreedyStrategy(settings),
            ThompsonSamplingStrategy(settings),
            LinUCBStrategy(settings),
            ProgressiveValidationStrategy(settings),
            DomainGuidedExploration(settings),
        ]
        self._ensure_data_directory()

    def _ensure_data_directory(self):
        """Ensure the data directory exists."""
        os.makedirs(self.settings.data_dir, exist_ok=True)

    def generate_variations(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate task variations using the ensemble of strategies."""
        all_variations = []

        for strategy in self.strategies:
            # Generate variations using each strategy
            variations = strategy.select_tasks({}, tasks)
            all_variations.extend(variations)

        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for task in all_variations:
            task_id = task.get("id", str(hash(str(task))))
            if task_id not in seen:
                seen.add(task_id)
                unique_variations.append(task)

        return unique_variations

    def cleanup(self):
        """Clean up resources."""
        for strategy in self.strategies:
            strategy.cleanup()

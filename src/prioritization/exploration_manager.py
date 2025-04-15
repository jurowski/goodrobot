import json
import os
from datetime import datetime
from typing import Any, Dict

from config.settings import Settings
from src.prioritization.strategy_ensemble import (
    DomainGuidedExploration,
    EpsilonGreedyStrategy,
    LinUCBStrategy,
    ProgressiveValidationStrategy,
    ThompsonSamplingStrategy,
)


class AdaptiveExplorationManager:
    """Manages exploration strategies for the RL model."""

    def __init__(self, settings: Settings):
        """Initialize the exploration manager."""
        self.settings = settings
        self.strategies = [
            EpsilonGreedyStrategy(settings),
            ThompsonSamplingStrategy(settings),
            LinUCBStrategy(settings),
            ProgressiveValidationStrategy(settings),
            DomainGuidedExploration(settings),
        ]
        self._ensure_data_directory()
        self._load_strategy_performance()

    def _ensure_data_directory(self):
        """Ensure the data directory exists."""
        os.makedirs(self.settings.data_dir, exist_ok=True)

    def _load_strategy_performance(self):
        """Load historical strategy performance data."""
        perf_file = os.path.join(self.settings.data_dir, "strategy_performance.json")
        if os.path.exists(perf_file):
            with open(perf_file, "r") as f:
                self.strategy_performance = json.load(f)
        else:
            self.strategy_performance = {
                strategy.__class__.__name__: {
                    "success_count": 0,
                    "total_count": 0,
                    "last_used": None,
                }
                for strategy in self.strategies
            }

    def _save_strategy_performance(self):
        """Save strategy performance data."""
        perf_file = os.path.join(self.settings.data_dir, "strategy_performance.json")
        with open(perf_file, "w") as f:
            json.dump(self.strategy_performance, f, indent=2)

    def select_strategy(self, state: Dict[str, Any]) -> Any:
        """Select the most appropriate exploration strategy for the given state."""
        # Calculate strategy scores
        strategy_scores = {}
        for strategy in self.strategies:
            strategy_name = strategy.__class__.__name__
            perf = self.strategy_performance[strategy_name]

            # Calculate success rate
            success_rate = (
                perf["success_count"] / perf["total_count"]
                if perf["total_count"] > 0
                else 0.5
            )

            # Calculate recency factor
            last_used = perf.get("last_used")
            recency_factor = 1.0
            if last_used:
                last_used_time = datetime.fromisoformat(last_used)
                hours_since_last_use = (
                    datetime.now() - last_used_time
                ).total_seconds() / 3600
                recency_factor = 1.0 / (1.0 + hours_since_last_use)

            # Calculate state compatibility
            state_compatibility = strategy.calculate_state_compatibility(state)

            # Combine factors
            strategy_scores[strategy_name] = (
                (success_rate * 0.4)
                + (recency_factor * 0.3)
                + (state_compatibility * 0.3)
            )

        # Select strategy with highest score
        selected_strategy_name = max(strategy_scores, key=strategy_scores.get)
        selected_strategy = next(
            s for s in self.strategies if s.__class__.__name__ == selected_strategy_name
        )

        # Update performance data
        self.strategy_performance[selected_strategy_name][
            "last_used"
        ] = datetime.now().isoformat()
        self._save_strategy_performance()

        return selected_strategy

    def update_strategy_performance(self, strategy_name: str, success: bool):
        """Update the performance data for a strategy."""
        if strategy_name in self.strategy_performance:
            perf = self.strategy_performance[strategy_name]
            perf["total_count"] += 1
            if success:
                perf["success_count"] += 1
            self._save_strategy_performance()

    def cleanup(self):
        """Clean up resources."""
        self._save_strategy_performance()
        for strategy in self.strategies:
            strategy.cleanup()

"""
User Behavior Perturbation Module for XAI Robustness Evaluation.

Implements 5 types of perturbations analogous to image corruptions:
1. Random Click Noise (analogous to Gaussian noise)
2. Temporal Shuffle (analogous to motion blur)
3. Behavior Dilution (analogous to brightness reduction)
4. Category Drift (analogous to contrast change)
5. Missing Values (analogous to pixelation)

Each perturbation has 5 severity levels (1-5).
"""

import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import copy

# Reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class PerturbationType(Enum):
    """Types of user behavior perturbations."""
    NOISE = "noise"
    SHUFFLE = "shuffle"
    DILUTION = "dilution"
    DRIFT = "drift"
    MISSING = "missing"


@dataclass
class PerturbationConfig:
    """Configuration for perturbations."""
    perturbation_type: PerturbationType
    severity: int  # 1-5
    random_seed: int = 42


class BasePerturbation(ABC):
    """Base class for user behavior perturbations."""

    # Severity level ratios
    SEVERITY_RATIOS = {
        1: 0.05,  # 5%
        2: 0.10,  # 10%
        3: 0.20,  # 20%
        4: 0.35,  # 35%
        5: 0.50,  # 50%
    }

    def __init__(self, severity: int = 1, random_seed: int = 42):
        """
        Initialize perturbation.

        Args:
            severity: Severity level (1-5)
            random_seed: Random seed for reproducibility
        """
        assert 1 <= severity <= 5, "Severity must be between 1 and 5"
        self.severity = severity
        self.random_seed = random_seed
        self.ratio = self.SEVERITY_RATIOS[severity]
        random.seed(random_seed)
        np.random.seed(random_seed)

    @abstractmethod
    def apply(self, history: List[Dict]) -> List[Dict]:
        """Apply perturbation to user history."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return perturbation name."""
        pass

    def __repr__(self):
        return f"{self.name}(severity={self.severity})"


class RandomClickNoise(BasePerturbation):
    """
    Add random click noise to user history.

    Analogous to Gaussian noise in images.
    Simulates accidental clicks or noisy user behavior data.
    """

    def __init__(
        self,
        severity: int = 1,
        random_seed: int = 42,
        all_items: Optional[List[str]] = None,
        all_categories: Optional[List[str]] = None
    ):
        super().__init__(severity, random_seed)
        self.all_items = all_items or []
        self.all_categories = all_categories or ['Electronics', 'Clothing', 'Books', 'Home', 'Sports']

    @property
    def name(self) -> str:
        return "RandomClickNoise"

    def apply(self, history: List[Dict]) -> List[Dict]:
        """Add random items to history."""
        perturbed = copy.deepcopy(history)
        num_noise = max(1, int(len(history) * self.ratio))

        # Generate random noise interactions
        existing_items = {h.get('item_id', '') for h in history}
        available_items = [i for i in self.all_items if i not in existing_items]

        if not available_items:
            available_items = [f"noise_item_{i}" for i in range(num_noise)]

        for _ in range(num_noise):
            noise_item = random.choice(available_items) if available_items else f"noise_{random.randint(0, 10000)}"
            noise_interaction = {
                'item_id': noise_item,
                'category': random.choice(self.all_categories),
                'rating': random.randint(1, 5),
                'timestamp': random.randint(0, 1000000),
                'is_noise': True,  # Mark as noise for analysis
            }
            # Insert at random position
            insert_pos = random.randint(0, len(perturbed))
            perturbed.insert(insert_pos, noise_interaction)

        return perturbed


class TemporalShuffle(BasePerturbation):
    """
    Shuffle temporal order of interactions.

    Analogous to motion blur in images.
    Simulates data latency or timestamp errors.
    """

    SEVERITY_RATIOS = {
        1: 0.10,  # 10% shuffled
        2: 0.25,  # 25% shuffled
        3: 0.50,  # 50% shuffled
        4: 0.75,  # 75% shuffled
        5: 1.00,  # 100% shuffled
    }

    @property
    def name(self) -> str:
        return "TemporalShuffle"

    def apply(self, history: List[Dict]) -> List[Dict]:
        """Shuffle portion of history timestamps."""
        perturbed = copy.deepcopy(history)
        n = len(perturbed)
        num_shuffle = max(2, int(n * self.ratio))

        # Select random indices to shuffle
        indices = random.sample(range(n), min(num_shuffle, n))

        # Get items at these indices
        items_to_shuffle = [perturbed[i] for i in indices]
        random.shuffle(items_to_shuffle)

        # Put shuffled items back
        for i, idx in enumerate(indices):
            perturbed[idx] = items_to_shuffle[i]

        return perturbed


class BehaviorDilution(BasePerturbation):
    """
    Remove random interactions from history.

    Analogous to brightness reduction in images.
    Simulates cold start or sparse data scenarios.
    """

    SEVERITY_RATIOS = {
        1: 0.10,  # Remove 10%
        2: 0.20,  # Remove 20%
        3: 0.30,  # Remove 30%
        4: 0.40,  # Remove 40%
        5: 0.50,  # Remove 50%
    }

    @property
    def name(self) -> str:
        return "BehaviorDilution"

    def apply(self, history: List[Dict]) -> List[Dict]:
        """Remove random interactions."""
        if len(history) <= 2:
            return copy.deepcopy(history)

        num_remove = max(1, int(len(history) * self.ratio))
        num_keep = len(history) - num_remove

        # Randomly select items to keep
        keep_indices = sorted(random.sample(range(len(history)), num_keep))
        perturbed = [history[i] for i in keep_indices]

        return perturbed


class CategoryDrift(BasePerturbation):
    """
    Add items from different categories.

    Analogous to contrast change in images.
    Simulates interest drift or category exploration.
    """

    SEVERITY_RATIOS = {
        1: 0.10,  # 10% from different categories
        2: 0.20,  # 20%
        3: 0.30,  # 30%
        4: 0.40,  # 40%
        5: 0.50,  # 50%
    }

    def __init__(
        self,
        severity: int = 1,
        random_seed: int = 42,
        drift_categories: Optional[List[str]] = None
    ):
        super().__init__(severity, random_seed)
        self.drift_categories = drift_categories or ['Automotive', 'Garden', 'Toys', 'Beauty', 'Pet']

    @property
    def name(self) -> str:
        return "CategoryDrift"

    def apply(self, history: List[Dict]) -> List[Dict]:
        """Add items from different (drift) categories."""
        perturbed = copy.deepcopy(history)
        num_drift = max(1, int(len(history) * self.ratio))

        for _ in range(num_drift):
            drift_item = {
                'item_id': f"drift_item_{random.randint(0, 10000)}",
                'category': random.choice(self.drift_categories),
                'rating': random.randint(1, 5),
                'timestamp': random.randint(0, 1000000),
                'is_drift': True,
            }
            insert_pos = random.randint(0, len(perturbed))
            perturbed.insert(insert_pos, drift_item)

        return perturbed


class MissingValues(BasePerturbation):
    """
    Remove specific interaction types (views, ratings, etc.).

    Analogous to pixelation in images.
    Simulates incomplete data collection.
    """

    SEVERITY_RATIOS = {
        1: 0.10,  # 10% missing
        2: 0.20,  # 20% missing
        3: 0.35,  # 35% missing
        4: 0.50,  # 50% missing
        5: 0.70,  # 70% missing
    }

    def __init__(
        self,
        severity: int = 1,
        random_seed: int = 42,
        fields_to_remove: Optional[List[str]] = None
    ):
        super().__init__(severity, random_seed)
        self.fields_to_remove = fields_to_remove or ['rating', 'timestamp', 'category']

    @property
    def name(self) -> str:
        return "MissingValues"

    def apply(self, history: List[Dict]) -> List[Dict]:
        """Remove random fields from interactions."""
        perturbed = copy.deepcopy(history)
        num_affected = max(1, int(len(history) * self.ratio))

        # Select random interactions to affect
        affected_indices = random.sample(range(len(perturbed)), min(num_affected, len(perturbed)))

        for idx in affected_indices:
            # Remove random field(s)
            fields = [f for f in self.fields_to_remove if f in perturbed[idx]]
            if fields:
                field_to_remove = random.choice(fields)
                perturbed[idx][field_to_remove] = None
                perturbed[idx]['has_missing'] = True

        return perturbed


class PerturbationFactory:
    """Factory for creating perturbations."""

    PERTURBATION_CLASSES = {
        PerturbationType.NOISE: RandomClickNoise,
        PerturbationType.SHUFFLE: TemporalShuffle,
        PerturbationType.DILUTION: BehaviorDilution,
        PerturbationType.DRIFT: CategoryDrift,
        PerturbationType.MISSING: MissingValues,
    }

    @classmethod
    def create(
        cls,
        perturbation_type: PerturbationType,
        severity: int = 1,
        random_seed: int = 42,
        **kwargs
    ) -> BasePerturbation:
        """
        Create a perturbation instance.

        Args:
            perturbation_type: Type of perturbation
            severity: Severity level (1-5)
            random_seed: Random seed
            **kwargs: Additional arguments for specific perturbation types

        Returns:
            Perturbation instance
        """
        perturbation_class = cls.PERTURBATION_CLASSES.get(perturbation_type)
        if not perturbation_class:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")

        return perturbation_class(severity=severity, random_seed=random_seed, **kwargs)

    @classmethod
    def create_all(
        cls,
        severity: int = 1,
        random_seed: int = 42,
        **kwargs
    ) -> Dict[PerturbationType, BasePerturbation]:
        """Create all perturbation types with given severity."""
        return {
            ptype: cls.create(ptype, severity, random_seed, **kwargs)
            for ptype in PerturbationType
        }

    @classmethod
    def get_all_configs(cls) -> List[Tuple[PerturbationType, int]]:
        """Get all perturbation type and severity combinations."""
        return [
            (ptype, severity)
            for ptype in PerturbationType
            for severity in range(1, 6)
        ]


def apply_perturbation_grid(
    history: List[Dict],
    all_items: Optional[List[str]] = None
) -> Dict[str, Dict[int, List[Dict]]]:
    """
    Apply all perturbations at all severity levels.

    Returns:
        {perturbation_type: {severity: perturbed_history}}
    """
    results = {}

    for ptype in PerturbationType:
        results[ptype.value] = {}
        for severity in range(1, 6):
            if ptype == PerturbationType.NOISE:
                perturbation = PerturbationFactory.create(
                    ptype, severity,
                    all_items=all_items or []
                )
            else:
                perturbation = PerturbationFactory.create(ptype, severity)

            results[ptype.value][severity] = perturbation.apply(history)

    return results


def demo():
    """Demo perturbation functionality."""
    # Create sample history
    sample_history = [
        {'item_id': f'item_{i}', 'category': 'Electronics', 'rating': 4, 'timestamp': i * 1000}
        for i in range(20)
    ]

    print("Original history length:", len(sample_history))
    print(f"First 3 items: {sample_history[:3]}")
    print()

    # Test each perturbation
    for ptype in PerturbationType:
        print(f"=== {ptype.value.upper()} ===")
        for severity in [1, 3, 5]:
            perturbation = PerturbationFactory.create(ptype, severity)
            perturbed = perturbation.apply(sample_history)
            print(f"  Severity {severity}: length={len(perturbed)}")
        print()


if __name__ == "__main__":
    demo()

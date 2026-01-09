"""
User Behavior Perturbation Module for XAI Robustness Evaluation.

Implements perturbations analogous to image corruptions for testing
the robustness of LLM-generated recommendation explanations.
"""

from .perturbation_factory import (
    PerturbationType,
    PerturbationConfig,
    BasePerturbation,
    RandomClickNoise,
    TemporalShuffle,
    BehaviorDilution,
    CategoryDrift,
    MissingValues,
    PerturbationFactory,
    apply_perturbation_grid,
)

__all__ = [
    'PerturbationType',
    'PerturbationConfig',
    'BasePerturbation',
    'RandomClickNoise',
    'TemporalShuffle',
    'BehaviorDilution',
    'CategoryDrift',
    'MissingValues',
    'PerturbationFactory',
    'apply_perturbation_grid',
]

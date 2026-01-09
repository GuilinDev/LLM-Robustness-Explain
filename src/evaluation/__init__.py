"""
Evaluation Module for XAI Robustness in Recommendation Systems.
"""

from .robustness_metrics import (
    RobustnessResult,
    TextSimilarity,
    RobustnessEvaluator,
    RobustnessAnalyzer,
)

__all__ = [
    'RobustnessResult',
    'TextSimilarity',
    'RobustnessEvaluator',
    'RobustnessAnalyzer',
]

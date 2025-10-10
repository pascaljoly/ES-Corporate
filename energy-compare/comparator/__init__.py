# comparator/__init__.py
"""
Energy comparison module for ML models.

This module provides tools to compare multiple ML models based on their
energy consumption, CO2 emissions, and performance metrics.
"""

from .core import (
    ModelComparator,
    ModelComparison,
    ComparisonResult,
    ComparisonMetric,
    ComparisonError
)

__all__ = [
    "ModelComparator",
    "ModelComparison", 
    "ComparisonResult",
    "ComparisonMetric",
    "ComparisonError"
]

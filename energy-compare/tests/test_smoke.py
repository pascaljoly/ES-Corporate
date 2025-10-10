# tests/test_smoke.py
"""Quick sanity checks - run these first to verify basic setup"""
import pytest

def test_imports():
    """Test that all modules can be imported"""
    from scorer.core import ModelEnergyScorer, ScoringResult
    from scorer.core import ValidationError, MeasurementError
    assert True

def test_scorer_instantiation():
    """Test that scorer can be created"""
    from scorer.core import ModelEnergyScorer
    scorer = ModelEnergyScorer()
    assert scorer is not None

def test_supported_tasks_defined():
    """Test that supported tasks list exists"""
    from scorer.core import SUPPORTED_TASKS
    assert isinstance(SUPPORTED_TASKS, list)
    assert len(SUPPORTED_TASKS) > 0
    assert "text_generation" in SUPPORTED_TASKS

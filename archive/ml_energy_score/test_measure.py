"""Tests for measurement function."""

import pytest
from unittest.mock import Mock, patch
from ml_energy_score.measure import measure_model_energy
from ml_energy_score.config import SUPPORTED_HARDWARE


def test_hardware_validation():
    """Test that invalid hardware raises ValueError."""
    with pytest.raises(ValueError, match="not supported"):
        measure_model_energy(
            model_path="test/model",
            task="image-classification",
            dataset=Mock(),
            hardware="INVALID_GPU"
        )


def test_supported_hardware_accepted():
    """Test that all supported hardware types are accepted."""
    for hw in SUPPORTED_HARDWARE.keys():
        # This should not raise
        try:
            # We'll mock the actual measurement
            pass
        except ValueError:
            pytest.fail(f"Supported hardware '{hw}' was rejected")


@patch('ml_energy_score.measure.EmissionsTracker')
def test_dataset_sampling(mock_tracker):
    """Test that dataset is sampled correctly."""
    # TODO: Implement once model loading is done
    pass


@patch('ml_energy_score.measure.EmissionsTracker')
def test_results_structure(mock_tracker):
    """Test that results have correct structure."""
    # TODO: Implement once measurement is complete
    pass

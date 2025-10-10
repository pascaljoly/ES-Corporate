# tests/conftest.py
"""Shared test fixtures - automatically loaded by pytest"""
import pytest
from unittest.mock import Mock

@pytest.fixture
def sample_model_id():
    """Sample model identifier"""
    return "gpt2"

@pytest.fixture
def sample_task():
    """Sample task"""
    return "text_generation"

@pytest.fixture
def mock_emissions_data():
    """Mock CodeCarbon emissions output"""
    data = Mock()
    data.energy_consumed = 0.045  # kWh
    data.emissions = 0.021  # kg CO2
    data.gpu_energy = 0.038
    data.cpu_energy = 0.005
    data.ram_energy = 0.002
    return data

@pytest.fixture
def mock_hardware_info():
    """Mock hardware detection result"""
    return {
        'gpu': 'NVIDIA RTX 4090',
        'cpu': 'AMD Ryzen 9 7950X',
        'ram_gb': 64,
        'measurement_method': 'nvidia-smi + RAPL'
    }

@pytest.fixture
def sample_measurements():
    """Sample measurement data for testing aggregation"""
    return [
        {
            'run': 0,
            'energy_kwh': 0.045,
            'co2_kg': 0.021,
            'duration_seconds': 32
        },
        {
            'run': 1,
            'energy_kwh': 0.047,
            'co2_kg': 0.022,
            'duration_seconds': 33
        },
        {
            'run': 2,
            'energy_kwh': 0.044,
            'co2_kg': 0.020,
            'duration_seconds': 31
        }
    ]

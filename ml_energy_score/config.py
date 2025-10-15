"""Configuration for ML Energy Score."""

import sys
from pathlib import Path

# Add project root to path to import config_loader
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config_loader import get_config

# Initialize configuration
config = get_config()

# Legacy constants for backward compatibility
SUPPORTED_HARDWARE = config.get_supported_hardware()
SUPPORTED_TASKS = config.get_supported_tasks()

# CodeCarbon settings from configuration
codecarbon_config = config.get_codecarbon_config()
PUE = codecarbon_config.get('pue', 1.2)
MEASURE_POWER_SECS = codecarbon_config.get('measure_power_secs', 1)

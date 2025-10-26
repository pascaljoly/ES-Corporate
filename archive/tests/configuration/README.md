# Configuration System Tests

This directory contains test files for the configuration system.

## Test Files

### `example_config_usage.py`
A comprehensive demonstration script that shows how to use the configuration system. This script demonstrates:

- Basic configuration loading and access
- Using configuration-aware scorers
- Model comparison with configuration
- Configuration value overrides
- Adding new model profiles

**Usage:**
```bash
cd /path/to/EStool
source energy-compare/venv/bin/activate
python tests/configuration/example_config_usage.py
```

### `test_configuration_system.py`
A comprehensive test suite that validates all aspects of the configuration system. This script tests:

- Basic configuration loading
- Model profile management
- Configuration-aware scorer functionality
- Model comparison with configuration
- Environment variable overrides
- Configuration validation
- Computer vision model support

**Usage:**
```bash
cd /path/to/EStool
source energy-compare/venv/bin/activate
python tests/configuration/test_configuration_system.py
```

## Running Tests

Both test files are designed to be run from the project root directory. They will automatically:

1. Add the project root to the Python path
2. Import the necessary modules
3. Run their respective tests/demonstrations

## Expected Output

- **example_config_usage.py**: Shows detailed output of configuration system features
- **test_configuration_system.py**: Shows "ALL TESTS PASSED!" if everything works correctly

## Dependencies

These tests require:
- PyYAML (for configuration file parsing)
- The configuration system modules (`config_loader.py`, `config_aware_scorer.py`)
- The energy comparison modules (`comparator`, `scorer`)

All dependencies should be available if you've installed the project requirements.

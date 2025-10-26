# Archive Directory

This directory contains the older functionality that was developed before October 25, 2025.

## 📁 Contents

### `energy-compare/`
- **Previous energy comparison framework**
- Includes scorer, comparator, and configuration system
- Moved to archive as the new `energy-measurement/` system is more focused

### `ml_energy_score/`
- **Previous ML energy scoring system**
- Includes measurement functions and configuration
- Replaced by the simpler `energy-measurement/` approach

### `tests/`
- **Previous test suites**
- Configuration system tests
- Moved to archive as tests are now integrated with the main functionality

### Configuration Files
- `config_loader.py`: Configuration system loader
- `config.yaml`: Configuration file
- `CONFIGURATION_GUIDE.md`: Configuration documentation

### Demo Files
- `interactive_demo.py`: Interactive demonstration
- `end_to_end_demo.py`: End-to-end demo
- `quick_demo.py`: Quick demo
- `real_world_example.py`: Real-world usage examples
- `DEMO_README.md`: Demo documentation

### Results
- `demo_results/`: Previous demo results
- `sample_results/`: Sample measurement results

## 🔄 Migration Notes

The new `energy-measurement/` system provides:
- ✅ **Simpler API**: Single function for energy measurement
- ✅ **Better Testing**: Comprehensive test suite with realistic data
- ✅ **Star Ratings**: Built-in energy efficiency scoring
- ✅ **Cleaner Structure**: Focused on core functionality

## 📅 Archive Date

**Archived**: October 25, 2025  
**Reason**: Repository cleanup to focus on the new energy measurement functionality

## 🔍 Accessing Archived Code

If you need to access the archived functionality:

```bash
# Navigate to specific archived components
cd archive/energy-compare/
cd archive/ml_energy_score/
cd archive/tests/
```

The archived code is still functional but is no longer the primary focus of the project.

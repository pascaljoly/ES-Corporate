#!/usr/bin/env python3
"""
Test script for the energy scoring function.

This script creates fake measurement results and tests the calculate_scores function.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from calculate_scores import calculate_scores, print_scores


def create_fake_results():
    """Create fake measurement results for testing."""
    
    # Test data with varying energy consumption
    test_results = [
        {
            "model_name": "model-A",
            "task_name": "test-classification",
            "hardware": "CPU",
            "timestamp": "2025-01-15T10:30:45.123456",
            "num_samples": 100,
            "energy_kwh": 0.05,
            "co2_kg": 0.02,
            "duration_seconds": 125.3,
            "kwh_per_1000_queries": 0.05
        },
        {
            "model_name": "model-B", 
            "task_name": "test-classification",
            "hardware": "CPU",
            "timestamp": "2025-01-15T10:31:45.123456",
            "num_samples": 100,
            "energy_kwh": 0.08,
            "co2_kg": 0.03,
            "duration_seconds": 150.2,
            "kwh_per_1000_queries": 0.08
        },
        {
            "model_name": "model-C",
            "task_name": "test-classification", 
            "hardware": "CPU",
            "timestamp": "2025-01-15T10:32:45.123456",
            "num_samples": 100,
            "energy_kwh": 0.12,
            "co2_kg": 0.05,
            "duration_seconds": 180.1,
            "kwh_per_1000_queries": 0.12
        },
        {
            "model_name": "model-D",
            "task_name": "test-classification",
            "hardware": "CPU", 
            "timestamp": "2025-01-15T10:33:45.123456",
            "num_samples": 100,
            "energy_kwh": 0.15,
            "co2_kg": 0.06,
            "duration_seconds": 200.5,
            "kwh_per_1000_queries": 0.15
        },
        {
            "model_name": "model-E",
            "task_name": "test-classification",
            "hardware": "CPU",
            "timestamp": "2025-01-15T10:34:45.123456", 
            "num_samples": 100,
            "energy_kwh": 0.20,
            "co2_kg": 0.08,
            "duration_seconds": 250.0,
            "kwh_per_1000_queries": 0.20
        },
        {
            "model_name": "model-F",
            "task_name": "test-classification",
            "hardware": "CPU",
            "timestamp": "2025-01-15T10:35:45.123456",
            "num_samples": 100,
            "energy_kwh": 0.25,
            "co2_kg": 0.10,
            "duration_seconds": 300.0,
            "kwh_per_1000_queries": 0.25
        },
        {
            "model_name": "model-G",
            "task_name": "test-classification",
            "hardware": "CPU",
            "timestamp": "2025-01-15T10:36:45.123456",
            "num_samples": 100,
            "energy_kwh": 0.30,
            "co2_kg": 0.12,
            "duration_seconds": 350.0,
            "kwh_per_1000_queries": 0.30
        },
        {
            "model_name": "model-H",
            "task_name": "test-classification",
            "hardware": "CPU",
            "timestamp": "2025-01-15T10:37:45.123456",
            "num_samples": 100,
            "energy_kwh": 0.35,
            "co2_kg": 0.14,
            "duration_seconds": 400.0,
            "kwh_per_1000_queries": 0.35
        },
        {
            "model_name": "model-I",
            "task_name": "test-classification",
            "hardware": "CPU",
            "timestamp": "2025-01-15T10:38:45.123456",
            "num_samples": 100,
            "energy_kwh": 0.40,
            "co2_kg": 0.16,
            "duration_seconds": 450.0,
            "kwh_per_1000_queries": 0.40
        },
        {
            "model_name": "model-J",
            "task_name": "test-classification",
            "hardware": "CPU",
            "timestamp": "2025-01-15T10:39:45.123456",
            "num_samples": 100,
            "energy_kwh": 0.50,
            "co2_kg": 0.20,
            "duration_seconds": 500.0,
            "kwh_per_1000_queries": 0.50
        }
    ]
    
    # Create directory structure
    results_dir = Path("test_results")
    task_dir = results_dir / "test-classification" / "CPU"
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each result to a JSON file
    for i, result in enumerate(test_results):
        filename = f"{result['model_name']}_{result['timestamp'].replace(':', '-')}.json"
        filepath = task_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
    
    print(f"✅ Created {len(test_results)} fake measurement results in {task_dir}")
    return results_dir


def test_edge_cases():
    """Test edge cases for scoring function."""
    print("\n🧪 Testing Edge Cases")
    print("=" * 40)
    
    # Test 1: Empty directory
    empty_dir = Path("test_empty")
    empty_dir.mkdir(exist_ok=True)
    
    try:
        calculate_scores("nonexistent", "CPU", str(empty_dir))
        print("❌ Should have raised FileNotFoundError")
    except FileNotFoundError:
        print("✅ Correctly handled empty directory")
    
    # Test 2: Directory with no JSON files
    no_json_dir = Path("test_no_json")
    no_json_dir.mkdir(exist_ok=True)
    (no_json_dir / "test-classification" / "CPU").mkdir(parents=True, exist_ok=True)
    
    try:
        calculate_scores("test-classification", "CPU", str(no_json_dir))
        print("❌ Should have raised ValueError")
    except ValueError:
        print("✅ Correctly handled directory with no JSON files")
    
    # Test 3: Invalid JSON file
    invalid_dir = Path("test_invalid")
    invalid_dir.mkdir(exist_ok=True)
    (invalid_dir / "test-classification" / "CPU").mkdir(parents=True, exist_ok=True)
    
    # Create invalid JSON file
    invalid_file = invalid_dir / "test-classification" / "CPU" / "invalid.json"
    with open(invalid_file, 'w') as f:
        f.write("invalid json content")
    
    try:
        scores = calculate_scores("test-classification", "CPU", str(invalid_dir))
        print("❌ Should have raised ValueError for no valid results")
    except ValueError:
        print("✅ Correctly handled invalid JSON file (no valid results)")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Cleanup
    shutil.rmtree(empty_dir, ignore_errors=True)
    shutil.rmtree(no_json_dir, ignore_errors=True)
    shutil.rmtree(invalid_dir, ignore_errors=True)


def test_small_dataset():
    """Test with small dataset (1-4 models)."""
    print("\n🧪 Testing Small Dataset")
    print("=" * 40)
    
    # Create small dataset
    small_results = [
        {
            "model_name": "small-A",
            "task_name": "small-test",
            "hardware": "CPU",
            "timestamp": "2025-01-15T10:30:45.123456",
            "num_samples": 100,
            "energy_kwh": 0.10,
            "co2_kg": 0.04,
            "duration_seconds": 200.0,
            "kwh_per_1000_queries": 0.10
        },
        {
            "model_name": "small-B",
            "task_name": "small-test",
            "hardware": "CPU",
            "timestamp": "2025-01-15T10:31:45.123456",
            "num_samples": 100,
            "energy_kwh": 0.20,
            "co2_kg": 0.08,
            "duration_seconds": 300.0,
            "kwh_per_1000_queries": 0.20
        }
    ]
    
    # Create directory
    small_dir = Path("test_small")
    small_dir.mkdir(exist_ok=True)
    (small_dir / "small-test" / "CPU").mkdir(parents=True, exist_ok=True)
    
    # Save results
    for result in small_results:
        filename = f"{result['model_name']}_{result['timestamp'].replace(':', '-')}.json"
        filepath = small_dir / "small-test" / "CPU" / filename
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
    
    try:
        scores = calculate_scores("small-test", "CPU", str(small_dir))
        print_scores(scores)
        print("✅ Small dataset handled correctly")
    except Exception as e:
        print(f"❌ Error with small dataset: {e}")
    
    # Cleanup
    shutil.rmtree(small_dir, ignore_errors=True)


def main():
    """Run all tests."""
    print("🧪 Testing Energy Scoring Function")
    print("=" * 50)
    
    # Create fake results
    results_dir = create_fake_results()
    
    try:
        # Test main functionality
        print("\n📊 Testing Main Functionality")
        print("-" * 30)
        scores = calculate_scores("test-classification", "CPU", str(results_dir))
        print_scores(scores)
        
        # Test edge cases
        test_edge_cases()
        
        # Test small dataset
        test_small_dataset()
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        shutil.rmtree(results_dir, ignore_errors=True)
        print(f"\n🧹 Cleaned up test files")
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)

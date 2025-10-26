#!/usr/bin/env python3
"""
Test script for energy measurement functionality.
"""

import json
import time
import os
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from measure_energy import measure_energy


def dummy_inference(sample):
    """Dummy inference function for testing."""
    # Simulate some processing time
    time.sleep(0.001)
    return f"processed_{sample}"


def test_energy_measurement():
    """Test the energy measurement function."""
    print("=== Testing Energy Measurement ===")
    
    # Create dummy dataset
    dummy_dataset = [
        {"text": f"Sample {i}", "label": i % 3}
        for i in range(200)
    ]
    
    print(f"Created dataset with {len(dummy_dataset)} samples")
    
    # Test parameters
    model_name = "test_model"
    task_name = "text-classification"
    hardware = "CPU"
    num_samples = 50
    output_dir = "test_results"
    
    print(f"Testing with {num_samples} samples...")
    
    try:
        # Measure energy
        results = measure_energy(
            inference_fn=dummy_inference,
            dataset=dummy_dataset,
            model_name=model_name,
            task_name=task_name,
            hardware=hardware,
            num_samples=num_samples,
            output_dir=output_dir
        )
        
        print("\n✅ Test completed successfully!")
        print(f"Results: {json.dumps(results, indent=2)}")
        
        # Verify results structure
        required_keys = [
            "model_name", "task_name", "hardware", "timestamp",
            "num_samples", "energy_kwh", "co2_kg", "duration_seconds",
            "kwh_per_1000_queries"
        ]
        
        for key in required_keys:
            assert key in results, f"Missing key: {key}"
        
        print("✅ All required keys present in results")
        
        # Verify output file exists
        task_dir = Path(output_dir) / task_name
        json_files = list(task_dir.glob(f"{model_name}_*.json"))
        assert len(json_files) > 0, "No output JSON file found"
        
        print(f"✅ Output file created: {json_files[0]}")
        
        # Verify JSON content
        with open(json_files[0], 'r') as f:
            saved_results = json.load(f)
        
        assert saved_results == results, "Saved results don't match returned results"
        print("✅ JSON file content matches returned results")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_hardware_validation():
    """Test hardware validation."""
    print("\n=== Testing Hardware Validation ===")
    
    dummy_dataset = [{"text": "test"}]
    
    # Test valid hardware
    try:
        results = measure_energy(
            inference_fn=dummy_inference,
            dataset=dummy_dataset,
            model_name="test",
            task_name="test",
            hardware="CPU",
            num_samples=1
        )
        print("✅ Valid hardware (CPU) accepted")
    except ValueError as e:
        print(f"❌ Valid hardware rejected: {e}")
        return False
    
    # Test invalid hardware
    try:
        measure_energy(
            inference_fn=dummy_inference,
            dataset=dummy_dataset,
            model_name="test",
            task_name="test",
            hardware="INVALID_HARDWARE",
            num_samples=1
        )
        print("❌ Invalid hardware accepted (should have failed)")
        return False
    except ValueError:
        print("✅ Invalid hardware correctly rejected")
    
    return True


def test_empty_dataset():
    """Test handling of empty dataset."""
    print("\n=== Testing Empty Dataset Handling ===")
    
    try:
        measure_energy(
            inference_fn=dummy_inference,
            dataset=[],
            model_name="test",
            task_name="test",
            hardware="CPU",
            num_samples=1
        )
        print("❌ Empty dataset accepted (should have failed)")
        return False
    except ValueError:
        print("✅ Empty dataset correctly rejected")
        return True


def main():
    """Run all tests."""
    print("🧪 Running Energy Measurement Tests")
    print("=" * 50)
    
    tests = [
        test_energy_measurement,
        test_hardware_validation,
        test_empty_dataset
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return passed == total


if __name__ == "__main__":
    main()

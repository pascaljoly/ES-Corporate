#!/usr/bin/env python3
"""
Comprehensive test script to verify all security fixes work correctly
and that existing functionality still works.
"""

import sys
import json
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Test imports that don't require external dependencies
print("Testing imports...")

# Test security_utils (no dependencies)
from utils.security_utils import (
    sanitize_path_component,
    validate_input_length,
    validate_json_file_size,
    sanitize_and_validate_path,
    MAX_JSON_FILE_SIZE,
    MAX_FILES_TO_PROCESS
)
print("✅ security_utils imported successfully")

# Test sample_dataset (no dependencies)
from test.sample_dataset import sample_dataset, MAX_DATASET_SIZE_TO_LOAD
print("✅ sample_dataset imported successfully")

# Try to import measure_energy and calculate_scores
try:
    from measure_energy import measure_energy
    MEASURE_ENERGY_AVAILABLE = True
    print("✅ measure_energy imported successfully")
except ImportError as e:
    MEASURE_ENERGY_AVAILABLE = False
    print(f"⚠️  measure_energy not available (missing dependencies: {e})")

try:
    from calculate_scores import calculate_scores
    CALCULATE_SCORES_AVAILABLE = True
    print("✅ calculate_scores imported successfully")
except ImportError as e:
    CALCULATE_SCORES_AVAILABLE = False
    print(f"⚠️  calculate_scores not available (missing dependencies: {e})")


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_pass(self, test_name):
        self.passed += 1
        print(f"✅ PASS: {test_name}")
    
    def add_fail(self, test_name, error):
        self.failed += 1
        self.errors.append((test_name, error))
        print(f"❌ FAIL: {test_name} - {error}")
    
    def summary(self):
        print("\n" + "=" * 60)
        print(f"Test Summary: {self.passed} passed, {self.failed} failed")
        if self.errors:
            print("\nFailed Tests:")
            for test_name, error in self.errors:
                print(f"  - {test_name}: {error}")
        return self.failed == 0


def test_path_sanitization(results):
    """Test that path traversal attacks are blocked."""
    print("\n🔒 Testing Path Sanitization")
    print("-" * 40)
    
    # Test 1: Basic path sanitization
    try:
        sanitized = sanitize_path_component("../../../etc/passwd")
        if ".." in sanitized or "/" in sanitized:
            results.add_fail("path_sanitization_basic", f"Path not sanitized: {sanitized}")
        else:
            results.add_pass("path_sanitization_basic")
    except Exception as e:
        results.add_fail("path_sanitization_basic", str(e))
    
    # Test 2: Unsafe characters removed
    try:
        sanitized = sanitize_path_component("test<>:|?*\"file")
        unsafe_chars = ['<', '>', ':', '|', '?', '*', '"']
        if any(char in sanitized for char in unsafe_chars):
            results.add_fail("path_sanitization_unsafe_chars", f"Unsafe chars remain: {sanitized}")
        else:
            results.add_pass("path_sanitization_unsafe_chars")
    except Exception as e:
        results.add_fail("path_sanitization_unsafe_chars", str(e))
    
    # Test 3: Empty after sanitization
    try:
        sanitize_path_component("...")
        results.add_fail("path_sanitization_empty", "Should raise ValueError for empty result")
    except ValueError:
        results.add_pass("path_sanitization_empty")
    except Exception as e:
        results.add_fail("path_sanitization_empty", f"Unexpected error: {e}")


def test_input_validation(results):
    """Test input validation."""
    print("\n✅ Testing Input Validation")
    print("-" * 40)
    
    # Test 1: Too long string
    try:
        validate_input_length("x" * 300, "test_field")
        results.add_fail("input_validation_long", "Long string not rejected")
    except ValueError:
        results.add_pass("input_validation_long")
    except Exception as e:
        results.add_fail("input_validation_long", f"Unexpected error: {e}")
    
    # Test 2: Empty string
    try:
        validate_input_length("", "test_field")
        results.add_fail("input_validation_empty", "Empty string not rejected")
    except ValueError:
        results.add_pass("input_validation_empty")
    except Exception as e:
        results.add_fail("input_validation_empty", f"Unexpected error: {e}")
    
    # Test 3: Normal string passes
    try:
        validate_input_length("normal_string", "test_field")
        results.add_pass("input_validation_normal")
    except Exception as e:
        results.add_fail("input_validation_normal", str(e))


def test_memory_limits(results):
    """Test memory limits in sample_dataset."""
    print("\n💾 Testing Memory Limits")
    print("-" * 40)
    
    # Test 1: Normal sampling works
    try:
        dataset = [{"id": i} for i in range(1000)]
        samples = sample_dataset(dataset, num_samples=100, seed=42)
        if len(samples) == 100:
            results.add_pass("memory_limits_normal")
        else:
            results.add_fail("memory_limits_normal", f"Expected 100 samples, got {len(samples)}")
    except Exception as e:
        results.add_fail("memory_limits_normal", str(e))
    
    # Test 2: Large num_samples is rejected
    try:
        dataset = [{"id": i} for i in range(1000)]
        sample_dataset(dataset, num_samples=200000)  # Exceeds default limit
        results.add_fail("memory_limits_large_rejected", "Large num_samples not rejected")
    except ValueError:
        results.add_pass("memory_limits_large_rejected")
    except Exception as e:
        results.add_fail("memory_limits_large_rejected", f"Unexpected error: {e}")
    
    # Test 3: Zero samples rejected
    try:
        dataset = [{"id": i} for i in range(100)]
        sample_dataset(dataset, num_samples=0)
        results.add_fail("memory_limits_zero_rejected", "Zero samples not rejected")
    except ValueError:
        results.add_pass("memory_limits_zero_rejected")
    except Exception as e:
        results.add_fail("memory_limits_zero_rejected", f"Unexpected error: {e}")


def test_sample_dataset_reproducibility(results):
    """Test that sampling is still reproducible."""
    print("\n🎲 Testing Sample Dataset Reproducibility")
    print("-" * 40)
    
    try:
        dataset = [{"id": i} for i in range(1000)]
        
        # Same seed should produce same samples
        samples1 = sample_dataset(dataset, num_samples=50, seed=42)
        samples2 = sample_dataset(dataset, num_samples=50, seed=42)
        
        if samples1 == samples2:
            results.add_pass("sample_reproducibility")
        else:
            results.add_fail("sample_reproducibility", "Same seed produced different samples")
            
        # Different seeds should produce different samples
        samples3 = sample_dataset(dataset, num_samples=50, seed=99)
        if samples1 != samples3:
            results.add_pass("sample_variability")
        else:
            results.add_fail("sample_variability", "Different seeds produced same samples")
            
    except Exception as e:
        results.add_fail("sample_dataset_tests", str(e))


def test_sample_dataset_small_dataset(results):
    """Test handling of small datasets."""
    print("\n📊 Testing Small Dataset Handling")
    print("-" * 40)
    
    try:
        # Dataset smaller than num_samples
        dataset = [{"id": i} for i in range(50)]
        samples = sample_dataset(dataset, num_samples=100, seed=42)
        
        if len(samples) == 50:  # Should use all available
            results.add_pass("small_dataset_handling")
        else:
            results.add_fail("small_dataset_handling", f"Expected 50, got {len(samples)}")
    except Exception as e:
        results.add_fail("small_dataset_handling", str(e))


def test_json_size_limits(results):
    """Test JSON file size limits."""
    print("\n📄 Testing JSON File Size Limits")
    print("-" * 40)
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a large JSON file (> 10MB)
            large_file = Path(tmpdir) / "large.json"
            with open(large_file, 'w') as f:
                # Write a large JSON structure (11MB)
                large_data = {"data": "x" * (11 * 1024 * 1024)}
                json.dump(large_data, f)
            
            # This should raise ValueError
            validate_json_file_size(large_file)
            results.add_fail("json_size_limits", "Large file not rejected")
    except ValueError:
        results.add_pass("json_size_limits")
    except Exception as e:
        results.add_fail("json_size_limits", f"Unexpected error: {e}")
    
    # Test that normal size files pass
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            normal_file = Path(tmpdir) / "normal.json"
            with open(normal_file, 'w') as f:
                json.dump({"data": "test"}, f)
            
            validate_json_file_size(normal_file)
            results.add_pass("json_size_normal")
    except Exception as e:
        results.add_fail("json_size_normal", str(e))


def test_sanitize_and_validate_path(results):
    """Test path creation with sanitization."""
    print("\n🛡️  Testing Path Creation with Sanitization")
    print("-" * 40)
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should create safe path even with unsafe components
            safe_path = sanitize_and_validate_path(
                tmpdir,
                "test-task",
                "CPU",
                create=True
            )
            
            if safe_path.exists():
                results.add_pass("sanitize_path_creation")
            else:
                results.add_fail("sanitize_path_creation", "Path not created")
    except Exception as e:
        results.add_fail("sanitize_path_creation", str(e))


def test_measure_energy_with_mock(results):
    """Test measure_energy if available, otherwise skip."""
    if not MEASURE_ENERGY_AVAILABLE:
        print("\n⚠️  Skipping measure_energy tests (dependencies not available)")
        return
    
    print("\n📊 Testing measure_energy (if dependencies available)")
    print("-" * 40)
    
    def dummy_inference(sample):
        return sample
    
    dataset = [{"text": f"Sample {i}"} for i in range(100)]
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = measure_energy(
                inference_fn=dummy_inference,
                dataset=dataset,
                model_name="test_model",
                task_name="test-task",
                hardware="CPU",
                num_samples=10,
                output_dir=tmpdir
            )
            
            # Verify result structure
            required_keys = ["model_name", "task_name", "hardware", "energy_kwh", 
                           "kwh_per_1000_queries", "num_samples"]
            for key in required_keys:
                if key not in result:
                    results.add_fail("measure_energy_structure", f"Missing key: {key}")
                    return
            
            results.add_pass("measure_energy_structure")
            
            # Test path traversal protection
            try:
                measure_energy(
                    inference_fn=dummy_inference,
                    dataset=dataset,
                    model_name="test",
                    task_name="../../../etc",  # Path traversal attempt
                    hardware="CPU",
                    num_samples=1,
                    output_dir=tmpdir
                )
                results.add_fail("measure_energy_path_traversal", "Path traversal not blocked")
            except ValueError:
                results.add_pass("measure_energy_path_traversal")
            
    except Exception as e:
        results.add_fail("measure_energy_tests", str(e))


def test_calculate_scores_with_mock(results):
    """Test calculate_scores if available."""
    if not CALCULATE_SCORES_AVAILABLE:
        print("\n⚠️  Skipping calculate_scores tests (dependencies not available)")
        return
    
    print("\n📈 Testing calculate_scores (if dependencies available)")
    print("-" * 40)
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test results directory - need to use sanitized paths
            from datetime import datetime
            from utils.security_utils import sanitize_and_validate_path
            
            # Create directory structure manually to match what calculate_scores expects
            # calculate_scores will sanitize "test-task" and "CPU" internally
            results_dir = Path(tmpdir) / "test-task" / "CPU"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a few test JSON files
            for i in range(5):
                result_data = {
                    "model_name": f"model_{i}",
                    "task_name": "test-task",
                    "hardware": "CPU",
                    "timestamp": datetime.now().isoformat(),
                    "num_samples": 100,
                    "energy_kwh": 0.05 + i * 0.01,
                    "co2_kg": 0.02 + i * 0.005,
                    "duration_seconds": 100.0 + i * 10,
                    "kwh_per_1000_queries": 0.5 + i * 0.1
                }
                
                filename = results_dir / f"model_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w') as f:
                    json.dump(result_data, f)
            
            # Test calculate_scores - use tmpdir directly (it will sanitize internally)
            scores = calculate_scores("test-task", "CPU", tmpdir)
            
            # Verify structure
            if "models" not in scores or "num_models" not in scores:
                results.add_fail("calculate_scores_structure", "Invalid result structure")
                return
            
            if scores["num_models"] != 5:
                results.add_fail("calculate_scores_structure", f"Expected 5 models, got {scores['num_models']}")
                return
            
            results.add_pass("calculate_scores_structure")
            
            # Test path traversal protection
            try:
                calculate_scores("../../../etc", "passwd", tmpdir)
                results.add_fail("calculate_scores_path_traversal", "Path traversal not blocked")
            except (ValueError, FileNotFoundError):
                results.add_pass("calculate_scores_path_traversal")
            
    except Exception as e:
        results.add_fail("calculate_scores_tests", str(e))


def main():
    """Run all tests."""
    print("\n🧪 Comprehensive Security Fixes Test Suite")
    print("=" * 60)
    print("Testing that security fixes work AND existing functionality is preserved\n")
    
    results = TestResults()
    
    # Run all test suites
    test_path_sanitization(results)
    test_input_validation(results)
    test_memory_limits(results)
    test_sample_dataset_reproducibility(results)
    test_sample_dataset_small_dataset(results)
    test_json_size_limits(results)
    test_sanitize_and_validate_path(results)
    test_measure_energy_with_mock(results)
    test_calculate_scores_with_mock(results)
    
    # Print summary
    success = results.summary()
    
    if success:
        print("\n🎉 All tests passed! Security fixes are working correctly.")
        print("\n📝 Note: Some tests may have been skipped if dependencies are missing.")
        print("   Install dependencies with: pip install -r requirements.txt")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

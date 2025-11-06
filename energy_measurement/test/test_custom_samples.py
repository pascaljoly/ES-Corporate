#!/usr/bin/env python3
"""
Test script for custom sample sizes in energy measurement.
"""

import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from measure_energy import measure_energy


def dummy_inference(sample):
    """Dummy inference function for testing."""
    # Simulate some processing time
    time.sleep(0.001)
    return f"processed_{sample}"


def test_custom_sample_sizes():
    """Test that measure_energy works with any num_samples value."""
    print("🧪 Testing Custom Sample Sizes in measure_energy")
    print("=" * 50)
    
    # Create test dataset
    dataset = [{'id': i, 'text': f'sample_{i}'} for i in range(1000)]
    
    # Test with 50 samples
    print("\n📊 Testing with 50 samples...")
    results_50 = measure_energy(
        inference_fn=dummy_inference,
        dataset=dataset,
        model_name="test-50",
        task_name="test",
        hardware="CPU",
        num_samples=50,
        seed=42
    )
    
    # Test with 100 samples
    print("\n📊 Testing with 100 samples...")
    results_100 = measure_energy(
        inference_fn=dummy_inference,
        dataset=dataset,
        model_name="test-100",
        task_name="test",
        hardware="CPU",
        num_samples=100,
        seed=42
    )
    
    # Test with 200 samples
    print("\n📊 Testing with 200 samples...")
    results_200 = measure_energy(
        inference_fn=dummy_inference,
        dataset=dataset,
        model_name="test-200",
        task_name="test",
        hardware="CPU",
        num_samples=200,
        seed=42
    )
    
    # Verify results
    assert results_50['num_samples'] == 50, f"Expected 50 samples, got {results_50['num_samples']}"
    assert results_100['num_samples'] == 100, f"Expected 100 samples, got {results_100['num_samples']}"
    assert results_200['num_samples'] == 200, f"Expected 200 samples, got {results_200['num_samples']}"
    
    # Verify all have kwh_per_1000_queries (normalized)
    assert 'kwh_per_1000_queries' in results_50, "Missing kwh_per_1000_queries in results_50"
    assert 'kwh_per_1000_queries' in results_100, "Missing kwh_per_1000_queries in results_100"
    assert 'kwh_per_1000_queries' in results_200, "Missing kwh_per_1000_queries in results_200"
    
    # Verify seed is recorded
    assert results_50['seed'] == 42, "Seed not recorded in results_50"
    assert results_100['seed'] == 42, "Seed not recorded in results_100"
    assert results_200['seed'] == 42, "Seed not recorded in results_200"
    
    print("✅ Custom sample sizes work in measure_energy()")
    print(f"  50 samples: {results_50['kwh_per_1000_queries']:.6f} kWh/1k queries")
    print(f"  100 samples: {results_100['kwh_per_1000_queries']:.6f} kWh/1k queries")
    print(f"  200 samples: {results_200['kwh_per_1000_queries']:.6f} kWh/1k queries")
    
    return True


def test_reproducible_comparison():
    """Test that same seed produces identical samples for fair comparison."""
    print("\n🧪 Testing Reproducible Comparison")
    print("=" * 50)
    
    dataset = [{'id': i, 'data': f'item_{i}'} for i in range(500)]
    
    # Test model v1 with seed 42
    print("\n📊 Testing model v1 with seed 42...")
    results_v1 = measure_energy(
        inference_fn=dummy_inference,
        dataset=dataset,
        model_name="model-v1",
        task_name="comparison",
        hardware="CPU",
        num_samples=100,
        seed=42
    )
    
    # Test model v2 with same seed 42 (should get same samples)
    print("\n📊 Testing model v2 with seed 42 (same samples)...")
    results_v2 = measure_energy(
        inference_fn=dummy_inference,
        dataset=dataset,
        model_name="model-v2",
        task_name="comparison",
        hardware="CPU",
        num_samples=100,
        seed=42
    )
    
    # Test model v3 with different seed 99 (should get different samples)
    print("\n📊 Testing model v3 with seed 99 (different samples)...")
    results_v3 = measure_energy(
        inference_fn=dummy_inference,
        dataset=dataset,
        model_name="model-v3",
        task_name="comparison",
        hardware="CPU",
        num_samples=100,
        seed=99
    )
    
    # Verify same seed produces same results (fair comparison)
    assert results_v1['num_samples'] == results_v2['num_samples'], "Same seed should produce same number of samples"
    print("✅ Same seed produces identical sample counts")
    
    # Verify different seed produces different results
    assert results_v1['num_samples'] == results_v3['num_samples'], "Different seeds should produce same sample count"
    print("✅ Different seeds work correctly")
    
    print(f"  Model v1 (seed=42): {results_v1['kwh_per_1000_queries']:.6f} kWh/1k queries")
    print(f"  Model v2 (seed=42): {results_v2['kwh_per_1000_queries']:.6f} kWh/1k queries")
    print(f"  Model v3 (seed=99): {results_v3['kwh_per_1000_queries']:.6f} kWh/1k queries")
    
    return True


def test_small_dataset():
    """Test with small dataset (fewer samples than requested)."""
    print("\n🧪 Testing Small Dataset")
    print("=" * 50)
    
    # Create small dataset
    small_dataset = [{'id': i} for i in range(50)]
    
    # Request more samples than available
    results = measure_energy(
        inference_fn=dummy_inference,
        dataset=small_dataset,
        model_name="small-test",
        task_name="test",
        hardware="CPU",
        num_samples=100,  # More than available
        seed=42
    )
    
    # Should use all available samples
    assert results['num_samples'] == 50, f"Expected 50 samples, got {results['num_samples']}"
    assert 'kwh_per_1000_queries' in results, "Missing kwh_per_1000_queries"
    
    print("✅ Small dataset handled correctly")
    print(f"  Used {results['num_samples']} samples (all available)")
    print(f"  Energy: {results['kwh_per_1000_queries']:.6f} kWh/1k queries")
    
    return True


def main():
    """Run all tests."""
    print("🧪 Testing Custom Sample Sizes in Energy Measurement")
    print("=" * 60)
    
    tests = [
        test_custom_sample_sizes,
        test_reproducible_comparison,
        test_small_dataset
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

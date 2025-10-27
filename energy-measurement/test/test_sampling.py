#!/usr/bin/env python3
"""
Comprehensive test suite for the dataset sampling function.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from .sample_dataset import sample_dataset, sample_dataset_with_replacement


def test_reproducibility():
    """Test that same seed produces identical samples."""
    print("🧪 Testing Reproducibility")
    print("-" * 30)
    
    # Create test dataset
    test_data = [{'id': i, 'value': f'item_{i}'} for i in range(5000)]
    
    # Sample twice with same seed
    samples_1 = sample_dataset(test_data, num_samples=100, seed=42)
    samples_2 = sample_dataset(test_data, num_samples=100, seed=42)
    
    # Verify identical
    assert samples_1 == samples_2, "Same seed should produce identical samples"
    print("✓ Reproducibility test passed: Same seed = same samples")
    
    # Verify order matters
    ids_1 = [s['id'] for s in samples_1]
    ids_2 = [s['id'] for s in samples_2]
    assert ids_1 == ids_2, "Sample order should be identical"
    print("✓ Order test passed: Sample order is reproducible")
    
    return True


def test_variability():
    """Test that different seeds produce different samples."""
    print("\n🧪 Testing Variability")
    print("-" * 30)
    
    test_data = [{'id': i, 'value': f'item_{i}'} for i in range(5000)]
    
    # Sample with different seeds
    samples_a = sample_dataset(test_data, num_samples=100, seed=42)
    samples_b = sample_dataset(test_data, num_samples=100, seed=99)
    
    # Verify different
    assert samples_a != samples_b, "Different seeds should produce different samples"
    print("✓ Variability test passed: Different seeds = different samples")
    
    # Check overlap (should be some but not 100%)
    ids_a = set(s['id'] for s in samples_a)
    ids_b = set(s['id'] for s in samples_b)
    overlap = len(ids_a & ids_b)
    print(f"  Overlap: {overlap}/100 samples (expected ~2-5 with large dataset)")
    assert overlap < 50, "Different seeds should have low overlap"
    
    return True


def test_small_dataset():
    """Test handling of small datasets."""
    print("\n🧪 Testing Small Dataset Handling")
    print("-" * 30)
    
    # Dataset smaller than requested samples
    small_data = [{'id': i} for i in range(50)]
    
    samples = sample_dataset(small_data, num_samples=100, seed=42)
    
    assert len(samples) == 50, "Should return all samples when dataset is small"
    print("✓ Small dataset test passed: Returns all samples when dataset < num_samples")
    
    return True


def test_custom_sample_sizes():
    """Test various sample sizes to ensure flexibility."""
    print("\n🧪 Testing Custom Sample Sizes")
    print("-" * 30)
    
    test_data = [{'id': i, 'value': f'item_{i}'} for i in range(5000)]
    
    # Test various sample sizes
    test_cases = [
        (10, "Very small"),
        (50, "Small"),
        (100, "Small standard"),
        (500, "Medium"),
        (1000, "Standard benchmark"),
        (2000, "Large"),
        (5000, "Maximum available")
    ]
    
    for size, description in test_cases:
        samples = sample_dataset(test_data, num_samples=size, seed=42)
        expected = min(size, len(test_data))
        assert len(samples) == expected, f"Should return {expected} samples for {description}"
        print(f"✓ Custom size test passed: {description} (num_samples={size}) → {len(samples)} samples")
    
    print("✓ All custom sample sizes work correctly - num_samples is fully flexible")
    return True


def test_model_comparison():
    """Test model comparison simulation."""
    print("\n🧪 Testing Model Comparison Simulation")
    print("-" * 30)
    
    # Simulate comparing two model versions
    dataset = [{'image': f'img_{i}.jpg'} for i in range(1000)]
    
    # Test with 100 samples (not default 1000)
    samples_v1 = sample_dataset(dataset, num_samples=100, seed=42)
    energy_v1 = sum(s['image'].count('0') for s in samples_v1)  # Dummy metric
    
    # Model v2 test (same seed, same custom size!)
    samples_v2 = sample_dataset(dataset, num_samples=100, seed=42)
    energy_v2 = sum(s['image'].count('0') for s in samples_v2)  # Dummy metric
    
    # Verify tested on same samples
    assert samples_v1 == samples_v2, "Both model versions tested on identical samples"
    print("✓ Model comparison test passed: Same seed ensures fair comparison (custom num_samples=100)")
    
    return True


def test_no_duplicates():
    """Test that sampling produces no duplicates."""
    print("\n🧪 Testing No Duplicates")
    print("-" * 30)
    
    test_data = [{'id': i, 'value': f'item_{i}'} for i in range(1000)]
    samples = sample_dataset(test_data, num_samples=500, seed=42)
    ids = [s['id'] for s in samples]
    
    assert len(ids) == len(set(ids)), "Should have no duplicate samples"
    print("✓ No duplicates test passed: Each sample appears only once")
    
    return True


def test_error_handling():
    """Test error handling for edge cases."""
    print("\n🧪 Testing Error Handling")
    print("-" * 30)
    
    test_data = [{'id': i} for i in range(100)]
    
    # Empty dataset
    try:
        sample_dataset([], num_samples=100)
        assert False, "Should raise error on empty dataset"
    except ValueError as e:
        print(f"✓ Empty dataset error: {e}")
    
    # Invalid num_samples
    try:
        sample_dataset(test_data, num_samples=0)
        assert False, "Should raise error on num_samples=0"
    except ValueError as e:
        print(f"✓ Zero samples error: {e}")
    
    # Negative num_samples
    try:
        sample_dataset(test_data, num_samples=-10)
        assert False, "Should raise error on negative num_samples"
    except ValueError as e:
        print(f"✓ Negative samples error: {e}")
    
    # Non-iterable dataset
    try:
        sample_dataset(123, num_samples=100)
        assert False, "Should raise error on non-iterable dataset"
    except TypeError as e:
        print(f"✓ Non-iterable error: {e}")
    
    print("✓ All error handling works correctly")
    return True


def test_with_replacement():
    """Test sampling with replacement."""
    print("\n🧪 Testing Sampling With Replacement")
    print("-" * 30)
    
    test_data = [{'id': i} for i in range(10)]  # Small dataset
    
    # Sample more than available with replacement
    samples = sample_dataset_with_replacement(test_data, num_samples=50, seed=42)
    
    assert len(samples) == 50, "Should return exactly 50 samples"
    
    # Check that we have duplicates (since we sampled 50 from 10)
    ids = [s['id'] for s in samples]
    unique_ids = set(ids)
    assert len(unique_ids) <= 10, "Should have at most 10 unique IDs"
    assert len(ids) > len(unique_ids), "Should have duplicates when sampling with replacement"
    
    print("✓ Sampling with replacement works correctly")
    return True


def test_huggingface_dataset_simulation():
    """Test with HuggingFace-style dataset simulation."""
    print("\n🧪 Testing HuggingFace Dataset Simulation")
    print("-" * 30)
    
    # Simulate HuggingFace dataset (iterable with __iter__)
    class MockHFDataset:
        def __init__(self, data):
            self.data = data
        
        def __iter__(self):
            return iter(self.data)
    
    hf_data = MockHFDataset([{'text': f'sample_{i}'} for i in range(1000)])
    
    samples = sample_dataset(hf_data, num_samples=100, seed=42)
    
    assert len(samples) == 100, "Should work with HuggingFace-style datasets"
    assert all('text' in s for s in samples), "Should preserve data structure"
    
    print("✓ HuggingFace dataset simulation works correctly")
    return True


def main():
    """Run all tests."""
    print("🧪 Testing Dataset Sampling Function")
    print("=" * 50)
    
    tests = [
        test_reproducibility,
        test_variability,
        test_small_dataset,
        test_custom_sample_sizes,
        test_model_comparison,
        test_no_duplicates,
        test_error_handling,
        test_with_replacement,
        test_huggingface_dataset_simulation
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
    
    print("\n" + "=" * 50)
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

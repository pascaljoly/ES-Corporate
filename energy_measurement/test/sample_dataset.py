#!/usr/bin/env python3
"""
Dataset sampling function for reproducible, fair model comparisons.

This module provides functionality to sample datasets with reproducible randomness,
ensuring that when comparing model versions (e.g., before/after fine-tuning,
fp32 vs int8 quantization), all models are tested on identical samples.
"""

import random
import itertools
from typing import Any, Iterable, List

# Memory safety limits
MAX_DATASET_SIZE_TO_LOAD = 100000  # Maximum items to load into memory
DEFAULT_MAX_LOAD_MULTIPLIER = 10  # Multiply num_samples by this for max load
MIN_LOAD_SIZE = 1000  # Minimum items to load even if num_samples is small


def sample_dataset(
    dataset: Iterable[Any],
    num_samples: int = 1000,
    seed: int = 42,
    max_dataset_size: int = MAX_DATASET_SIZE_TO_LOAD
) -> List[Any]:
    """
    Sample dataset with reproducible randomness for fair model comparisons.
    
    This ensures that when comparing model versions (e.g., before/after fine-tuning,
    fp32 vs int8 quantization), all models are tested on identical samples.
    
    The num_samples parameter is fully flexible - adjust based on your dataset size
    and testing needs. Small datasets may use 100 samples, standard benchmarking
    uses 1000, but any value is valid.
    
    Args:
        dataset: Any iterable (list, HuggingFace dataset, generator, etc.)
        num_samples: Number of samples to extract (default 1000, but fully customizable)
        seed: Random seed for reproducibility (default 42)
        max_dataset_size: Maximum number of items to load into memory (default: 100,000)
                          This prevents memory exhaustion with very large datasets
        
    Returns:
        List of sampled items from dataset
        
    Raises:
        ValueError: If dataset is empty, num_samples is not positive, or exceeds limits
        TypeError: If dataset is not iterable
        MemoryError: If dataset is too large to safely load into memory
        
    Examples:
        >>> # Small dataset - use fewer samples
        >>> samples = sample_dataset(small_dataset, num_samples=100, seed=42)
        >>> 
        >>> # Standard benchmarking
        >>> samples = sample_dataset(large_dataset, num_samples=1000, seed=42)
        >>> 
        >>> # Large-scale testing
        >>> samples = sample_dataset(huge_dataset, num_samples=5000, seed=42)
        >>> 
        >>> # Compare model versions with SAME samples
        >>> samples = sample_dataset(my_dataset, num_samples=500, seed=42)
        >>> results_v1 = measure_energy(model_v1, samples, ...)
        >>> results_v2 = measure_energy(model_v2, samples, ...)
    """
    
    # Validate inputs
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    
    if num_samples > max_dataset_size:
        raise ValueError(
            f"num_samples ({num_samples}) exceeds maximum dataset size limit "
            f"({max_dataset_size}). This prevents memory exhaustion. "
            "Consider using a smaller sample size or streaming sampling."
        )
    
    # Check if dataset is iterable
    try:
        iter(dataset)
    except TypeError:
        raise TypeError(f"Dataset must be iterable, got {type(dataset).__name__}")
    
    # Convert dataset to list with memory limits (handle generators, iterators, etc.)
    try:
        # Calculate safe limit: load enough to get good samples, but cap at max_dataset_size
        # Load more than num_samples to ensure we have enough variety
        max_samples_to_load = min(
            max(num_samples * DEFAULT_MAX_LOAD_MULTIPLIER, MIN_LOAD_SIZE),
            max_dataset_size
        )
        
        dataset_list = list(itertools.islice(dataset, max_samples_to_load))
    except MemoryError:
        raise MemoryError(
            f"Insufficient memory to load dataset (tried to load up to {max_samples_to_load} items). "
            f"Try reducing num_samples (current: {num_samples}) or max_dataset_size (current: {max_dataset_size})."
        )
    except Exception as e:
        raise TypeError(f"Could not iterate over dataset: {e}")
    
    # Handle empty dataset
    if not dataset_list:
        raise ValueError("Dataset is empty")
    
    # Handle small datasets
    if len(dataset_list) <= num_samples:
        print(f"Warning: Dataset has only {len(dataset_list)} samples, using all of them (requested {num_samples})")
        return dataset_list
    
    # Use local Random instance instead of global random state for thread safety
    rng = random.Random(seed)
    
    # Sample without replacement (no duplicates)
    sampled = rng.sample(dataset_list, num_samples)
    
    # Print summary
    print(f"Sampled {num_samples} from {len(dataset_list)} samples (seed={seed})")
    
    return sampled


def sample_dataset_with_replacement(
    dataset: Iterable[Any],
    num_samples: int = 1000,
    seed: int = 42,
    max_dataset_size: int = MAX_DATASET_SIZE_TO_LOAD
) -> List[Any]:
    """
    Sample dataset with replacement (allows duplicates) for specific use cases.
    
    This is useful when you need exactly num_samples but your dataset is smaller,
    or when you want to test with repeated samples.
    
    Args:
        dataset: Any iterable
        num_samples: Number of samples to extract
        seed: Random seed for reproducibility
        max_dataset_size: Maximum number of items to load into memory (default: 100,000)
        
    Returns:
        List of sampled items (may contain duplicates)
        
    Raises:
        ValueError: If dataset is empty, num_samples is not positive, or exceeds limits
        TypeError: If dataset is not iterable
        MemoryError: If dataset is too large to safely load into memory
    """
    
    # Validate inputs
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    
    if num_samples > max_dataset_size:
        raise ValueError(
            f"num_samples ({num_samples}) exceeds maximum dataset size limit "
            f"({max_dataset_size}). This prevents memory exhaustion."
        )
    
    # Convert dataset to list with memory limits
    try:
        dataset_list = list(itertools.islice(dataset, max_dataset_size))
    except MemoryError:
        raise MemoryError(
            f"Insufficient memory to load dataset (tried to load up to {max_dataset_size} items). "
            f"Try reducing num_samples (current: {num_samples}) or max_dataset_size (current: {max_dataset_size})."
        )
    except Exception as e:
        raise TypeError(f"Could not iterate over dataset: {e}")
    
    if not dataset_list:
        raise ValueError("Dataset is empty")
    
    # Use local Random instance instead of global random state for thread safety
    rng = random.Random(seed)
    
    # Sample with replacement
    sampled = [rng.choice(dataset_list) for _ in range(num_samples)]
    
    print(f"Sampled {num_samples} with replacement from {len(dataset_list)} samples (seed={seed})")
    
    return sampled


if __name__ == "__main__":
    # Example usage
    test_data = [{'id': i, 'value': f'item_{i}'} for i in range(1000)]
    
    print("Testing sample_dataset function:")
    
    # Test 1: Basic sampling
    samples = sample_dataset(test_data, num_samples=100, seed=42)
    print(f"Sampled {len(samples)} items")
    
    # Test 2: Reproducibility
    samples_1 = sample_dataset(test_data, num_samples=50, seed=42)
    samples_2 = sample_dataset(test_data, num_samples=50, seed=42)
    assert samples_1 == samples_2, "Same seed should produce identical samples"
    print("✓ Reproducibility test passed")
    
    # Test 3: Different seeds
    samples_a = sample_dataset(test_data, num_samples=50, seed=42)
    samples_b = sample_dataset(test_data, num_samples=50, seed=99)
    assert samples_a != samples_b, "Different seeds should produce different samples"
    print("✓ Variability test passed")
    
    print("All tests passed!")

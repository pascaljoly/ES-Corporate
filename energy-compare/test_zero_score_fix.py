#!/usr/bin/env python3
"""
Test to verify that no model gets a score of 0.0 after the fix.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from comparator import ModelComparator, ComparisonMetric
from scorer.core import ScoringResult


def test_no_zero_scores():
    """Test that no model gets a score of 0.0"""
    print("=== Testing Zero Score Prevention ===")
    
    # Create test cases that could potentially result in zero scores
    test_cases = [
        {
            "name": "Identical values",
            "results": [
                ScoringResult(
                    model_id="model1",
                    task="text_generation",
                    measurements={
                        'energy_per_1k_wh': 5.0,
                        'co2_per_1k_g': 2.5,
                        'samples_per_second': 100,
                        'duration_seconds': 10.0
                    },
                    hardware={'gpu': 'Test GPU'},
                    metadata={'timestamp': '2025-10-03'}
                ),
                ScoringResult(
                    model_id="model2",
                    task="text_generation",
                    measurements={
                        'energy_per_1k_wh': 5.0,  # Identical values
                        'co2_per_1k_g': 2.5,
                        'samples_per_second': 100,
                        'duration_seconds': 10.0
                    },
                    hardware={'gpu': 'Test GPU'},
                    metadata={'timestamp': '2025-10-03'}
                )
            ]
        },
        {
            "name": "Zero values",
            "results": [
                ScoringResult(
                    model_id="model1",
                    task="text_generation",
                    measurements={
                        'energy_per_1k_wh': 0.0,  # Zero values
                        'co2_per_1k_g': 0.0,
                        'samples_per_second': 0,
                        'duration_seconds': 0
                    },
                    hardware={'gpu': 'Test GPU'},
                    metadata={'timestamp': '2025-10-03'}
                ),
                ScoringResult(
                    model_id="model2",
                    task="text_generation",
                    measurements={
                        'energy_per_1k_wh': 0.0,
                        'co2_per_1k_g': 0.0,
                        'samples_per_second': 0,
                        'duration_seconds': 0
                    },
                    hardware={'gpu': 'Test GPU'},
                    metadata={'timestamp': '2025-10-03'}
                )
            ]
        },
        {
            "name": "Extreme values",
            "results": [
                ScoringResult(
                    model_id="model1",
                    task="text_generation",
                    measurements={
                        'energy_per_1k_wh': 1000.0,  # Very high energy
                        'co2_per_1k_g': 500.0,
                        'samples_per_second': 1,     # Very low performance
                        'duration_seconds': 1000.0
                    },
                    hardware={'gpu': 'Test GPU'},
                    metadata={'timestamp': '2025-10-03'}
                ),
                ScoringResult(
                    model_id="model2",
                    task="text_generation",
                    measurements={
                        'energy_per_1k_wh': 0.1,    # Very low energy
                        'co2_per_1k_g': 0.05,
                        'samples_per_second': 1000, # Very high performance
                        'duration_seconds': 0.1
                    },
                    hardware={'gpu': 'Test GPU'},
                    metadata={'timestamp': '2025-10-03'}
                )
            ]
        },
        {
            "name": "Single model",
            "results": [
                ScoringResult(
                    model_id="single_model",
                    task="text_generation",
                    measurements={
                        'energy_per_1k_wh': 3.0,
                        'co2_per_1k_g': 1.5,
                        'samples_per_second': 80,
                        'duration_seconds': 12.5
                    },
                    hardware={'gpu': 'Test GPU'},
                    metadata={'timestamp': '2025-10-03'}
                )
            ]
        }
    ]
    
    comparator = ModelComparator()
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        
        result = comparator.compare_models_from_results(test_case['results'])
        
        # Check that no model has a score of 0.0
        zero_scores = [model for model in result.models if model.score == 0.0]
        if zero_scores:
            print(f"❌ FAILED: Found {len(zero_scores)} models with score 0.0")
            for model in zero_scores:
                print(f"   - {model.model_id}: {model.score}")
        else:
            print(f"✅ PASSED: No models with score 0.0")
        
        # Print all scores for verification
        print("   Scores:")
        for model in result.get_rankings():
            print(f"   - {model.model_id}: {model.score:.3f}")
        
        # Verify all scores are >= 0.1
        min_score = min(model.score for model in result.models)
        if min_score >= 0.1:
            print(f"   ✅ Minimum score: {min_score:.3f} (>= 0.1)")
        else:
            print(f"   ❌ Minimum score: {min_score:.3f} (< 0.1)")


def test_different_weight_combinations():
    """Test various weight combinations to ensure no zero scores"""
    print("\n=== Testing Different Weight Combinations ===")
    
    # Create test results
    results = [
        ScoringResult(
            model_id="model1",
            task="text_generation",
            measurements={
                'energy_per_1k_wh': 2.0,
                'co2_per_1k_g': 1.0,
                'samples_per_second': 100,
                'duration_seconds': 10.0
            },
            hardware={'gpu': 'Test GPU'},
            metadata={'timestamp': '2025-10-03'}
        ),
        ScoringResult(
            model_id="model2",
            task="text_generation",
            measurements={
                'energy_per_1k_wh': 4.0,
                'co2_per_1k_g': 2.0,
                'samples_per_second': 80,
                'duration_seconds': 12.5
            },
            hardware={'gpu': 'Test GPU'},
            metadata={'timestamp': '2025-10-03'}
        )
    ]
    
    comparator = ModelComparator()
    
    # Test different weight combinations
    weight_combinations = [
        {
            "name": "Energy only",
            "weights": {ComparisonMetric.ENERGY_EFFICIENCY: 1.0}
        },
        {
            "name": "CO2 only", 
            "weights": {ComparisonMetric.CO2_EFFICIENCY: 1.0}
        },
        {
            "name": "Performance only",
            "weights": {ComparisonMetric.PERFORMANCE: 1.0}
        },
        {
            "name": "Speed only",
            "weights": {ComparisonMetric.SPEED: 1.0}
        },
        {
            "name": "Cost-effectiveness only",
            "weights": {ComparisonMetric.COST_EFFECTIVENESS: 1.0}
        },
        {
            "name": "Balanced",
            "weights": {
                ComparisonMetric.ENERGY_EFFICIENCY: 0.3,
                ComparisonMetric.CO2_EFFICIENCY: 0.2,
                ComparisonMetric.PERFORMANCE: 0.3,
                ComparisonMetric.SPEED: 0.2
            }
        }
    ]
    
    for combo in weight_combinations:
        print(f"\nTesting: {combo['name']}")
        
        result = comparator.compare_models_from_results(
            results,
            custom_weights=combo['weights']
        )
        
        # Check for zero scores
        zero_scores = [model for model in result.models if model.score == 0.0]
        if zero_scores:
            print(f"   ❌ FAILED: Found models with score 0.0")
        else:
            print(f"   ✅ PASSED: No zero scores")
        
        # Print scores
        for model in result.get_rankings():
            print(f"   - {model.model_id}: {model.score:.3f}")


if __name__ == "__main__":
    print("Zero Score Prevention Testing")
    print("=" * 40)
    
    try:
        test_no_zero_scores()
        test_different_weight_combinations()
        
        print("\n" + "=" * 40)
        print("✅ All zero score prevention tests completed!")
        print("No model should ever get a score of 0.0")
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        import traceback
        traceback.print_exc()

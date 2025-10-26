#!/usr/bin/env python3
"""
Comprehensive test of the configuration system.

This script tests all aspects of the configuration system to ensure
everything works correctly together.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Add energy-compare to path
sys.path.append(str(project_root / "energy-compare"))

from config_loader import get_config, ConfigError
from scorer.config_aware_scorer import ConfigAwareModelScorer
from comparator import ModelComparator, ComparisonMetric


def test_basic_configuration():
    """Test basic configuration loading and access"""
    print("=== Testing Basic Configuration ===")
    
    config = get_config()
    
    # Test basic access
    assert config.get_supported_hardware() is not None
    assert config.get_supported_tasks() is not None
    assert config.get_metric_weights() is not None
    
    # Test dot notation access
    min_stars = config.get("scoring.star_rating.min_stars")
    max_stars = config.get("scoring.star_rating.max_stars")
    assert min_stars < max_stars
    
    # Test validation
    config.validate_config()
    
    print("✅ Basic configuration test passed")


def test_model_profiles():
    """Test model profile loading"""
    print("\n=== Testing Model Profiles ===")
    
    config = get_config()
    
    # Test getting model profiles
    distilgpt2_profile = config.get_model_profile("distilgpt2", "text_generation")
    assert "energy_per_1k_wh" in distilgpt2_profile
    assert "co2_per_1k_g" in distilgpt2_profile
    assert "samples_per_second" in distilgpt2_profile
    
    # Test default profile for unknown model
    unknown_profile = config.get_model_profile("unknown_model", "text_generation")
    assert "energy_per_1k_wh" in unknown_profile
    
    print("✅ Model profiles test passed")


def test_config_aware_scorer():
    """Test configuration-aware scorer"""
    print("\n=== Testing Configuration-Aware Scorer ===")
    
    scorer = ConfigAwareModelScorer()
    
    # Test scoring with known model
    result = scorer.score(
        model="distilgpt2",
        task="text-generation",
        n_samples=10,
        runs=1
    )
    
    assert result.model_id == "distilgpt2"
    assert result.task == "text-generation"
    assert "energy_per_1k_wh" in result.measurements
    assert "co2_per_1k_g" in result.measurements
    assert "samples_per_second" in result.measurements
    
    # Test available models
    available_models = scorer.get_available_models("text_generation")
    assert "distilgpt2" in available_models
    
    # Test profile validation
    valid_profile = {
        "energy_per_1k_wh": 1.0,
        "co2_per_1k_g": 0.5,
        "samples_per_second": 100,
        "duration_seconds": 10.0,
        "size_mb": 50,
        "architecture": "Test"
    }
    is_valid, error = scorer.validate_model_profile(valid_profile)
    assert is_valid
    assert error is None
    
    # Test invalid profile
    invalid_profile = {"energy_per_1k_wh": 1.0}  # Missing required fields
    is_valid, error = scorer.validate_model_profile(invalid_profile)
    assert not is_valid
    assert error is not None
    
    print("✅ Configuration-aware scorer test passed")


def test_model_comparison():
    """Test model comparison with configuration"""
    print("\n=== Testing Model Comparison ===")
    
    scorer = ConfigAwareModelScorer()
    comparator = ModelComparator(scorer=scorer)
    
    # Test comparison
    model_specs = [
        ("distilgpt2", "text-generation"),
        ("gpt2", "text-generation")
    ]
    
    result = comparator.compare_models(
        model_specs=model_specs,
        n_samples=10,
        runs=1
    )
    
    assert len(result.models) == 2
    assert result.task == "text-generation"
    assert "winner" in result.summary
    assert "winner_stars" in result.summary
    
    # Test rankings
    rankings = result.get_rankings()
    assert len(rankings) == 2
    assert rankings[0].rank == 1
    assert rankings[1].rank == 2
    
    print("✅ Model comparison test passed")


def test_environment_variables():
    """Test environment variable overrides"""
    print("\n=== Testing Environment Variables ===")
    
    # Set environment variables
    os.environ["ENERGY_SCORE_SCORING_METRIC_WEIGHTS_ENERGY_EFFICIENCY"] = "0.7"
    os.environ["ENERGY_SCORE_SCORING_METRIC_WEIGHTS_CO2_EFFICIENCY"] = "0.3"
    os.environ["ENERGY_SCORE_SCORING_METRIC_WEIGHTS_PERFORMANCE"] = "0.0"
    os.environ["ENERGY_SCORE_SCORING_METRIC_WEIGHTS_SPEED"] = "0.0"
    
    # Create new config instance to test env vars
    from config_loader import ConfigLoader
    config = ConfigLoader()
    config.update_from_env()
    
    weights = config.get_metric_weights()
    assert weights["energy_efficiency"] == 0.7
    assert weights["co2_efficiency"] == 0.3
    assert weights["performance"] == 0.0
    assert weights["speed"] == 0.0
    
    # Test validation
    config.validate_config()
    
    # Clean up environment variables
    del os.environ["ENERGY_SCORE_SCORING_METRIC_WEIGHTS_ENERGY_EFFICIENCY"]
    del os.environ["ENERGY_SCORE_SCORING_METRIC_WEIGHTS_CO2_EFFICIENCY"]
    del os.environ["ENERGY_SCORE_SCORING_METRIC_WEIGHTS_PERFORMANCE"]
    del os.environ["ENERGY_SCORE_SCORING_METRIC_WEIGHTS_SPEED"]
    
    print("✅ Environment variables test passed")


def test_configuration_validation():
    """Test configuration validation"""
    print("\n=== Testing Configuration Validation ===")
    
    config = get_config()
    
    # Test valid configuration
    config.validate_config()
    
    # Test invalid weights (should raise error)
    config._set_nested_value("scoring.metric_weights.energy_efficiency", 0.8)
    config._set_nested_value("scoring.metric_weights.co2_efficiency", 0.5)  # Total > 1.0
    
    try:
        config.validate_config()
        assert False, "Should have raised ConfigError"
    except ConfigError:
        pass  # Expected
    
    # Restore valid weights
    config._set_nested_value("scoring.metric_weights.energy_efficiency", 0.4)
    config._set_nested_value("scoring.metric_weights.co2_efficiency", 0.3)
    config.validate_config()
    
    print("✅ Configuration validation test passed")


def test_computer_vision_models():
    """Test computer vision model profiles"""
    print("\n=== Testing Computer Vision Models ===")
    
    config = get_config()
    
    # Test computer vision model profiles
    birefnet_profile = config.get_model_profile("BiRefNet", "computer_vision")
    assert "energy_per_1k_wh" in birefnet_profile
    assert "architecture" in birefnet_profile
    
    yolov8n_profile = config.get_model_profile("YOLOv8n", "computer_vision")
    assert "energy_per_1k_wh" in yolov8n_profile
    
    # Test that computer vision models are in the config
    cv_models = config.get("model_profiles.computer_vision", {})
    assert "BiRefNet" in cv_models
    assert "YOLOv8n" in cv_models
    assert "ResNet-50" in cv_models
    
    print("✅ Computer vision models test passed")


def main():
    """Run all tests"""
    print("Configuration System Comprehensive Test")
    print("=" * 50)
    
    try:
        test_basic_configuration()
        test_model_profiles()
        test_config_aware_scorer()
        test_model_comparison()
        test_environment_variables()
        test_configuration_validation()
        test_computer_vision_models()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("The configuration system is working correctly.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

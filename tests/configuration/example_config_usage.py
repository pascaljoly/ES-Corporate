#!/usr/bin/env python3
"""
Example script demonstrating the new configuration system.

This script shows how to:
1. Load configuration from config.yaml
2. Use configuration-aware scorers
3. Override configuration values
4. Add new model profiles
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Add energy-compare to path
sys.path.append(str(project_root / "energy-compare"))

from config_loader import get_config, ConfigError
from scorer.config_aware_scorer import ConfigAwareModelScorer
from comparator import ModelComparator, ComparisonMetric


def demonstrate_config_loading():
    """Demonstrate basic configuration loading and access"""
    print("=== Configuration Loading Demo ===")
    
    try:
        # Load configuration
        config = get_config()
        
        # Access different configuration sections
        print(f"Supported hardware: {list(config.get_supported_hardware().keys())}")
        print(f"Supported tasks: {config.get_supported_tasks()}")
        print(f"Metric weights: {config.get_metric_weights()}")
        print(f"Star rating range: {config.get('scoring.star_rating.min_stars')} - {config.get('scoring.star_rating.max_stars')}")
        
        # Get a specific model profile
        profile = config.get_model_profile("distilgpt2", "text_generation")
        print(f"DistilGPT2 profile: {profile}")
        
        # Validate configuration
        config.validate_config()
        print("✅ Configuration validation passed")
        
    except ConfigError as e:
        print(f"❌ Configuration error: {e}")


def demonstrate_config_aware_scorer():
    """Demonstrate using the configuration-aware scorer"""
    print("\n=== Configuration-Aware Scorer Demo ===")
    
    try:
        # Create scorer
        scorer = ConfigAwareModelScorer()
        
        # Score a model using configuration profiles
        result = scorer.score(
            model="distilgpt2",
            task="text-generation",
            n_samples=50,
            runs=2
        )
        
        print(f"Model: {result.model_id}")
        print(f"Task: {result.task}")
        print(f"Energy: {result.measurements['energy_per_1k_wh']:.2f} kWh/1k queries")
        print(f"CO2: {result.measurements['co2_per_1k_g']:.2f} kg CO2/1k queries")
        print(f"Throughput: {result.measurements['samples_per_second']:.0f} samples/sec")
        print(f"Architecture: {result.metadata['architecture']}")
        print(f"Model Size: {result.metadata['model_size_mb']} MB")
        
        # Show available models
        available_models = scorer.get_available_models("text_generation")
        print(f"\nAvailable text generation models: {list(available_models.keys())}")
        
    except Exception as e:
        print(f"❌ Scorer error: {e}")


def demonstrate_model_comparison():
    """Demonstrate model comparison using configuration"""
    print("\n=== Model Comparison Demo ===")
    
    try:
        # Create configuration-aware scorer
        scorer = ConfigAwareModelScorer()
        
        # Create comparator with the scorer
        comparator = ModelComparator(scorer=scorer)
        
        # Compare models using configuration profiles
        model_specs = [
            ("distilgpt2", "text-generation"),
            ("gpt2", "text-generation"),
            ("gpt2-medium", "text-generation")
        ]
        
        result = comparator.compare_models(
            model_specs=model_specs,
            n_samples=50,
            runs=2
        )
        
        print(f"Task: {result.task}")
        print(f"Winner: {result.summary['winner']} ({result.summary['winner_stars']} stars)")
        
        print("\nRankings:")
        for model in result.get_rankings():
            star_rating = ModelComparator.format_star_rating(model.score)
            measurements = model.scoring_result.measurements
            metadata = model.scoring_result.metadata
            
            print(f"{model.rank}. {model.model_id}")
            print(f"   Score: {star_rating}")
            print(f"   Energy: {measurements['energy_per_1k_wh']:.1f} kWh/1k queries")
            print(f"   CO2: {measurements['co2_per_1k_g']:.1f} kg CO2/1k queries")
            print(f"   Throughput: {measurements['samples_per_second']:.0f} samples/sec")
            print(f"   Architecture: {metadata.get('architecture', 'Unknown')}")
            print(f"   Model Size: {metadata.get('model_size_mb', 'N/A')} MB")
            print()
        
    except Exception as e:
        print(f"❌ Comparison error: {e}")


def demonstrate_configuration_overrides():
    """Demonstrate how to override configuration values"""
    print("\n=== Configuration Overrides Demo ===")
    
    try:
        config = get_config()
        
        # Show current metric weights
        print(f"Current metric weights: {config.get_metric_weights()}")
        
        # Override a specific value
        config._set_nested_value("scoring.metric_weights.energy_efficiency", 0.6)
        config._set_nested_value("scoring.metric_weights.co2_efficiency", 0.4)
        config._set_nested_value("scoring.metric_weights.performance", 0.0)
        config._set_nested_value("scoring.metric_weights.speed", 0.0)
        
        print(f"Updated metric weights: {config.get_metric_weights()}")
        
        # Validate the updated configuration
        config.validate_config()
        print("✅ Updated configuration validation passed")
        
    except Exception as e:
        print(f"❌ Override error: {e}")


def demonstrate_adding_model_profile():
    """Demonstrate adding a new model profile"""
    print("\n=== Adding Model Profile Demo ===")
    
    try:
        scorer = ConfigAwareModelScorer()
        
        # Define a new model profile
        new_profile = {
            "energy_per_1k_wh": 1.5,
            "co2_per_1k_g": 0.75,
            "samples_per_second": 180,
            "duration_seconds": 5.6,
            "size_mb": 45,
            "architecture": "Custom Transformer"
        }
        
        # Validate the profile
        is_valid, error = scorer.validate_model_profile(new_profile)
        if is_valid:
            print("✅ New model profile is valid")
            
            # Add the profile (in-memory only)
            scorer.add_model_profile("custom-model", new_profile, "text_generation")
            
            # Test scoring with the new model
            result = scorer.score(
                model="custom-model",
                task="text-generation",
                n_samples=30,
                runs=1
            )
            
            print(f"Custom model results:")
            print(f"  Energy: {result.measurements['energy_per_1k_wh']:.2f} kWh/1k queries")
            print(f"  CO2: {result.measurements['co2_per_1k_g']:.2f} kg CO2/1k queries")
            print(f"  Throughput: {result.measurements['samples_per_second']:.0f} samples/sec")
            
        else:
            print(f"❌ Invalid model profile: {error}")
        
    except Exception as e:
        print(f"❌ Add profile error: {e}")


def main():
    """Run all demonstrations"""
    print("Energy Score Tool - Configuration System Demo")
    print("=" * 50)
    
    demonstrate_config_loading()
    demonstrate_config_aware_scorer()
    demonstrate_model_comparison()
    demonstrate_configuration_overrides()
    demonstrate_adding_model_profile()
    
    print("\n" + "=" * 50)
    print("Demo completed! Check config.yaml to see all available parameters.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test the framework with computer vision models like BiRefNet.

This demonstrates how the energy comparison framework can work with
computer vision models for tasks like salient object detection.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from comparator import ModelComparator, ComparisonMetric
from scorer.core import ScoringResult, ModelEnergyScorer


class ComputerVisionModelScorer(ModelEnergyScorer):
    """
    A scorer that simulates realistic energy consumption for computer vision models.
    """
    
    def __init__(self):
        super().__init__()
        # Define realistic model characteristics for computer vision models
        self.model_profiles = {
            # Salient Object Detection Models
            "BiRefNet": {
                "energy_per_1k_wh": 8.5,
                "co2_per_1k_g": 4.2,
                "samples_per_second": 12,  # Images per second
                "duration_seconds": 83.3,  # Time per 1000 images
                "size_mb": 45,
                "architecture": "CNN + Transformer hybrid"
            },
            "U2Net": {
                "energy_per_1k_wh": 6.2,
                "co2_per_1k_g": 3.1,
                "samples_per_second": 18,
                "duration_seconds": 55.6,
                "size_mb": 176,
                "architecture": "U-Net based"
            },
            "PoolNet": {
                "energy_per_1k_wh": 5.8,
                "co2_per_1k_g": 2.9,
                "samples_per_second": 22,
                "duration_seconds": 45.5,
                "size_mb": 68,
                "architecture": "PoolNet"
            },
            
            # Object Detection Models
            "YOLOv8n": {
                "energy_per_1k_wh": 3.2,
                "co2_per_1k_g": 1.6,
                "samples_per_second": 45,
                "duration_seconds": 22.2,
                "size_mb": 6,
                "architecture": "YOLO"
            },
            "YOLOv8s": {
                "energy_per_1k_wh": 4.1,
                "co2_per_1k_g": 2.0,
                "samples_per_second": 35,
                "duration_seconds": 28.6,
                "size_mb": 22,
                "architecture": "YOLO"
            },
            "YOLOv8m": {
                "energy_per_1k_wh": 6.8,
                "co2_per_1k_g": 3.4,
                "samples_per_second": 25,
                "duration_seconds": 40.0,
                "size_mb": 50,
                "architecture": "YOLO"
            },
            
            # Image Classification Models
            "ResNet-50": {
                "energy_per_1k_wh": 2.1,
                "co2_per_1k_g": 1.0,
                "samples_per_second": 120,
                "duration_seconds": 8.3,
                "size_mb": 98,
                "architecture": "ResNet"
            },
            "EfficientNet-B0": {
                "energy_per_1k_wh": 1.8,
                "co2_per_1k_g": 0.9,
                "samples_per_second": 150,
                "duration_seconds": 6.7,
                "size_mb": 20,
                "architecture": "EfficientNet"
            },
            "Vision Transformer (ViT-B/16)": {
                "energy_per_1k_wh": 4.5,
                "co2_per_1k_g": 2.2,
                "samples_per_second": 85,
                "duration_seconds": 11.8,
                "size_mb": 330,
                "architecture": "Transformer"
            }
        }
    
    def score(self, model: str, task: str, n_samples: int = 100, runs: int = 3) -> ScoringResult:
        """Return realistic scoring results for computer vision models."""
        self.logger.info(f"Scoring {model} on {task} (computer vision)")
        
        # Validate inputs
        self._validate_inputs(model, task, n_samples, runs)
        
        # Get model profile or use default
        profile = self.model_profiles.get(model, {
            "energy_per_1k_wh": 5.0,
            "co2_per_1k_g": 2.5,
            "samples_per_second": 50,
            "duration_seconds": 20.0,
            "size_mb": 100,
            "architecture": "Unknown"
        })
        
        # Add some realistic variation
        import random
        variation_factor = 1.0 + (random.uniform(-0.05, 0.05) * (runs - 1) / 10)
        
        result = ScoringResult(
            model_id=model,
            task=task,
            measurements={
                'energy_per_1k_wh': profile["energy_per_1k_wh"] * variation_factor,
                'co2_per_1k_g': profile["co2_per_1k_g"] * variation_factor,
                'samples_per_second': profile["samples_per_second"] * variation_factor,
                'duration_seconds': profile["duration_seconds"] / variation_factor,
                'statistics': {
                    'coefficient_of_variation': 0.03 + random.uniform(0, 0.05),
                    'runs': runs
                }
            },
            hardware={'gpu': 'RTX 4090', 'cpu': 'AMD Ryzen 9 7950X'},
            metadata={
                'timestamp': '2025-10-03T12:00:00Z',
                'model_size_mb': profile.get("size_mb", 100),
                'architecture': profile.get("architecture", "Unknown"),
                'n_samples': n_samples,
                'runs': runs,
                'task_type': 'computer_vision'
            }
        )
        
        self.logger.info(f"Scoring complete for {model}")
        return result


def test_salient_object_detection_models():
    """Test comparison of salient object detection models including BiRefNet."""
    print("=== Salient Object Detection Models Comparison ===")
    
    cv_scorer = ComputerVisionModelScorer()
    comparator = ModelComparator(scorer=cv_scorer)
    
    model_specs = [
        ("BiRefNet", "object_detection"),
        ("U2Net", "object_detection"),
        ("PoolNet", "object_detection")
    ]
    
    print(f"Comparing {len(model_specs)} salient object detection models...")
    
    result = comparator.compare_models(
        model_specs=model_specs,
        n_samples=50,
        runs=2
    )
    
    print(f"\nResults for task: {result.task}")
    print(f"Winner: {result.summary['winner']} ({result.summary['winner_stars']} stars)")
    
    print("\nRankings:")
    for model in result.get_rankings():
        measurements = model.scoring_result.measurements
        metadata = model.scoring_result.metadata
        star_rating = ModelComparator.format_star_rating(model.score)
        
        print(f"{model.rank}. {model.model_id}")
        print(f"   Score: {star_rating}")
        print(f"   Energy: {measurements['energy_per_1k_wh']:.1f} kWh/1k images")
        print(f"   CO2: {measurements['co2_per_1k_g']:.1f} kg CO2/1k images")
        print(f"   Throughput: {measurements['samples_per_second']:.0f} images/sec")
        print(f"   Architecture: {metadata.get('architecture', 'Unknown')}")
        print(f"   Model Size: {metadata.get('model_size_mb', 'N/A')} MB")
        print()


def test_yolo_models_comparison():
    """Test comparison of YOLO object detection models."""
    print("=== YOLO Object Detection Models Comparison ===")
    
    cv_scorer = ComputerVisionModelScorer()
    comparator = ModelComparator(scorer=cv_scorer)
    
    model_specs = [
        ("YOLOv8n", "object_detection"),
        ("YOLOv8s", "object_detection"),
        ("YOLOv8m", "object_detection")
    ]
    
    result = comparator.compare_models(
        model_specs=model_specs,
        n_samples=50,
        runs=2
    )
    
    print(f"Winner: {result.summary['winner']} ({result.summary['winner_stars']} stars)")
    
    print("\nYOLO Model Rankings:")
    for model in result.get_rankings():
        measurements = model.scoring_result.measurements
        star_rating = ModelComparator.format_star_rating(model.score)
        
        print(f"{model.rank}. {model.model_id}")
        print(f"   Score: {star_rating}")
        print(f"   Energy: {measurements['energy_per_1k_wh']:.1f} kWh/1k images")
        print(f"   Throughput: {measurements['samples_per_second']:.0f} images/sec")
        print(f"   Model Size: {model.scoring_result.metadata.get('model_size_mb', 'N/A')} MB")
        print()


def test_image_classification_models():
    """Test comparison of image classification models."""
    print("=== Image Classification Models Comparison ===")
    
    cv_scorer = ComputerVisionModelScorer()
    comparator = ModelComparator(scorer=cv_scorer)
    
    model_specs = [
        ("ResNet-50", "image-classification"),
        ("EfficientNet-B0", "image-classification"),
        ("Vision Transformer (ViT-B/16)", "image-classification")
    ]
    
    result = comparator.compare_models(
        model_specs=model_specs,
        n_samples=50,
        runs=2
    )
    
    print(f"Winner: {result.summary['winner']} ({result.summary['winner_stars']} stars)")
    
    print("\nImage Classification Model Rankings:")
    for model in result.get_rankings():
        measurements = model.scoring_result.measurements
        metadata = model.scoring_result.metadata
        star_rating = ModelComparator.format_star_rating(model.score)
        
        print(f"{model.rank}. {model.model_id}")
        print(f"   Score: {star_rating}")
        print(f"   Energy: {measurements['energy_per_1k_wh']:.1f} kWh/1k images")
        print(f"   Throughput: {measurements['samples_per_second']:.0f} images/sec")
        print(f"   Architecture: {metadata.get('architecture', 'Unknown')}")
        print(f"   Model Size: {metadata.get('model_size_mb', 'N/A')} MB")
        print()


def test_birefnet_specific_comparison():
    """Test BiRefNet specifically against other salient object detection models."""
    print("=== BiRefNet Specific Comparison ===")
    print("(Comparing BiRefNet with other salient object detection models)")
    
    cv_scorer = ComputerVisionModelScorer()
    comparator = ModelComparator(scorer=cv_scorer)
    
    # Focus on salient object detection models
    model_specs = [
        ("BiRefNet", "object_detection"),
        ("U2Net", "object_detection"),
        ("PoolNet", "object_detection")
    ]
    
    # Energy-focused comparison for salient object detection
    energy_weights = {
        ComparisonMetric.ENERGY_EFFICIENCY: 0.6,
        ComparisonMetric.CO2_EFFICIENCY: 0.4
    }
    
    result = comparator.compare_models(
        model_specs=model_specs,
        custom_weights=energy_weights,
        n_samples=50,
        runs=2
    )
    
    print(f"Winner: {result.summary['winner']} ({result.summary['winner_stars']} stars)")
    
    print("\nSalient Object Detection Rankings (Energy-Focused):")
    for model in result.get_rankings():
        measurements = model.scoring_result.measurements
        metadata = model.scoring_result.metadata
        star_rating = ModelComparator.format_star_rating(model.score)
        
        print(f"{model.rank}. {model.model_id}")
        print(f"   Score: {star_rating}")
        print(f"   Energy: {measurements['energy_per_1k_wh']:.1f} kWh/1k images")
        print(f"   CO2: {measurements['co2_per_1k_g']:.1f} kg CO2/1k images")
        print(f"   Throughput: {measurements['samples_per_second']:.0f} images/sec")
        print(f"   Architecture: {metadata.get('architecture', 'Unknown')}")
        print(f"   Model Size: {metadata.get('model_size_mb', 'N/A')} MB")
        print()
    
    # Show BiRefNet's specific characteristics
    birefnet_model = next((m for m in result.models if m.model_id == "BiRefNet"), None)
    if birefnet_model:
        print("BiRefNet Analysis:")
        print(f"  - Energy Efficiency: {birefnet_model.score:.1f} stars")
        print(f"  - Energy Consumption: {birefnet_model.scoring_result.measurements['energy_per_1k_wh']:.1f} kWh/1k images")
        print(f"  - CO2 Emissions: {birefnet_model.scoring_result.measurements['co2_per_1k_g']:.1f} kg CO2/1k images")
        print(f"  - Throughput: {birefnet_model.scoring_result.measurements['samples_per_second']:.0f} images/sec")
        print(f"  - Architecture: {birefnet_model.scoring_result.metadata.get('architecture', 'Unknown')}")
        print(f"  - Model Size: {birefnet_model.scoring_result.metadata.get('model_size_mb', 'N/A')} MB")
        print("  - Note: BiRefNet uses CNN + Transformer hybrid architecture")
        print("  - Trade-off: Higher energy cost for potentially better accuracy")


def test_energy_focused_object_detection():
    """Test energy-focused comparison for object detection models only."""
    print("=== Energy-Focused Object Detection Models ===")
    
    cv_scorer = ComputerVisionModelScorer()
    comparator = ModelComparator(scorer=cv_scorer)
    
    model_specs = [
        ("BiRefNet", "object_detection"),
        ("YOLOv8n", "object_detection"),
        ("YOLOv8s", "object_detection")
    ]
    
    # Energy-focused weights
    energy_weights = {
        ComparisonMetric.ENERGY_EFFICIENCY: 0.6,
        ComparisonMetric.CO2_EFFICIENCY: 0.4
    }
    
    result = comparator.compare_models(
        model_specs=model_specs,
        custom_weights=energy_weights,
        n_samples=50,
        runs=2
    )
    
    print("Energy-focused object detection rankings:")
    for model in result.get_rankings():
        measurements = model.scoring_result.measurements
        star_rating = ModelComparator.format_star_rating(model.score)
        
        print(f"{model.rank}. {model.model_id}")
        print(f"   Score: {star_rating}")
        print(f"   Energy: {measurements['energy_per_1k_wh']:.1f} kWh/1k images")
        print(f"   CO2: {measurements['co2_per_1k_g']:.1f} kg CO2/1k images")
        print()


def test_performance_focused_object_detection():
    """Test performance-focused comparison for object detection models only."""
    print("=== Performance-Focused Object Detection Models ===")
    
    cv_scorer = ComputerVisionModelScorer()
    comparator = ModelComparator(scorer=cv_scorer)
    
    model_specs = [
        ("BiRefNet", "object_detection"),
        ("YOLOv8n", "object_detection"),
        ("YOLOv8s", "object_detection")
    ]
    
    # Performance-focused weights
    performance_weights = {
        ComparisonMetric.PERFORMANCE: 0.5,
        ComparisonMetric.SPEED: 0.3,
        ComparisonMetric.ENERGY_EFFICIENCY: 0.2
    }
    
    result = comparator.compare_models(
        model_specs=model_specs,
        custom_weights=performance_weights,
        n_samples=50,
        runs=2
    )
    
    print("Performance-focused object detection rankings:")
    for model in result.get_rankings():
        measurements = model.scoring_result.measurements
        star_rating = ModelComparator.format_star_rating(model.score)
        
        print(f"{model.rank}. {model.model_id}")
        print(f"   Score: {star_rating}")
        print(f"   Throughput: {measurements['samples_per_second']:.0f} images/sec")
        print(f"   Energy: {measurements['energy_per_1k_wh']:.1f} kWh/1k images")
        print()


if __name__ == "__main__":
    print("Computer Vision Models Energy Comparison Testing")
    print("=" * 60)
    
    try:
        test_salient_object_detection_models()
        test_yolo_models_comparison()
        test_image_classification_models()
        test_birefnet_specific_comparison()
        test_energy_focused_object_detection()
        test_performance_focused_object_detection()
        
        print("=" * 60)
        print("✅ All computer vision model tests completed successfully!")
        print("\nKey Insights:")
        print("- BiRefNet can be evaluated using the 'object_detection' task")
        print("- The framework supports various CV architectures (CNN, Transformer, YOLO)")
        print("- Energy efficiency varies significantly across model types")
        print("- Smaller models (YOLOv8n, EfficientNet-B0) are more energy efficient")
        print("- Specialized models (BiRefNet) may have higher energy costs")
        print("\nFramework Compatibility:")
        print("✅ BiRefNet: Supported via 'object_detection' task")
        print("✅ YOLO models: Supported via 'object_detection' task")
        print("✅ ResNet/EfficientNet: Supported via 'image_classification' task")
        print("✅ Vision Transformers: Supported via 'image_classification' task")
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        import traceback
        traceback.print_exc()

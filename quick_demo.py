#!/usr/bin/env python3
"""
Quick 30-Second Demo for Management

Shows the key value proposition in under 30 seconds:
- Model scoring
- Model comparison
- Business benefits
"""

import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "energy-compare"))

from scorer.config_aware_scorer import ConfigAwareModelScorer
from comparator import ModelComparator


def quick_demo():
    """30-second demo showing key capabilities"""
    print("🚀 Energy Score Tool - Quick Demo")
    print("=" * 40)
    
    # Create scorer and comparator
    scorer = ConfigAwareModelScorer()
    comparator = ModelComparator(scorer=scorer)
    
    # Quick model comparison
    models = [
        ("distilgpt2", "text-generation"),
        ("gpt2", "text-generation"),
        ("gpt2-medium", "text-generation")
    ]
    
    print("\n📊 Comparing 3 GPT-2 Models...")
    result = comparator.compare_models(models, n_samples=20, runs=1)
    
    print(f"\n🏆 Winner: {result.summary['winner']} ({result.summary['winner_stars']} stars)")
    
    print("\n📋 Rankings:")
    for model in result.get_rankings():
        measurements = model.scoring_result.measurements
        print(f"  {model.rank}. {model.model_id} - {model.score:.1f} stars")
        print(f"     Energy: {measurements['energy_per_1k_wh']:.1f} kWh/1k | "
              f"CO2: {measurements['co2_per_1k_g']:.1f} kg | "
              f"Speed: {measurements['samples_per_second']:.0f} samples/sec")
    
    print("\n💡 Business Value:")
    print("  • Cost Optimization: Choose energy-efficient models")
    print("  • Sustainability: Track CO2 emissions")
    print("  • Easy Configuration: No code changes needed")
    print("  • Production Ready: Complete end-to-end solution")
    
    print("\n✅ Ready for production use!")


if __name__ == "__main__":
    quick_demo()

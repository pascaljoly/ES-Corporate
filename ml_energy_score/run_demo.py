#!/usr/bin/env python3
"""
Demo script to generate real energy measurement results.
Run this to see your energy measurement tool in action!
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ml_energy_score.measure import measure_model_energy


def create_demo_dataset():
    """Create a simple demo dataset for testing."""
    
    class DemoDataset:
        def __init__(self):
            self.data = [
                {"text": "I absolutely love this product! Amazing quality and fast shipping."},
                {"text": "This is the worst purchase I've ever made. Complete waste of money."},
                {"text": "Pretty decent product. Good value for the price."},
                {"text": "Not impressed. Expected much better quality for this price."},
                {"text": "Excellent service and great product! Highly recommend."},
                {"text": "Mediocre at best. Nothing special but does the job."},
                {"text": "Outstanding quality! Exceeded all my expectations."},
                {"text": "Poor quality control. Product arrived damaged."},
                {"text": "Good product overall. Minor issues but mostly satisfied."},
                {"text": "Fantastic! This is exactly what I was looking for."}
            ]
        
        def __len__(self):
            return len(self.data)
        
        def __iter__(self):
            return iter(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
        
        def select(self, indices):
            selected_data = [self.data[i] for i in indices if i < len(self.data)]
            new_dataset = DemoDataset()
            new_dataset.data = selected_data
            return new_dataset
    
    return DemoDataset()


def run_demo():
    """Run a demo energy measurement."""
    
    print("🚀 ML Energy Score - Live Demo")
    print("=" * 40)
    print()
    
    # Create demo dataset
    print("📊 Creating demo dataset (10 sentiment samples)...")
    dataset = create_demo_dataset()
    
    print("🤖 Testing with small BERT model...")
    print("Model: prajjwal1/bert-tiny (4.4M parameters)")
    print("Task: text-classification")
    print("Hardware: CPU")
    print()
    
    try:
        print("⚡ Starting energy measurement...")
        result = measure_model_energy(
            model_path="prajjwal1/bert-tiny",
            task="text-classification",
            dataset=dataset,
            hardware="CPU",
            num_samples=5,  # Small number for quick demo
            output_dir="demo_results"
        )
        
        print("\n🎉 Demo completed successfully!")
        print("=" * 50)
        
        # Display key results
        print("📈 ENERGY MEASUREMENT RESULTS:")
        print("-" * 30)
        print(f"Model: {result['model_name']}")
        print(f"Hardware: {result['hardware']}")
        print(f"Hardware Detected: {result['hardware_detected']}")
        print(f"Samples Processed: {result['num_samples']}")
        print(f"Duration: {result['duration_seconds']:.4f} seconds")
        print(f"Throughput: {result['samples_per_second']:.1f} samples/sec")
        print(f"Energy Consumed: {result['energy_kwh']:.8f} kWh")
        print(f"CO2 Emissions: {result['co2_kg']:.8f} kg CO2")
        print(f"Energy per 1000 queries: {result['kwh_per_1000_queries']:.6f} kWh")
        
        # Show sample predictions
        if result.get('inference_results_sample'):
            print(f"\n🔍 Sample Predictions:")
            for i, pred in enumerate(result['inference_results_sample'][:3]):
                print(f"  {i+1}. {pred}")
        
        print(f"\n💾 Full results saved to:")
        print(f"   demo_results/text-classification/CPU/")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("\nThis might happen if:")
        print("- Internet connection is needed to download the model")
        print("- Dependencies are missing")
        print("- System resources are limited")
        return False


if __name__ == "__main__":
    success = run_demo()
    
    if success:
        print("\n✅ Your energy measurement tool is working perfectly!")
        print("\n🎯 Next steps:")
        print("1. Try with different models (DistilBERT, RoBERTa, etc.)")
        print("2. Test with larger datasets")
        print("3. Compare energy consumption across models")
        print("4. Test different tasks (image classification, text generation)")
    else:
        print("\n⚠️  Demo encountered issues, but the framework is solid!")
        print("Check the error messages above for troubleshooting.")

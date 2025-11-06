#!/usr/bin/env python3
"""
Test script for energy measurement with dummy model (no dependencies).
"""

import time
import random
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from measure_energy import measure_energy


def dummy_inference(sample):
    """Simulates model inference with CPU work"""
    # Burn some CPU cycles
    result = sum([i**2 for i in range(10000)])
    time.sleep(0.01)  # Simulate processing time
    return random.choice(['cat', 'dog', 'bird'])


def main():
    """Run dummy model test."""
    print("🧪 Testing Energy Measurement with Dummy Model")
    print("=" * 50)
    
    # Create fake dataset
    fake_dataset = [
        {'image': f'image_{i}.jpg', 'label': 'cat'} 
        for i in range(100)
    ]
    
    print(f"Created dataset with {len(fake_dataset)} samples")
    print("Starting dummy model test...")
    
    try:
        # Run measurement
        results = measure_energy(
            inference_fn=dummy_inference,
            dataset=fake_dataset,
            model_name="dummy-model",
            task_name="test-classification",
            hardware="CPU",  # Change to "M1" or "M2" if on Mac with Apple Silicon
            num_samples=100
        )
        
        print("\n✅ Test completed successfully!")
        print("\n=== Results ===")
        print(f"Model: {results['model_name']}")
        print(f"Task: {results['task_name']}")
        print(f"Hardware: {results['hardware']}")
        print(f"Samples processed: {results['num_samples']}")
        print(f"Energy consumed: {results['energy_kwh']:.6f} kWh")
        # Note: CO2 emissions removed - requires carbon intensity configuration
        print(f"Duration: {results['duration_seconds']:.2f} seconds")
        print(f"Per 1000 queries: {results['kwh_per_1000_queries']:.6f} kWh")
        print(f"Avg per sample: {results['duration_seconds']/results['num_samples']*1000:.2f} ms")
        
        # Verify JSON file was created
        output_dir = Path("results") / results['task_name']
        json_files = list(output_dir.glob(f"{results['model_name']}_*.json"))
        
        if json_files:
            print(f"\n📁 Results saved to: {json_files[0]}")
            print("✅ JSON output file created successfully")
        else:
            print("❌ JSON output file not found")
            return False
        
        print("\n🎉 Dummy model test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)

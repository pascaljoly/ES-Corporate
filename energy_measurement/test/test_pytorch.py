#!/usr/bin/env python3
"""
Test script for energy measurement with real PyTorch model.
"""

import torch
import torchvision.models as models
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from measure_energy import measure_energy


def main():
    """Run PyTorch model test."""
    print("🧪 Testing Energy Measurement with PyTorch Model")
    print("=" * 50)
    
    print("Loading MobileNetV2 model...")
    try:
        model = models.mobilenet_v2(pretrained=True)
        model.eval()
        print("✅ MobileNetV2 loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    def pytorch_inference(sample):
        """Real PyTorch inference on MobileNetV2"""
        # Create random image tensor (224x224 RGB)
        img_tensor = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = model(img_tensor)
        
        # Return predicted class
        return output.argmax().item()
    
    # Create fake dataset (simulating image data)
    dataset = [{'image_id': i} for i in range(100)]
    print(f"Created dataset with {len(dataset)} samples")
    
    print("Starting PyTorch model test...")
    
    try:
        # Run measurement
        results = measure_energy(
            inference_fn=pytorch_inference,
            dataset=dataset,
            model_name="mobilenet-v2",
            task_name="image-classification",
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
        print(f"Avg per sample: {results['duration_seconds']/results['num_samples']*1000:.2f} ms")
        print(f"Per 1000 queries: {results['kwh_per_1000_queries']:.6f} kWh")
        
        # Verify JSON file was created
        output_dir = Path("results") / results['task_name']
        json_files = list(output_dir.glob(f"{results['model_name']}_*.json"))
        
        if json_files:
            print(f"\n📁 Results saved to: {json_files[0]}")
            print("✅ JSON output file created successfully")
        else:
            print("❌ JSON output file not found")
            return False
        
        print("\n🎉 PyTorch model test passed!")
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

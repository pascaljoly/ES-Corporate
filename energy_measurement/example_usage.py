#!/usr/bin/env python3
"""
Example usage of the energy measurement script with a simple PyTorch model.
"""

import torch
import torch.nn as nn
import time
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from measure_energy import measure_energy


class SimpleModel(nn.Module):
    """A simple neural network for demonstration."""
    
    def __init__(self, input_size=10, hidden_size=64, output_size=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.layers(x)


def create_model():
    """Create and return a simple model."""
    model = SimpleModel()
    model.eval()  # Set to evaluation mode
    return model


def pytorch_inference(sample):
    """Inference function for PyTorch model."""
    # Convert sample to tensor
    if isinstance(sample, dict):
        # Assume sample has 'features' key
        input_tensor = torch.tensor(sample['features'], dtype=torch.float32).unsqueeze(0)
    else:
        # Assume sample is already a list/array
        input_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
    
    return output


def create_dummy_dataset(num_samples=100):
    """Create a dummy dataset for testing."""
    dataset = []
    for i in range(num_samples):
        # Create random features
        features = [float(j) for j in range(10)]  # 10 features
        dataset.append({
            'features': features,
            'label': i % 3,
            'id': i
        })
    return dataset


def example_text_classification():
    """Example with text classification task."""
    print("=== Text Classification Example ===")
    
    # Create dummy text dataset
    text_dataset = [
        {
            'text': f"This is sample text {i} for classification",
            'label': i % 3,
            'id': i
        }
        for i in range(100)
    ]
    
    def text_inference(sample):
        """Simple text processing inference."""
        # Simulate text processing
        text = sample['text']
        # Simulate some computation
        time.sleep(0.001)
        return {'prediction': len(text) % 3, 'confidence': 0.8}
    
    print("Measuring energy for text classification...")
    
    results = measure_energy(
        inference_fn=text_inference,
        dataset=text_dataset,
        model_name="text_classifier_v1",
        task_name="text-classification",
        hardware="CPU",
        num_samples=50,
        output_dir="example_results"
    )
    
    print(f"Text classification results: {results}")
    return results


def example_pytorch_model():
    """Example with PyTorch model."""
    print("\n=== PyTorch Model Example ===")
    
    global model
    model = create_model()
    
    # Create dummy dataset
    dataset = create_dummy_dataset(100)
    
    print("Measuring energy for PyTorch model...")
    
    results = measure_energy(
        inference_fn=pytorch_inference,
        dataset=dataset,
        model_name="simple_nn_v1",
        task_name="image-classification",
        hardware="CPU",
        num_samples=30,
        output_dir="example_results"
    )
    
    print(f"PyTorch model results: {results}")
    return results


def example_computer_vision():
    """Example with computer vision task."""
    print("\n=== Computer Vision Example ===")
    
    def cv_inference(sample):
        """Computer vision inference function."""
        # Simulate image processing
        image_data = sample['image_data']
        # Simulate some computation
        time.sleep(0.002)
        return {
            'predictions': [0.7, 0.2, 0.1],
            'bboxes': [[10, 20, 30, 40], [50, 60, 70, 80]]
        }
    
    # Create dummy image dataset
    cv_dataset = [
        {
            'image_data': [0.1] * 224 * 224 * 3,  # Simulate RGB image
            'image_id': i,
            'annotations': []
        }
        for i in range(50)
    ]
    
    print("Measuring energy for computer vision model...")
    
    results = measure_energy(
        inference_fn=cv_inference,
        dataset=cv_dataset,
        model_name="yolo_v8n",
        task_name="object-detection",
        hardware="CPU",
        num_samples=20,
        output_dir="example_results"
    )
    
    print(f"Computer vision results: {results}")
    return results


def main():
    """Run example usage scenarios."""
    print("🚀 Energy Measurement Examples")
    print("=" * 50)
    
    try:
        # Run examples
        text_results = example_text_classification()
        pytorch_results = example_pytorch_model()
        cv_results = example_computer_vision()
        
        print("\n" + "=" * 50)
        print("📊 Summary of Results:")
        print(f"Text Classification: {text_results['kwh_per_1000_queries']:.4f} kWh/1k queries")
        print(f"PyTorch Model: {pytorch_results['kwh_per_1000_queries']:.4f} kWh/1k queries")
        print(f"Computer Vision: {cv_results['kwh_per_1000_queries']:.4f} kWh/1k queries")
        
        print("\n✅ All examples completed successfully!")
        print("📁 Check the 'example_results' directory for saved JSON files")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

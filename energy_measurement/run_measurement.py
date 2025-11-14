#!/usr/bin/env python3
"""
CLI Wrapper for Energy Measurement Tool

This script allows you to measure energy consumption without writing any code.
Just point it to your existing inference function and dataset.

Usage:
    python run_measurement.py --help
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from energy_measurement.measure_energy import measure_energy


def import_from_file(file_path, item_name):
    """
    Import a function or variable from a Python file.

    Args:
        file_path: Path to the Python file
        item_name: Name of the function/variable to import

    Returns:
        The imported function or variable
    """
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load the module
    spec = importlib.util.spec_from_file_location("user_module", file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_module"] = module
    spec.loader.exec_module(module)

    # Get the item
    if not hasattr(module, item_name):
        raise AttributeError(f"'{item_name}' not found in {file_path}")

    return getattr(module, item_name)


def import_from_module(module_name, item_name):
    """
    Import a function or variable from an installed module.

    Args:
        module_name: Name of the module (e.g., 'my_package.models')
        item_name: Name of the function/variable to import

    Returns:
        The imported function or variable
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_name}': {e}")

    if not hasattr(module, item_name):
        raise AttributeError(f"'{item_name}' not found in module '{module_name}'")

    return getattr(module, item_name)


def load_item(source, item_name):
    """
    Load a function or variable from either a file or module.

    Args:
        source: Either a file path (e.g., 'models/my_model.py') or module name (e.g., 'my_package.models')
        item_name: Name of the function/variable to import

    Returns:
        The imported function or variable
    """
    # Check if source is a file path
    if '/' in source or source.endswith('.py'):
        return import_from_file(source, item_name)
    else:
        # Assume it's a module name
        return import_from_module(source, item_name)


def main():
    parser = argparse.ArgumentParser(
        description='Measure energy consumption of ML model inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  1. Using a file with your inference function and dataset:
     python run_measurement.py \\
       --inference-source my_model.py \\
       --inference-fn predict \\
       --dataset-source my_data.py \\
       --dataset-fn load_data \\
       --model-name "my_model_v1" \\
       --task "text-classification" \\
       --hardware CPU

  2. Using an installed package:
     python run_measurement.py \\
       --inference-source my_package.models \\
       --inference-fn run_inference \\
       --dataset-source my_package.data \\
       --dataset-fn get_test_data \\
       --model-name "bert_classifier" \\
       --task "sentiment-analysis" \\
       --hardware T4

  3. Dataset as a variable (not a function):
     python run_measurement.py \\
       --inference-source models.py \\
       --inference-fn my_inference \\
       --dataset-source data.py \\
       --dataset-var test_dataset \\
       --model-name "resnet50" \\
       --task "image-classification" \\
       --hardware GPU

Notes:
  - Inference function should accept one sample and return prediction
  - Dataset should be iterable (list, array, generator, etc.)
  - Use --dataset-fn if dataset is returned by a function
  - Use --dataset-var if dataset is a variable
        """
    )

    # Inference function parameters
    parser.add_argument(
        '--inference-source',
        required=True,
        help='File path (e.g., models/my_model.py) or module name (e.g., my_package.models)'
    )
    parser.add_argument(
        '--inference-fn',
        required=True,
        help='Name of the inference function'
    )

    # Dataset parameters
    parser.add_argument(
        '--dataset-source',
        required=True,
        help='File path (e.g., data/loader.py) or module name (e.g., my_package.data)'
    )

    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        '--dataset-fn',
        help='Name of function that returns the dataset'
    )
    dataset_group.add_argument(
        '--dataset-var',
        help='Name of variable containing the dataset'
    )

    # Measurement parameters
    parser.add_argument(
        '--model-name',
        required=True,
        help='Name of the model being measured'
    )
    parser.add_argument(
        '--task',
        required=True,
        help='Task name (e.g., "text-classification", "image-classification")'
    )
    parser.add_argument(
        '--hardware',
        required=True,
        choices=['CPU', 'T4', 'V100', 'A100', 'A100-80GB', 'H100', 'H100-SXM', 'M1', 'M2'],
        help='Hardware type'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='Number of samples to measure (default: 1000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Directory to save results (default: results)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ENERGY MEASUREMENT CLI")
    print("=" * 70)
    print()

    # Load inference function
    print(f"Loading inference function...")
    print(f"  Source: {args.inference_source}")
    print(f"  Function: {args.inference_fn}")
    try:
        inference_fn = load_item(args.inference_source, args.inference_fn)
        print(f"  ✓ Successfully loaded")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        sys.exit(1)
    print()

    # Load dataset
    print(f"Loading dataset...")
    print(f"  Source: {args.dataset_source}")
    dataset_item = args.dataset_fn or args.dataset_var
    print(f"  {'Function' if args.dataset_fn else 'Variable'}: {dataset_item}")
    try:
        dataset_item_obj = load_item(args.dataset_source, dataset_item)

        # If it's a function, call it to get the dataset
        if args.dataset_fn:
            print(f"  Calling {args.dataset_fn}() to load data...")
            dataset = dataset_item_obj()
        else:
            dataset = dataset_item_obj

        # Try to get dataset size
        try:
            dataset_size = len(dataset)
            print(f"  ✓ Successfully loaded ({dataset_size} samples)")
        except TypeError:
            print(f"  ✓ Successfully loaded (generator/iterator)")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        sys.exit(1)
    print()

    # Run measurement
    print(f"Measuring energy consumption...")
    print(f"  Model: {args.model_name}")
    print(f"  Task: {args.task}")
    print(f"  Hardware: {args.hardware}")
    print(f"  Samples: {args.num_samples}")
    print(f"  Seed: {args.seed}")
    print()

    try:
        results = measure_energy(
            inference_fn=inference_fn,
            dataset=dataset,
            model_name=args.model_name,
            task_name=args.task,
            hardware=args.hardware,
            num_samples=args.num_samples,
            seed=args.seed,
            output_dir=args.output_dir
        )

        print()
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print()
        print(f"✓ Energy consumed: {results['energy_kwh']:.6f} kWh")
        print(f"✓ Duration: {results['duration_seconds']:.2f} seconds")
        print(f"✓ Energy per 1000 queries: {results['kwh_per_1000_queries']:.6f} kWh")
        print()
        print(f"Samples measured: {results['num_samples']}")
        print(f"Timestamp: {results['timestamp']}")
        print()
        print("=" * 70)

        return 0

    except Exception as e:
        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)
        print()
        print(f"Measurement failed: {e}")
        print()
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

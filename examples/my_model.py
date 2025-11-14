"""
Example model file for CLI usage.

This shows what your model file might look like.
The CLI will import the predict() function from this file.
"""


def predict(sample):
    """
    Example inference function.

    Replace this with your actual model inference code.

    Args:
        sample: One item from your dataset

    Returns:
        Your model's prediction
    """
    # Example: Extract text from sample
    text = sample.get('text', '')

    # YOUR MODEL CODE GOES HERE
    # For example:
    # - prediction = your_model.predict(text)
    # - prediction = transformer(text)
    # - prediction = api_call(text)

    # Dummy prediction for demonstration
    prediction = len(text) % 3

    return {'prediction': prediction, 'confidence': 0.9}

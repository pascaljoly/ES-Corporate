"""
Example data loader file for CLI usage.

This shows what your data loading file might look like.
The CLI will import the load_dataset() function from this file.
"""


def load_dataset():
    """
    Load and return your dataset.

    Replace this with your actual data loading code.

    Returns:
        List or iterable of samples
    """
    # YOUR DATA LOADING CODE GOES HERE
    # For example:
    # - return pd.read_csv('data.csv').to_dict('records')
    # - return load_dataset('imdb', split='test')
    # - return json.load(open('data.json'))

    # Dummy dataset for demonstration
    dataset = [
        {'text': f'Sample text number {i}', 'label': i % 3}
        for i in range(100)
    ]

    return dataset


# Or you can have the dataset as a variable:
test_data = [
    {'text': 'First sample', 'label': 0},
    {'text': 'Second sample', 'label': 1},
    {'text': 'Third sample', 'label': 2},
]

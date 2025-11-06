# setup.py
from setuptools import setup, find_packages

setup(
    name="ml-energy-score",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "codecarbon>=2.4.0",
        "numpy>=1.20.0",
    ],
    author="Pascal Joly",
    description="Energy measurement and comparison tool for ML models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pascaljoly/ml-energy-score",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)

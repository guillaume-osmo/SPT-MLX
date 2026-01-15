"""
Setup script for SPT-MLX package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="spt-mlx",
    version="1.0.0",
    author="SPT-MLX Contributors",
    description="Standalone MLX implementation of SMILES2PropertiesTransformer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/SPT-MLX",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "mlx>=0.0.1",
        "numpy>=1.20.0",
        "rdkit-pypi>=2022.9.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
)

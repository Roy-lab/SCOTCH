from setuptools import setup, find_packages

setup(
    name="SCOTCH",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "seaborn",
        "PyEnrichAnalyzer",
        "anndata", 
        "torchmetrics", 
        "torch",
        "imageio"
    ],
    author="Spencer Halberg-Spencer",
    author_email="shalberg@wisc.edu",
    description="Single Cell Orthogonal Tri-factorication for Clustering High-dimensional data",
    url="https://github.com/Roy-lab/scotch",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)


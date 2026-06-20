"""
Setup script for audit-quality-panel-regression-bei package
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="audit-quality-panel-regression-bei",
    version="0.1.0",
    author="abiyyubarraq",
    author_email="siblal001@gmail.com",
    description="Panel data analysis for audit quality research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abiyyubarraq/audit-quality-panel-regression-bei",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "statsmodels>=0.14.0",
        "linearmodels>=5.3",
        "openpyxl>=3.1.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "tabulate>=0.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
        ],
    },
)

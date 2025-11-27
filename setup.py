from setuptools import setup, find_packages

setup(
    name="netrisk-project",
    version="1.0.0",
    description="NetRisk Project",
    author="Balint Mazzag",
    author_email="balint.mazzag@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "jupyter>=1.0.0",
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pytest>=7.4.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
)
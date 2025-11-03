from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pso",
    version="0.1.0",
    author="Stephen Monnet",
    author_email="stephen.monnet@outlook.com",
    description="A Particle Swarm Optimization (PSO) library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stiefen1/pso",
    project_urls={
        "Bug Tracker": "https://github.com/stiefen1/pso/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    entry_points={
        "console_scripts": [
            # Add any command-line scripts here if needed
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
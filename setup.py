#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open(os.path.join(this_directory, "requirements.txt"), encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="myvllm",
    version="0.1.0",
    author="wejoncy",
    author_email="",
    description="Multimodal LLM Serving Framework for Gemma 3-4B-IT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wejoncy/gemma_serving",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "benchmarks": [
            "matplotlib>=3.5.0",
            "numpy>=1.21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "myvllm-serve=myvllm.serving.api:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="llm, serving, gemma, multimodal, inference, pytorch, transformers",
    project_urls={
        "Bug Reports": "https://github.com/wejoncy/gemma_serving/issues",
        "Source": "https://github.com/wejoncy/gemma_serving",
    },
)
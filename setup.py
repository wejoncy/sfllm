#!/usr/bin/env python3

from setuptools import setup

# Package metadata and base requirements are configured in pyproject.toml.
# Keep optional groups here so "all" is derived rather than maintained twice.
extras_require = {
    "flashinfer": [
        "flashinfer-python>=0.6.18",
        "sglang-kernel>=0.4.6.post1",
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "httpx>=0.27.0",
        "black>=22.0.0",
        "isort>=5.0.0",
        "flake8>=5.0.0",
        "mypy>=1.0.0",
    ],
    "benchmarks": [
        "matplotlib>=3.5.0",
        "datasets>=2.15.0",
        "pillow>=10.0.1",
        "pybase64>=1.2.0",
        "modelscope>=1.10.0",
    ],
}
extras_require["all"] = list(
    dict.fromkeys(dep for group in extras_require.values() for dep in group)
)

setup(
    extras_require=extras_require,
    # Preserve the legacy Home-page field alongside project.urls in pyproject.toml.
    url="https://github.com/wejoncy/gemma_serving",
    zip_safe=False,
)

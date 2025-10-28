"""
sfllm: Multimodal LLM Serving Framework for Gemma 3-4B-IT

A high-performance serving framework that follows the OpenAI API protocol 
with support for both text and image inputs.
"""

__author__ = "wejoncy"
from .version import __version__
from . import engine
from . import serving
from . import models
from . import layers
from . import kernels

__all__ = [
    "engine",
    "serving", 
    "models",
    "layers",
    "kernels",
    "__version__",
]
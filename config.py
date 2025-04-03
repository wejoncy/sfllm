"""Configuration settings for the Gemma serving application."""

# Default model path
DEFAULT_MODEL_PATH = "/root/work/gemma-3-4b-it"

# Server settings
DEFAULT_HOST = "0.0.0.0"

# Queue settings
MAX_QUEUE_SIZE = 100
REQUEST_TIMEOUT = 600.0

# Inference settings
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95

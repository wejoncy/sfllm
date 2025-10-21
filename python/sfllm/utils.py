import logging
import os
import json
import torch

def configure_logger(server_args, prefix: str = ""):
    if SFLLM_LOGGING_CONFIG_PATH := os.getenv("SFLLM_LOGGING_CONFIG_PATH"):
        if not os.path.exists(SFLLM_LOGGING_CONFIG_PATH):
            raise Exception(
                "Setting SFLLM_LOGGING_CONFIG_PATH from env with "
                f"{SFLLM_LOGGING_CONFIG_PATH} but it does not exist!"
            )
        with open(SFLLM_LOGGING_CONFIG_PATH, encoding="utf-8") as file:
            custom_config = json.loads(file.read())
        logging.config.dictConfig(custom_config)
        return
    format = f"[%(asctime)s{prefix}] %(message)s"
    # format = f"[%(asctime)s.%(msecs)03d{prefix}] %(message)s"
    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format=format,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

def get_device_core_count(device_id: int = 0) -> int:
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_properties(device_id).multi_processor_count

    return 0
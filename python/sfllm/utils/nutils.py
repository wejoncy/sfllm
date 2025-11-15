import logging
import os
import json
import torch

DEFAULT_CUDA_GRAPH_BATCH_SIZES = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16]+list(range(20, 2048+1, 4))
MAX_PROCESSED_TOKENS = 1024*200

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

# @torch.compile(dynamic=True, backend="inductor")
def resolve_future_token_ids(input_ids, future_token_ids_map):
    input_ids[:] = torch.where(
        input_ids < 0,
        future_token_ids_map[torch.clamp(-input_ids, min=0)],
        input_ids,
    )
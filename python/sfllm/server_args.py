import dataclasses
import argparse
from typing import List, Literal, Optional

@dataclasses.dataclass
class ServerArgs:
    # Model and tokenizer
    model_path: str
    tokenizer_path: Optional[str] = None
    tokenizer_mode: str = "auto"
    tokenizer_worker_num: int = 1
    dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto"
    mem_fraction: float = 0.7
    max_context_length: int = 8192
    disable_overlap: bool = False
    # speculative decoding
    speculative_algorithm: Optional[str] = None
    draft_model_path: Optional[str] = None
    speculative_eagle_topk: int = 4
    speculative_num_steps: int = 4
    speculative_num_draft_tokens: int = 8
    
    # Optimization/debug options
    cuda_graph_max_bs: Optional[int] = 64
    cuda_graph_bs: Optional[List[int]] = None
    disable_cuda_graph: bool = False

    # Logging
    log_level: str = "info"
    enable_debug: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        # Model and tokenizer
        parser.add_argument(
            "--model-path",
            "--model",
            type=str,
            help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
            required=True,
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default=ServerArgs.dtype,
            choices=["float16", "bfloat16", "float32"],
            help="The data type for model weights and computations.",
        )
        parser.add_argument(
            "--tokenizer-path",
            type=str,
            default=ServerArgs.tokenizer_path,
            help="The path of the tokenizer.",
        )
        parser.add_argument(
            "--tokenizer-worker-num",
            type=int,
            default=ServerArgs.tokenizer_worker_num,
            help="The worker num of the tokenizer manager.",
        )
        parser.add_argument(
            "--cuda-graph-max-bs",
            type=int,
            default=ServerArgs.cuda_graph_max_bs,
            help="Set the maximum batch size for cuda graph. It will extend the cuda graph capture batch size to this value.",
        )
        parser.add_argument(
            "--cuda-graph-bs",
            type=int,
            nargs="+",
            help="Set the list of batch sizes for cuda graph.",
        )
        parser.add_argument(
            "--disable-cuda-graph",
            action="store_true",
            help="Disable cuda graph.",
        )
        parser.add_argument(
            "--disable-overlap",
            action="store_true",
            help="Disable overlapping of data transfer and computation.",
        )
        parser.add_argument(
            "--tokenizer-mode",
            type=str,
            default=ServerArgs.tokenizer_mode,
            choices=["auto", "slow"],
            help="Tokenizer mode. 'auto' will use the fast "
            "tokenizer if available, and 'slow' will "
            "always use the slow tokenizer.",
        )
        parser.add_argument(
            "--mem-fraction",
            type=float,
            default=ServerArgs.mem_fraction,
            help="The fraction of GPU memory to allocate for the model.",
        )
        parser.add_argument(
            "--max-context-length",
            type=int,
            default=ServerArgs.max_context_length,
            help="The maximum context length for the model.",
        )

        # Logging
        parser.add_argument(
            "--log-level",
            type=str,
            default=ServerArgs.log_level,
            help="The logging level of all loggers.",
        )

        # speculative decoding
        parser.add_argument(
            "--speculative-algorithm",
            type=str,
            default=ServerArgs.speculative_algorithm,
            choices=[None, "eagle3"],
            help="The speculative decoding algorithm to use.",
        )
        parser.add_argument(
            "--draft-model-path",
            type=str,
            default=ServerArgs.draft_model_path,
            help="The path of the draft model.",
        )
        parser.add_argument(
            "--speculative-eagle-topk",
            type=int,
            default=ServerArgs.speculative_eagle_topk,
            help="The top-k value for Eagle speculative decoding.",
        )
        parser.add_argument(
            "--speculative-num-steps",
            type=int,
            default=ServerArgs.speculative_num_steps,
            help="The number of speculative steps to perform.",
        )
        parser.add_argument(
            "--speculative-num-draft-tokens",
            type=int,
            default=ServerArgs.speculative_num_draft_tokens,
            help="The number of draft tokens to use in speculative decoding.",
        )
        parser.add_argument(
            "--enable-debug",
            action="store_true",
            help="Enable debug mode.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})

    def __post_init__(self):
        import platform
        if platform.system() == "Windows":
            self.mem_fraction = min(self.mem_fraction, 0.56)
            print("Warning: On Windows, setting mem_fraction to {self.mem_fraction} for better stability.")
        self.rl_on_policy_target = None
        if self.speculative_algorithm is not None:
            self.disable_overlap = True
        set_global_server_args_for_scheduler(self)

# NOTE: This is a global variable to hold the server args for scheduler.
_global_server_args: Optional[ServerArgs] = None


def set_global_server_args_for_scheduler(server_args: ServerArgs):
    global _global_server_args
    _global_server_args = server_args


def get_global_server_args() -> ServerArgs:
    if _global_server_args is None:
        raise ValueError("Global server args is not set yet!")

    return _global_server_args
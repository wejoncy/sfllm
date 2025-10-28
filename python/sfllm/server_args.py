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
    mem_fraction: float = 0.8
    max_context_length: int = 4096
    disable_overlap: bool = False
    
    # Optimization/debug options
    cuda_graph_max_bs: Optional[int] = 64
    cuda_graph_bs: Optional[List[int]] = None
    disable_cuda_graph: bool = False

    # Logging
    log_level: str = "info"

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

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})

    def __post_init__(self):
        import platform
        if platform.system() == "Windows":
            self.mem_fraction = min(self.mem_fraction, 0.7)
            self.disable_overlap = True
            print("Warning: On Windows, setting mem_fraction to 0.7 and disable_overlap to True for better stability.")
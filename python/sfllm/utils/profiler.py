import ctypes
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import torch


logger = logging.getLogger(__name__)


class SchedulerProfilerMixin:
    def __init__(self):
        self.torch_profiler = None
        self.torch_profiler_output_dir: Optional[Path] = None
        self.profile_steps: Optional[int] = None
        self.profiler_start_forward_ct: Optional[int] = None
        self.profile_in_progress: bool = False
        self.forward_ct:int = 0
        self.init_profile(None, None, None)
        self.profile_id = str(time.time())

    def init_profile(
        self,
        output_dir: Optional[str],
        start_step: Optional[int],
        num_steps: Optional[int],
    ) :
        if self.profile_in_progress:
            return

        # self.profile_by_stage = profile_by_stage

        if output_dir is None:
            output_dir = os.getenv("SFLLM_TORCH_PROFILER_DIR", "/tmp")
        self.torch_profiler_output_dir = Path(output_dir).expanduser()
        if start_step:
            self.profiler_start_forward_ct = max(start_step, self.forward_ct + 1)
        self.profile_steps = num_steps
        return

    def start_profiler(self):
        logger.info(
            f"Profiling starts. Traces will be saved to: {self.torch_profiler_output_dir} (with profile id: {self.profile_id})",
        )
        if os.getenv("NSYS_PROFILING_SESSION_ID") is not None:
            self.profile_in_progress = True
            import ctypes
            self.libcudart = ctypes.CDLL('libcudart.so')
            self.libcudart.cudaProfilerStart()
            logger.info("Nsys profiling detected, skip torch profiler...")
            return
        self.torch_profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                with_stack=True,
                record_shapes=False,
            )
        self.torch_profiler.start()
        self.profile_in_progress = True
        logger.info("Start profiling...")

    def stop_profiler(self):
        if not self.profile_in_progress:
            return
        if os.getenv("NSYS_PROFILING_SESSION_ID") is not None:
            torch.cuda.synchronize()
            self.libcudart.cudaProfilerStop()
            logger.info("Nsys profiling detected, skip torch profiler...")
            return
        logger.info("Stop profiling...")
        self.torch_profiler_output_dir.mkdir(parents=True, exist_ok=True)
        self.torch_profiler.stop()
        trace_file = os.path.join(
            self.torch_profiler_output_dir,
            self.profile_id + "-TP-0.trace.json.gz",
        )
        self.torch_profiler.export_chrome_trace(trace_file)
        self.profile_in_progress = False

import torch
import os

# Load the C++ extension dynamically
lib_path = os.path.join(os.path.dirname(__file__), "common_ops.cpython-312-x86_64-linux-gnu.so")
if os.path.exists(lib_path):
    torch.ops.load_library(lib_path)
else:
    # Try alternative loading approach
    try:
        import sf_kernel.common_ops
    except ImportError:
        pass

# Make the functions available
try:
    build_tree_kernel_efficient = torch.ops.sfkernels.build_tree_kernel_efficient
    verify_tree_greedy = torch.ops.sfkernels.verify_tree_greedy
    __all__ = ["build_tree_kernel_efficient", "verify_tree_greedy"]
except AttributeError:
    print("Warning: CUDA kernels not available. Functions may not be registered properly.")
    __all__ = []

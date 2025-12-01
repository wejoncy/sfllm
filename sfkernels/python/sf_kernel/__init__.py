import torch
import os
import platform
import sys
import sysconfig

# Load the CUDA extension library dynamically
try:
    # Try to load the compiled extension
    import sf_kernel.common_ops
    # The TORCH_LIBRARY_FRAGMENT will automatically register the ops
except ImportError:
    # Fallback: try to find the shared library manually
    # Get Python version info
    py_version = f"{sys.version_info.major}{sys.version_info.minor}"
    
    if platform.system() == "Windows":
        # Windows: common_ops.cp{version}-win_amd64.pyd
        arch = "win_amd64" if platform.machine().lower() in ['amd64', 'x86_64'] else "win32"
        lib_names = [f"common_ops.cp{py_version}-{arch}.pyd"]
    else:
        # Linux: common_ops.cpython-{version}-{arch}-linux-gnu.so
        arch = platform.machine().lower()
        if arch == 'x86_64':
            arch_suffix = "x86_64-linux-gnu"
        elif arch in ['aarch64', 'arm64']:
            arch_suffix = "aarch64-linux-gnu"
        else:
            arch_suffix = f"{arch}-linux-gnu"
        
        lib_names = [
            f"common_ops.cpython-{py_version}-{arch_suffix}.so",
            "common_ops.so"  # Fallback generic name
        ]
    
    for lib_name in lib_names:
        lib_path = os.path.join(os.path.dirname(__file__), lib_name)
        if os.path.exists(lib_path):
            torch.ops.load_library(lib_path)
            break
    else:
        print(f"Warning: Could not load sf_kernel CUDA extension. Tried: {lib_names}")

# Make the functions available
try:
    build_tree_kernel_efficient = torch.ops.sfkernels.build_tree_kernel_efficient
    verify_tree_greedy = torch.ops.sfkernels.verify_tree_greedy
    # rmsnorm = torch.ops.sfkernels.rmsnorm
    __all__ = ["build_tree_kernel_efficient", "verify_tree_greedy"]  # , "rmsnorm"]
except AttributeError:
    print("Warning: CUDA kernels not available. Functions may not be registered properly.")
    __all__ = []

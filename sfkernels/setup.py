# Copyright 2025 SGLang Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import platform
import sys
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()
arch = platform.machine().lower()


def _get_version():
    return "0.1.0"


operator_namespace = "sfkernels"
include_dirs = [
    root / "include",
    root / "csrc",
]

sources = [
    "csrc/common_extension.cc",
    "csrc/eagle_utils.cu",
]

cxx_flags = ["-O3"]
libraries = ["c10", "torch", "torch_python"]
extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib", f"-L/usr/lib/{arch}-linux-gnu"]

default_target = "70"
amdgpu_target = os.environ.get("AMDGPU_TARGET", default_target)

# if torch.cuda.is_available():
#     try:
#         amdgpu_target = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
#     except Exception as e:
#         print(f"Warning: Failed to detect GPU properties: {e}")
# else:
#     print(f"Warning: torch.cuda not available. Using default target: {amdgpu_target}")

if amdgpu_target not in ["70", "86"]:
    print(
        f"Warning: Unsupported GPU architecture detected '{amdgpu_target}'. Expected 'sm70' or 'sm86'."
    )
    sys.exit(1)

hipcc_flags = [
    "-DNDEBUG",
    f"-DOPERATOR_NAMESPACE={operator_namespace}",
    "-O3",
    "-Xcompiler",
    "-fPIC",
    "-std=c++17",
    "-gencode",
    f"arch=compute_{amdgpu_target},code=sm_{amdgpu_target}",
    "-DENABLE_BF16",
]

ext_modules = [
    CUDAExtension(
        name="sf_kernel.common_ops",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "nvcc": hipcc_flags,
            "cxx": cxx_flags,
        },
        libraries=libraries,
        extra_link_args=extra_link_args,
        py_limited_api=False,
    ),
]

setup(
    name="sf_kernel",
    version=_get_version(),
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)

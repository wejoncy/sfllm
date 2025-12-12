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
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()
arch = platform.machine().lower()


def _get_version():
    return "0.1.0"


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = ""):
        super().__init__(name, sources=[])
        self.sourcedir = str(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: Extension) -> None:
        import torch

        cmake_env = os.environ.copy()
        if platform.system() == "Windows":
            try:
                from setuptools._distutils._msvccompiler import _get_vc_env

                plat_name = getattr(self, "plat_name", None) or "win-amd64"
                plat_spec = "x64" if "amd64" in plat_name else "x86"
                cmake_env.update(_get_vc_env(plat_spec))
            except Exception:
                # Best effort: CMake may still work if the environment is already configured.
                pass

        ext_fullpath = Path(self.get_ext_fullpath(ext.name)).resolve()
        extdir = ext_fullpath.parent
        extdir.mkdir(parents=True, exist_ok=True)

        cfg = "Debug" if self.debug else "Release"
        build_temp = Path(self.build_temp) / ext.name.replace(".", "_")
        build_temp.mkdir(parents=True, exist_ok=True)

        # Generator selection: Ninja gives the best incremental experience.
        cmake_generator = os.environ.get("CMAKE_GENERATOR")
        use_ninja = cmake_generator is None and shutil.which("ninja") is not None

        # Keep the build directory stable -> incremental rebuilds.
        gpu_target = os.environ.get("GPU_TARGET")
        if gpu_target is None and torch.cuda.is_available() and torch.version.hip is None:
            try:
                prop = torch.cuda.get_device_properties(0)
                gpu_target = str(prop.major * 10 + prop.minor)
            except Exception:
                gpu_target = "86"
        gpu_target = gpu_target or "86"

        # PyTorch's CMake prefers TORCH_CUDA_ARCH_LIST (e.g. 8.6) over CMAKE_CUDA_ARCHITECTURES.
        try:
            gpu_target_int = int(gpu_target)
            torch_cuda_arch_list = f"{gpu_target_int // 10}.{gpu_target_int % 10}"
        except Exception:
            torch_cuda_arch_list = "8.6"

        torch_cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", torch_cuda_arch_list)

        operator_namespace = os.environ.get("OPERATOR_NAMESPACE", "sfkernels")

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={build_temp}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DGPU_TARGET={gpu_target}",
            f"-DOPERATOR_NAMESPACE={operator_namespace}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            # Let CMake find Torch via PyTorch's bundled CMake config.
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
            f"-DTORCH_CUDA_ARCH_LIST={torch_cuda_arch_list}",
        ]

        build_args = ["--config", cfg]
        if use_ninja:
            cmake_args += ["-G", "Ninja"]
        elif cmake_generator is not None:
            cmake_args += ["-G", cmake_generator]

        # Parallel build
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build_args += ["--", f"-j{os.cpu_count() or 8}"]

        subprocess.check_call(["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, env=cmake_env)
        subprocess.check_call(["cmake", "--build", ".", *build_args], cwd=build_temp, env=cmake_env)

        # CMake builds an untagged module name (e.g. common_ops.pyd). The setuptools
        # build pipeline expects an ABI-tagged filename (e.g. common_ops.cp312-win_amd64.pyd)
        # and may copy/rename it for us; clean up the untagged artifact.
        try:
            untagged = extdir / "common_ops.pyd"
            if untagged.exists() and untagged != ext_fullpath and ext_fullpath.exists():
                untagged.unlink()
                manifest = extdir / "common_ops.pyd.manifest"
                if manifest.exists():
                    manifest.unlink()
        except Exception:
            pass


# D:\Users\wen\miniconda3\Lib\site-packages\torch\include\torch\csrc\dynamo\compiled_autograd.h:1134
#  else if constexpr (::std::is_same_v<T, ::string>
operator_namespace = "sfkernels"
include_dirs = [
    root / "include",
    root / "csrc",
    root / "third_patry/flashinfer/include",
]

sources = [
    "csrc/common_extension.cc",
    "csrc/spec/eagle_utils.cu",
    "csrc/elementwise/activation.cu",
    "csrc/elementwise/layernorm.cu",
    "csrc/elementwise/rope.cu",
    "csrc/quant/per_token_quant_fp8.cu",
    "csrc/quant/per_tensor_quant_fp8.cu",
]

default_target = "70"
gpu_target = os.environ.get("GPU_TARGET", default_target)

if torch.cuda.is_available():
    try:
        if torch.version.hip is not None:
            gpu_target = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        else:
            prop = torch.cuda.get_device_properties(0)
            gpu_target = prop.major * 10 + prop.minor
            gpu_target = str(gpu_target)
    except Exception as e:
        print(f"Warning: Failed to detect GPU properties: {e}")
else:
    print(f"Warning: torch.cuda not available. Using default target: {gpu_target}")

cxx_flags = []
extra_link_args = []
nvcc_flags = [
    "-DNDEBUG",
    f"-DOPERATOR_NAMESPACE={operator_namespace}",
    "-O3",
    "-Xcompiler",
    "-fPIC",
    "-std=c++17",
    "-gencode",
    f"arch=compute_{gpu_target},code=sm_{gpu_target}",
    "-DENABLE_BF16",
    "-DFLASHINFER_ENABLE_F16",
    "-DFLASHINFER_ENABLE_BF16",
]
if os.name == "nt":
    # Windows specific flags
    cxx_flags = ["/openmp", "/std:c++17", "/MD"]
    torch_lib_path = Path(torch.__file__).parent / "lib"
    extra_link_args = [f"/LIBPATH:{torch_lib_path}"]
    libraries = []  # On Windows, libraries are linked via extra_link_args
else:
    if torch.version.hip is not None:
        cxx_flags = ["-O3"]
        libraries = ["hiprtc", "amdhip64", "c10", "torch", "torch_python"]
        extra_link_args = [
            "-Wl,-rpath,$ORIGIN/../../torch/lib", f"-L/usr/lib/{arch}-linux-gnu"]
        nvcc_flags = [
            "-DNDEBUG",
            f"-DOPERATOR_NAMESPACE={operator_namespace}",
            "-O3",
            "-Xcompiler",
            "-fPIC",
            "-std=c++17",
            f"--amdgpu-target={gpu_target}",
            "-DENABLE_BF16",
            "-DENABLE_FP8",
        ]
    elif torch.version.cuda is not None:
        # Linux specific flags
        cxx_flags = ["-O3", "-fopenmp", "-lgomp", "-std=c++17"]
        extra_link_args = [
            "-Wl,-rpath,$ORIGIN/../../torch/lib", f"-L/usr/lib/{arch}-linux-gnu"]
        libraries = ["c10", "torch", "torch_python"]
    else:
        assert False, "Unsupported platform or CUDA/HIP not available."


if gpu_target not in ["70", "86", "80", "90", "gfx942"]:
    print(
        f"Warning: Unsupported GPU architecture detected '{gpu_target}'. Expected 'sm70' or 'sm86'."
    )
    sys.exit(1)

# NVCC flags - platform specific
if os.name == "nt":
    # Windows NVCC flags
    nvcc_flags.remove("-fPIC")
    nvcc_flags.remove("-Xcompiler")

ext_modules = [
    CUDAExtension(
        name="sf_kernel.common_ops",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "nvcc": nvcc_flags,
            "cxx": cxx_flags,
        },
        libraries=libraries,
        extra_link_args=extra_link_args,
        py_limited_api=False,
    ),
]

use_cmake = os.environ.get("USE_CMAKE", "0") in ("1", "true", "True")
if use_cmake:
    ext_modules = [CMakeExtension("sf_kernel.common_ops", sourcedir=str(root))]
    cmdclass = {"build_ext": CMakeBuild}
else:
    cmdclass = {"build_ext": BuildExtension.with_options(use_ninja=True)}

setup(
    name="sf_kernel",
    version=_get_version(),
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)

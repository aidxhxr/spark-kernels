"""Build the spark_kernels PyTorch extension.

    pip install -e . --no-build-isolation      # RTX 5090 (CUDA 13, torch with sm_120 support)
    TORCH_CUDA_ARCH_LIST="12.0" pip install -e . --no-build-isolation   # same thing, explicit
    TORCH_CUDA_ARCH_LIST="12.1" pip install -e . --no-build-isolation   # DGX Spark (GB10)

The default architecture is compute capability 12.0 (RTX 5090 and the other RTX Blackwell
cards). Set TORCH_CUDA_ARCH_LIST to build for something else: "12.1" for the DGX Spark,
"12.0;12.1" for both.
"""

import glob
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Target architecture. torch's BuildExtension honors TORCH_CUDA_ARCH_LIST and, if it is
# set, appends its own -gencode flags. If it is *not* set we pin the RTX 5090 (sm_120) explicitly
# so the build does not fall back to "whatever GPU torch detects" heuristics.
# ---------------------------------------------------------------------------
arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
gencode_flags = []
if not arch_list:
    gencode_flags = ["-gencode", "arch=compute_120,code=sm_120"]

nvcc_flags = [
    "-O3",
    "-lineinfo",  # keep source correlation for Nsight Compute
    "--expt-relaxed-constexpr",
    "-std=c++17",
] + gencode_flags

sources = [os.path.join("python", "csrc", "bindings.cpp")] + sorted(
    glob.glob(os.path.join(ROOT, "src", "kernels", "*.cu"))
)

ext = CUDAExtension(
    name="spark_kernels._C",
    sources=sources,
    include_dirs=[os.path.join(ROOT, "include")],
    extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": nvcc_flags},
)

setup(
    name="spark-kernels",
    version="0.1.0",
    packages=["spark_kernels"],
    package_dir={"": "python"},
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    zip_safe=False,
)

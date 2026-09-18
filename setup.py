"""Build the spark_kernels PyTorch extension.

    pip install -e .            # on the DGX Spark (CUDA 13, torch with sm_121 support)
    TORCH_CUDA_ARCH_LIST="12.1" pip install -e .   # same thing, explicit

The default architecture is compute capability 12.1 (GB10 / DGX Spark). Set
TORCH_CUDA_ARCH_LIST to build for something else, e.g. "12.0" for RTX Blackwell cards
or "8.9" for Ada.
"""

import glob
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Target architecture. torch's BuildExtension honors TORCH_CUDA_ARCH_LIST and, if it is
# set, appends its own -gencode flags. If it is *not* set we pin GB10 (sm_121) explicitly
# so the build does not fall back to "whatever GPU torch detects" heuristics.
# ---------------------------------------------------------------------------
arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
gencode_flags = []
if not arch_list:
    gencode_flags = ["-gencode", "arch=compute_121,code=sm_121"]

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

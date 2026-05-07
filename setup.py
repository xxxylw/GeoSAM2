# Copyright 2025 VAST-AI-Research and the GeoSAM2 authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Build script for GeoSAM2.

Most of the project is pure Python and is configured in ``pyproject.toml``.
This file exists to compile the optional CUDA extension that accelerates
connected-components labelling. The Python package is still importable and
usable without the extension (a Python fallback path is taken at runtime).
"""

from __future__ import annotations

import os

from setuptools import setup

BUILD_CUDA_EXT = os.environ.get("GEOSAM2_BUILD_CUDA", "1") not in {"0", "false", "False"}


def _maybe_cuda_extensions():
    """Return the CUDA extension list, or an empty list if Torch/CUDA is unavailable."""
    if not BUILD_CUDA_EXT:
        return []
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except ImportError:
        # Torch is required to build the extension; skip silently so that a
        # `pip install` without torch still installs the Python sources.
        return []

    srcs = ["sam2/csrc/connected_components.cu"]
    compile_args = {
        "cxx": ["-O3"],
        "nvcc": [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ],
    }
    return [CUDAExtension("sam2._C", srcs, extra_compile_args=compile_args)]


def _cmdclass():
    if not BUILD_CUDA_EXT:
        return {}
    try:
        from torch.utils.cpp_extension import BuildExtension
    except ImportError:
        return {}
    return {"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)}


setup(
    ext_modules=_maybe_cuda_extensions(),
    cmdclass=_cmdclass(),
)

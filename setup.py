"""
Alternative build script for the 'aligner' C++ module.
Uses setuptools + pybind11 directly, no cmake needed.

Usage:
    pip install pybind11
    python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
import pybind11
import sys
import os

# The #include chain means band_doubling.cpp is the single compilation unit
sources = [os.path.join("aligner", "band_doubling.cpp")]

extra_compile_args = []
extra_link_args = []
define_macros = []

if sys.platform == "win32":
    extra_compile_args = ["/O2", "/std:c++17", "/DNDEBUG", "/EHsc"]
    # Try to enable AVX2 on MSVC
    extra_compile_args.append("/arch:AVX2")
    define_macros.append(("HAVE_AVX2", None))
else:
    extra_compile_args = [
        "-O3", "-std=c++17", "-DNDEBUG",
        "-march=native", "-ffast-math",
    ]
    # GCC/Clang AVX2
    extra_compile_args.append("-mavx2")
    define_macros.append(("HAVE_AVX2", None))

ext = Extension(
    name="aligner",
    sources=sources,
    include_dirs=[
        pybind11.get_include(),
        "aligner",
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    define_macros=define_macros,
    language="c++",
)

setup(
    name="msa_band_neural",
    version="0.1.0",
    ext_modules=[ext],
)

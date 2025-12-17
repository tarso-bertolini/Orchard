import sys
from setuptools import setup, Extension
import pybind11
import distutils.unixccompiler

# Monkeypatch to allow .mm files
if ".mm" not in distutils.unixccompiler.UnixCCompiler.src_extensions:
    distutils.unixccompiler.UnixCCompiler.src_extensions.append(".mm")

# Compiler flags
cflags = ["-std=c++17", "-Isrc"]
ldflags = ["-framework", "Metal", "-framework", "Foundation"]

# Define the extension
orchard_core = Extension(
    "orchard_core",
    sources=[
        "src/bindings.cpp",
        "src/platform/metal_backend.mm",
        "src/runtime/tensor.cpp",
        "src/runtime/model.cpp",
        "src/runtime/kv_cache.cpp"
    ],
    include_dirs=["src", pybind11.get_include()],
    extra_compile_args=cflags,
    extra_link_args=ldflags,
    language="c++"
)

setup(
    name="orchard",
    version="0.1.0",
    description="High-performance LLM runtime for Apple Silicon",
    author="Orchard Team",
    packages=["orchard"],
    ext_modules=[orchard_core],
    install_requires=[
        "numpy",
        "safetensors",
        "tokenizers",
        "huggingface_hub"
    ],
    entry_points={
        "console_scripts": [
            "orchard=orchard.cli:main",
        ],
    },
    python_requires=">=3.9",
)

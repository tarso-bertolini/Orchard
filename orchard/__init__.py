from .model import Llama
from .loader import load_weights
import os
import orchard_core

# Monkey patch initialize to automatically pass the kernel path
_original_initialize = orchard_core.MetalBackend.initialize

def _initialize_wrapper(self, resource_path=None):
    if resource_path is None:
        # Default to the installed kernels directory
        resource_path = os.path.join(os.path.dirname(__file__), "kernels")
    return _original_initialize(self, resource_path)

orchard_core.MetalBackend.initialize = _initialize_wrapper

__version__ = "0.1.0"

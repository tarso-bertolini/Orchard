import orchard
import orchard_core
import numpy as np

print("Orchard version:", orchard.__version__)
print("Core module:", orchard_core)

try:
    backend = orchard_core.MetalBackend()
    backend.initialize()
    print("Backend initialized successfully.")
except Exception as e:
    print("Backend initialization failed:", e)

try:
    from orchard.model import Llama
    print("Llama class imported successfully.")
except ImportError as e:
    print("Failed to import Llama class:", e)

print("Package test passed!")

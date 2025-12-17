# Orchard

**A high-performance LLM runtime built natively for Apple Silicon.**

Orchard is a specialized inference engine designed to extract maximum performance from Apple Silicon (M1/M2/M3) chips. It features custom Metal kernels for 4-bit quantization, achieving state-of-the-art speed for local LLM inference.

## Key Features

*   **Apple Silicon Native**: Built directly on Metal (Objective-C++) for zero-overhead GPU access.
*   **4-bit Quantization**: Custom INT4 kernels allow running 7B models on devices with 8GB RAM.
*   **High Performance**: ~82x faster than CPU inference for quantized workloads.
*   **Python Bindings**: Lightweight Python control plane via `pybind11`.

---

## Installation

### Prerequisites
*   macOS 13.0+ (Ventura or later)
*   Apple Silicon (M1/M2/M3)
*   Python 3.9+
*   Xcode Command Line Tools (`xcode-select --install`)

### Building from Source

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/orchard.git
    cd orchard
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install pybind11 numpy setuptools
    ```

4.  **Build the extension:**
    ```bash
    python3 setup.py build_ext --inplace
    ```

---

## Usage

### High-Level API (Llama-2)

Orchard provides a high-level Python API to load and run models.

```python
from orchard import Llama

# Load a model (supports Hugging Face format with safetensors)
# This will automatically quantize weights to INT4 on load.
model = Llama("path/to/llama-2-7b-hf")

# Generate text
output = model.generate("The future of AI is", max_tokens=50)
print(output)
```

### Low-Level API (Metal Kernels)

You can also access the raw Metal backend directly:

```python
import orchard_core
import numpy as np

# Initialize the Metal backend
backend = orchard_core.MetalBackend()
backend.initialize()

print(f"Running on: {backend.get_device_name()}")

# Create tensors on GPU
M, N, K = 1024, 1024, 1024
t_a = orchard_core.Tensor(backend, [M, K], orchard_core.DType.Float32)
t_b = orchard_core.Tensor(backend, [K, N], orchard_core.DType.Float32)
t_c = orchard_core.Tensor(backend, [M, N], orchard_core.DType.Float32)

# Move data from NumPy to GPU
data_a = np.random.rand(M, K).astype(np.float32)
t_a.copy_from_host(data_a)

# Run Matrix Multiplication
backend.run_matmul(t_a, t_b, t_c, M, N, K)

# Get results back
result = np.zeros((M, N), dtype=np.float32)
t_c.copy_to_host(result)
```

### Running Benchmarks

We include scripts to verify performance on your machine:

```bash
# Run raw matrix multiplication benchmarks
python3 benchmarks/benchmark_metal.py

# Run a full Llama-2-7B layer simulation
python3 benchmarks/benchmark_llm_simulation.py
```

---

## Performance

On an **Apple M2**, Orchard achieves:

*   **INT4 GEMV (Llama-2 Layer)**: 0.42 ms (~82x faster than NumPy FP16)
*   **Projected Throughput**: ~6.7 tokens/sec (Llama-2-7B, 4-bit)

See [BENCHMARKS.md](docs/BENCHMARKS.md) for detailed analysis.

---

## Architecture

```
Python API (control plane)
   ↓ pybind11
C++ Runtime Core
   ↓ Obj-C++
Metal Execution Layer (MSL)
   ↓
Apple Silicon GPU
```

### Language responsibilities

| Layer    | Language | Responsibility                       |
| -------- | -------- | ------------------------------------ |
| API      | Python   | Model lifecycle, LoRA orchestration  |
| Runtime  | C++      | Execution graph, KV cache, residency |
| Platform | Obj-C++  | Metal device and command queues      |
| Kernels  | Metal    | Matmul, attention, LoRA fusion       |

Python never touches tensors, GPU memory, or per-token execution.

---

## Memory Model

| Component       | Location    | Notes                    |
| --------------- | ----------- | ------------------------ |
| Base weights    | GPU-private | Loaded once, read-only   |
| LoRA adapters   | GPU-private | Mutable, hot-swappable   |
| KV cache        | GPU-private | Persistent across tokens |
| Activations     | GPU-private | Transient                |
| Control objects | CPU         | Python only              |

Base model weights are loaded once and never evicted during inference. This design exploits Apple’s unified memory architecture directly.

---

## LoRA as a Runtime Primitive

Orchard treats LoRA as a **runtime-level construct**, not a fine-tuning afterthought.

Design goals:

* Attach and detach adapters at runtime
* Multiple LoRAs active simultaneously
* Per-adapter alpha scaling
* Direct fusion into Metal projection kernels

Mathematically:

```
y = xW + α · (xB)A
```

The low-rank update path is fused into projection kernels. There are no separate execution passes and no graph recompilation.

---

## Performance Strategy

* INT8 / INT4 quantized base weights
* FP16 / BF16 LoRA paths
* Persistent KV cache
* Aggressive kernel fusion
* Token-level scheduling

Performance is achieved through memory residency and kernel fusion rather than kernel-level system modifications.

---

## Planned Python API (Conceptual)

```python
import orchard

model = orchard.load(
    "llama-8b",
    precision="int8",
    residency="persistent"
)

model.attach_lora("email_style.lora", alpha=0.8)
model.attach_lora("kotlin_dev.lora", alpha=0.6)

response = model.generate(
    "Write a professional Kotlin coroutine example",
    max_tokens=200
)
```

The API surface is intentionally minimal. Orchard is an execution engine, not a framework.

---

## Roadmap

### Phase 1 — Metal viability

* Standalone Metal matmul benchmarks
* FP16 and INT8 kernels
* Single transformer block

### Phase 2 — Native runtime core

* Persistent GPU residency
* Zero reallocation token loop

### Phase 3 — Python bindings

* Control-plane API
* No performance regression

### Phase 4 — LoRA inference

* GPU-resident adapters
* Runtime composition

### Phase 5 — KV cache optimization

* Long-context stability (4k–8k tokens)

### Phase 6 — Multi-LoRA composition

* Additive fusion in-kernel

### Phase 7 — On-device LoRA training (optional)

* Adapter-only backpropagation

---

## Project Status

**Early stage — architecture locked, implementation starting**

* No stable API yet
* No backward compatibility guarantees
* Rapid iteration expected

This repository exists to build the correct system deliberately, rather than shipping a prematurely abstracted solution.

---

## Intended Audience

* Systems engineers
* GPU and Metal developers
* ML runtime engineers
* Researchers focused on local AI
* Advanced Apple Silicon users

This project is not intended for plug-and-play inference workflows.

---

## License

License to be defined.

---

## Closing Note

Apple Silicon enables a different class of local AI systems.

Orchard exists to explore that space at the level of memory management, kernel design, and runtime architecture.

This repository represents the beginning of that effort.

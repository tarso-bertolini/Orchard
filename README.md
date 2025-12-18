# 🍎 Orchard

**High-Performance LLM Runtime for Apple Silicon**

[![PyPI Version](https://img.shields.io/pypi/v/orchard-llm?color=blue&style=flat-square)](https://pypi.org/project/orchard-llm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Platform: macOS](https://img.shields.io/badge/Platform-macOS%20(Apple%20Silicon)-black.svg?style=flat-square)](https://www.apple.com/mac/)
[![Metal: Native](https://img.shields.io/badge/Metal-Native-red.svg?style=flat-square)](https://developer.apple.com/metal/)

**Orchard** is a specialized inference engine built from the ground up to extract maximum performance from Apple Silicon (M1/M2/M3/M4) chips. By bypassing generic frameworks and targeting the Metal API directly with custom kernels, Orchard achieves state-of-the-art speed and efficiency for local Large Language Model (LLM) inference.

---

## 🚀 Key Features

- **Apple Silicon Native**: Built directly on Metal (Objective-C++) for zero-overhead GPU access. No PyTorch, no TensorFlow—just raw compute.
- **4-bit Quantization**: Custom `INT4` kernels allow running massive models on consumer hardware (e.g., 7B models on 8GB RAM) with minimal accuracy loss.
- **Blazing Fast**: Up to **82x faster** than CPU inference for quantized workloads.
- **Zero-Copy Architecture**: Fully exploits Apple's Unified Memory Architecture (UMA), sharing memory pointers between CPU and GPU where possible.
- **Pythonic Control**: A lightweight, high-level Python API controls the heavy lifting done in C++.

---

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install orchard-llm
```

### From Source

Prerequisites:
*   macOS 13.0+ (Ventura or later)
*   Apple Silicon (M1/M2/M3/M4)
*   Python 3.9+
*   Xcode Command Line Tools (`xcode-select --install`)

```bash
git clone https://github.com/tarso-bertolini/Orchard.git
cd Orchard
pip install .
```

---

## ⚡ Quick Start

### CLI Usage

Orchard comes with a powerful CLI to manage your local AI environment.

**1. Check System Compatibility**
```bash
orchard info
```

**2. Download & Run a Model**
```bash
# Download a model (e.g., Llama-2-7b)
orchard download

# Optimize it for Apple Silicon (4-bit quantization)
orchard optimize --model models/Llama-2-7b-chat --output models/Llama-2-7b-chat-opt

# Chat!
orchard run --model models/Llama-2-7b-chat-opt --prompt "Why is the sky blue?"
```

### Python API

Integrate Orchard into your own applications with just a few lines of code.

```python
from orchard import Llama

# Load an optimized model
model = Llama("models/Llama-2-7b-chat-opt")

# Generate text
prompts = [
    "Write a haiku about coding.",
    "Explain quantum computing in one sentence."
]

results = model.generate_batch(
    prompts=prompts,
    max_tokens=50,
    temperature=0.7
)

for prompt, result in zip(prompts, results):
    print(f"Prompt: {prompt}\nResult: {result}\n")
```

---

## 📊 Performance

**Device:** Apple M2 (Unified Memory)

| Operation | Implementation | Time | Speedup |
| :--- | :--- | :--- | :--- |
| **Matrix Mul (FP32)** | NumPy (CPU) | 15.15 ms | 1.0x |
| **Matrix Mul (FP32)** | Metal (GPU) | 26.82 ms | 0.6x (Overhead bound) |
| **Llama Layer (INT4)** | NumPy (FP16) | 34.58 ms | 1.0x |
| **Llama Layer (INT4)** | **Orchard (Metal)** | **0.42 ms** | **82.3x** |

*See [BENCHMARKS.md](docs/BENCHMARKS.md) for the full report.*

---

## 🏗 Architecture

Orchard uses a hybrid execution model to balance flexibility and performance.

```mermaid
graph TD
    User[User Python Script] --> API[Orchard Python API]
    API --> Loader[Model Loader & Quantizer]
    API --> Bindings[PyBind11 Interface]
    
    subgraph "Orchard Core (C++)"
        Bindings --> Backend[Metal Backend]
        Backend --> Tensor[Tensor Runtime]
    end
    
    subgraph "Metal GPU"
        Tensor --> Kernels[Custom Metal Shaders]
        Kernels --> MatMul[SIMD MatMul]
        Kernels --> RMS[RMSNorm]
        Kernels --> RoPE[Rotary Embeddings]
        Kernels --> Softmax[Softmax]
    end
```

---

## 🗺 Roadmap

- [x] **Phase 1: Metal Viability** (Kernels for MatMul, Softmax, RMSNorm)
- [x] **Phase 2: Runtime Core** (C++ Backend, Tensor Abstraction)
- [x] **Phase 3: Packaging** (CLI, PyPI structure, Offline Optimization)
- [ ] **Phase 4: Advanced Features** (Batched Inference, Continuous Batching)
- [ ] **Phase 5: LoRA Support** (Runtime adapter composition)

---

## 🤝 Contributing

Orchard is an open exploration of high-performance computing on Apple Silicon. Contributions are welcome!

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/amazing-feature`).
3.  Commit your changes (`git commit -m 'Add some amazing feature'`).
4.  Push to the branch (`git push origin feature/amazing-feature`).
5.  Open a Pull Request.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

*Built with 🍎 in Cupertino (and beyond).*

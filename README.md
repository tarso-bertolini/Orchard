# Orchard

**A high-performance LLM runtime built natively for Apple Silicon.**

Orchard is an early-stage systems project focused on extracting the maximum possible performance and personalization capability from Apple Silicon (M1+) when running large language models locally.

This repository marks the **beginning of the project**. The codebase is intentionally minimal today; the architecture, constraints, and execution plan are defined upfront to avoid premature abstraction and performance regressions.

---

## What Orchard Is

Orchard is **not** a general-purpose machine learning framework.

It is a **specialized LLM runtime** with the following design goals:

* Persistent GPU-resident model weights
* Metal-fused transformer kernels (attention and MLP)
* First-class LoRA support with runtime composition
* On-device personalization with no data leaving the machine
* Apple Silicon–native execution (M1 and newer only)

The Python API is intentionally thin. All performance-critical logic lives in native code.

---

## What Orchard Is Not

To be explicit, Orchard is **not**:

* A PyTorch replacement
* A Core ML wrapper
* A CUDA-style portability layer
* A research sandbox
* A kernel-level macOS modification

The project deliberately embraces Apple Silicon constraints instead of abstracting them away.

---

## Architecture Overview

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

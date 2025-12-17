# Project Status

**Date:** December 17, 2025

## Completed Phases

### Phase 1: Metal Viability
*   **Status:** Complete
*   **Description:** Verified that Metal kernels can be compiled (at runtime) and executed on the Apple Silicon GPU.
*   **Key Components:**
    *   `src/platform/metal_backend.mm`: Handles Metal device, command queue, and kernel compilation.
    *   `src/kernels/matmul.metal`: Naive matrix multiplication kernel.
    *   `Makefile`: Build system.

### Phase 2: Native Runtime Core
*   **Status:** Complete
*   **Description:** Implemented the core C++ runtime abstractions for memory management and execution.
*   **Key Components:**
    *   `src/runtime/tensor.h/cpp`: Manages GPU memory buffers, shapes, and dtypes.
    *   `src/runtime/model.h/cpp`: Manages persistent model weights.
    *   `src/runtime/kv_cache.h/cpp`: Manages persistent KV cache state.
    *   `src/main.cpp`: Demonstrates a zero-reallocation token generation loop.

## Next Steps

### Phase 3: Python Bindings
*   Integrate `pybind11`.
*   Expose `Model`, `Tensor` (opaque handle), and `generate` function to Python.

### Phase 4: LoRA Inference
*   Implement LoRA adapter loading.
*   Implement fused LoRA kernels (MatMul + LoRA).

### Phase 5: KV Cache Optimization
*   Implement actual Attention kernels that use the KV cache.
*   Optimize for long context.

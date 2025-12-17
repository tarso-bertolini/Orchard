import orchard_core
import numpy as np
import sys
import os

def test_full_suite():
    print("Initializing MetalBackend...")
    backend = orchard_core.MetalBackend()
    
    # Point to the local src/kernels directory for testing
    kernel_path = os.path.abspath("src/kernels")
    backend.initialize(kernel_path)
    
    if not backend.is_available():
        print("Metal is not available!")
        return

    print(f"Device: {backend.get_device_name()}")

    # --- 1. MatMul Test ---
    print("\n--- Testing MatMul ---")
    M, N, K = 128, 128, 128
    t_a = orchard_core.Tensor(backend, [M, K], orchard_core.DType.Float32)
    t_b = orchard_core.Tensor(backend, [K, N], orchard_core.DType.Float32)
    t_c = orchard_core.Tensor(backend, [M, N], orchard_core.DType.Float32)

    a_data = np.random.rand(M, K).astype(np.float32)
    b_data = np.random.rand(K, N).astype(np.float32)
    t_a.copy_from_host(a_data)
    t_b.copy_from_host(b_data)

    backend.run_matmul(t_a, t_b, t_c, M, N, K)
    
    c_data = np.zeros((M, N), dtype=np.float32)
    t_c.copy_to_host(c_data)
    expected_matmul = np.matmul(a_data, b_data)
    
    if np.allclose(c_data, expected_matmul, atol=1e-3):
        print("MatMul PASSED")
    else:
        print("MatMul FAILED")
        print("Max diff:", np.max(np.abs(c_data - expected_matmul)))

    # --- 2. Add Test ---
    print("\n--- Testing Add ---")
    size = 1024
    # Use Float16 for element-wise kernels as they are implemented as _fp16
    t_x = orchard_core.Tensor(backend, [size], orchard_core.DType.Float16)
    t_y = orchard_core.Tensor(backend, [size], orchard_core.DType.Float16)
    t_z = orchard_core.Tensor(backend, [size], orchard_core.DType.Float16)

    x_data = np.random.rand(size).astype(np.float16)
    y_data = np.random.rand(size).astype(np.float16)
    t_x.copy_from_host(x_data)
    t_y.copy_from_host(y_data)

    backend.run_add(t_x, t_y, t_z, size)

    z_data = np.zeros(size, dtype=np.float16)
    t_z.copy_to_host(z_data)
    expected_add = x_data + y_data

    if np.allclose(z_data, expected_add, atol=1e-3):
        print("Add PASSED")
    else:
        print("Add FAILED")

    # --- 3. Mul Test ---
    print("\n--- Testing Mul ---")
    backend.run_mul(t_x, t_y, t_z, size)
    t_z.copy_to_host(z_data)
    expected_mul = x_data * y_data

    if np.allclose(z_data, expected_mul, atol=1e-3):
        print("Mul PASSED")
    else:
        print("Mul FAILED")

    # --- 4. Silu Test ---
    print("\n--- Testing Silu ---")
    backend.run_silu(t_x, t_z, size)
    t_z.copy_to_host(z_data)
    
    # Silu = x * sigmoid(x) = x / (1 + exp(-x))
    # Compute in float32 for reference accuracy
    x_f32 = x_data.astype(np.float32)
    expected_silu = (x_f32 / (1 + np.exp(-x_f32))).astype(np.float16)

    if np.allclose(z_data, expected_silu, atol=1e-3):
        print("Silu PASSED")
    else:
        print("Silu FAILED")

    # --- 5. Softmax Test ---
    print("\n--- Testing Softmax ---")
    rows, cols = 10, 128
    t_sm_in = orchard_core.Tensor(backend, [rows, cols], orchard_core.DType.Float16)
    t_sm_out = orchard_core.Tensor(backend, [rows, cols], orchard_core.DType.Float16)

    sm_in_data = np.random.rand(rows, cols).astype(np.float16)
    t_sm_in.copy_from_host(sm_in_data)

    backend.run_softmax(t_sm_in, t_sm_out, rows, cols)

    sm_out_data = np.zeros((rows, cols), dtype=np.float16)
    t_sm_out.copy_to_host(sm_out_data)

    # Numpy softmax implementation (compute in float32)
    sm_in_f32 = sm_in_data.astype(np.float32)
    exp_data = np.exp(sm_in_f32 - np.max(sm_in_f32, axis=1, keepdims=True))
    expected_softmax = (exp_data / np.sum(exp_data, axis=1, keepdims=True)).astype(np.float16)

    if np.allclose(sm_out_data, expected_softmax, atol=1e-3):
        print("Softmax PASSED")
    else:
        print("Softmax FAILED")
        print("Max diff:", np.max(np.abs(sm_out_data - expected_softmax)))

    # --- 6. Embedding Test ---
    print("\n--- Testing Embedding ---")
    num_tokens = 4
    vocab_size = 100
    hidden_dim = 64
    
    # Input IDs (indices)
    input_ids = np.array([1, 50, 99, 0], dtype=np.int32)
    # Weights table (vocab_size, hidden_dim) - Float16
    weights_data = np.random.rand(vocab_size, hidden_dim).astype(np.float16)

    # Tensors
    t_ids = orchard_core.Tensor(backend, [num_tokens], orchard_core.DType.Float32) # Container for int32
    t_weights = orchard_core.Tensor(backend, [vocab_size, hidden_dim], orchard_core.DType.Float16)
    t_out = orchard_core.Tensor(backend, [num_tokens, hidden_dim], orchard_core.DType.Float16)

    t_ids.copy_from_host(input_ids)
    t_weights.copy_from_host(weights_data)

    backend.run_embedding(t_ids, t_weights, t_out, num_tokens, hidden_dim)

    out_data = np.zeros((num_tokens, hidden_dim), dtype=np.float16)
    t_out.copy_to_host(out_data)

    expected_emb = weights_data[input_ids]

    if np.allclose(out_data, expected_emb, atol=1e-3):
        print("Embedding PASSED")
    else:
        print("Embedding FAILED")
        print("Max diff:", np.max(np.abs(out_data - expected_emb)))

if __name__ == "__main__":
    test_full_suite()

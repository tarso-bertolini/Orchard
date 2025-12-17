import orchard_core
import numpy as np
import sys

def test_matmul():
    print("Initializing MetalBackend...")
    backend = orchard_core.MetalBackend()
    backend.initialize()
    
    if not backend.is_available():
        print("Metal is not available!")
        return

    print(f"Device: {backend.get_device_name()}")

    M, N, K = 16, 16, 16
    
    # Create tensors
    # Note: Tensor constructor takes (backend, shape, dtype)
    # DType.Float32 is 0 (based on enum order usually, but we exposed it)
    
    print("Creating tensors...")
    t_a = orchard_core.Tensor(backend, [M, K], orchard_core.DType.Float32)
    t_b = orchard_core.Tensor(backend, [K, N], orchard_core.DType.Float32)
    t_c = orchard_core.Tensor(backend, [M, N], orchard_core.DType.Float32)

    # Prepare data
    a_data = np.random.rand(M, K).astype(np.float32)
    b_data = np.random.rand(K, N).astype(np.float32)
    
    print("Copying data to device...")
    t_a.copy_from_host(a_data)
    t_b.copy_from_host(b_data)

    print("Running matmul...")
    backend.run_matmul(t_a, t_b, t_c, M, N, K)

    print("Copying result back...")
    c_data = np.zeros((M, N), dtype=np.float32)
    t_c.copy_to_host(c_data)

    # Verify
    expected = np.matmul(a_data, b_data)
    
    # Check for closeness
    if np.allclose(c_data, expected, atol=1e-4):
        print("Matmul test PASSED!")
    else:
        print("Matmul test FAILED!")
        print("Expected sample:", expected[0][:4])
        print("Actual sample:", c_data[0][:4])

def test_gemv_q4_0():
    print("\nTesting GEMV Q4_0...")
    backend = orchard_core.MetalBackend()
    backend.initialize()
    
    if not backend.is_available():
        print("Metal is not available!")
        return

    K = 32
    N = 1
    
    # Weights: N * (K/2) bytes
    # Scales: N * (K/32) halves (float16)
    # Input: K halves
    # Output: N halves
    
    # Create tensors
    # Note: We use Int8 for weights (uchar), Float16 for scales/input/output
    t_weights = orchard_core.Tensor(backend, [N, K // 2], orchard_core.DType.Int8)
    t_scales = orchard_core.Tensor(backend, [N, K // 32], orchard_core.DType.Float16)
    t_input = orchard_core.Tensor(backend, [K], orchard_core.DType.Float16)
    t_output = orchard_core.Tensor(backend, [N], orchard_core.DType.Float16)
    
    # Prepare data
    # We want weights to be 1.0.
    # Formula: w_real = scale * (w_quant - 8)
    # Let scale = 1.0. Then 1.0 = 1.0 * (w_quant - 8) => w_quant = 9.
    # Packed byte: 0x99 (153 decimal)
    
    weights_data = np.full((N, K // 2), 0x99, dtype=np.uint8)
    scales_data = np.full((N, K // 32), 1.0, dtype=np.float16)
    input_data = np.full((K,), 1.0, dtype=np.float16)
    
    print("Copying data...")
    t_weights.copy_from_host(weights_data)
    t_scales.copy_from_host(scales_data)
    t_input.copy_from_host(input_data)
    
    print("Running GEMV Q4_0...")
    backend.run_gemv_q4_0(t_weights, t_scales, t_input, t_output, K, N)
    
    print("Copying result...")
    output_data = np.zeros((N,), dtype=np.float16)
    t_output.copy_to_host(output_data)
    
    expected = 32.0
    print(f"Result: {output_data[0]}, Expected: {expected}")
    
    if np.isclose(output_data[0], expected, atol=0.1):
        print("GEMV Q4_0 test PASSED!")
    else:
        print("GEMV Q4_0 test FAILED!")

if __name__ == "__main__":
    try:
        test_matmul()
        test_gemv_q4_0()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

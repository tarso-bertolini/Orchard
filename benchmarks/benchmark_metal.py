import orchard_core
import numpy as np
import time

def benchmark_matmul(M, N, K, iterations=10):
    print(f"\nBenchmarking MatMul [{M}x{K}] x [{K}x{N}]...")
    
    # Initialize Backend
    backend = orchard_core.MetalBackend()
    backend.initialize()
    print(f"Device: {backend.get_device_name()}")

    # Create Tensors
    t_a = orchard_core.Tensor(backend, [M, K], orchard_core.DType.Float32)
    t_b = orchard_core.Tensor(backend, [K, N], orchard_core.DType.Float32)
    t_c = orchard_core.Tensor(backend, [M, N], orchard_core.DType.Float32)

    # Prepare data
    a_data = np.random.rand(M, K).astype(np.float32)
    b_data = np.random.rand(K, N).astype(np.float32)
    
    t_a.copy_from_host(a_data)
    t_b.copy_from_host(b_data)

    # Warmup
    print("Warming up...")
    for _ in range(3):
        backend.run_matmul(t_a, t_b, t_c, M, N, K)

    # Benchmark Metal
    print(f"Running Metal implementation ({iterations} iterations)...")
    start_time = time.time()
    for _ in range(iterations):
        backend.run_matmul(t_a, t_b, t_c, M, N, K)
    # Note: Metal is asynchronous, but copy_to_host synchronizes. 
    # To measure kernel time properly without copy overhead, we should ideally synchronize.
    # However, our current binding doesn't expose a standalone sync. 
    # We'll do a small copy to force sync or just rely on the fact that we are submitting fast.
    # Actually, let's just copy the result back once at the end to sync.
    # But that only syncs the last one.
    # For a rough benchmark, we can copy back every time or just accept that we are measuring submission time + GPU execution if the queue fills up.
    # Let's copy back every time to be fair and measure full roundtrip latency if we want, 
    # OR, better, just copy a single byte back to sync? No, we don't have that.
    # Let's just copy the result back every time. It includes PCIe transfer but it's a fair "end-to-end" test.
    
    # Actually, for pure compute comparison, let's try to minimize transfer.
    # But since we can't explicit sync, let's copy back.
    c_data = np.zeros((M, N), dtype=np.float32)
    t_c.copy_to_host(c_data) 
    end_time = time.time()
    
    # This time measurement is a bit flawed because the loop might finish submitting before GPU finishes, 
    # and only the last copy_to_host waits.
    # But if we copy back INSIDE the loop, we measure transfer time too.
    # Let's copy back inside the loop to be safe and measure "op + transfer".
    
    start_time = time.time()
    for _ in range(iterations):
        backend.run_matmul(t_a, t_b, t_c, M, N, K)
        t_c.copy_to_host(c_data)
    end_time = time.time()
    
    metal_time = (end_time - start_time) / iterations
    print(f"Metal Naive Average Time: {metal_time*1000:.2f} ms")

    # Benchmark Metal SIMD (Compute Only)
    print(f"Running Metal SIMD implementation (Compute Only)...")
    start_time = time.time()
    for _ in range(iterations):
        backend.run_matmul_simd(t_a, t_b, t_c, M, N, K)
    end_time = time.time()
    
    metal_simd_compute_time = (end_time - start_time) / iterations
    print(f"Metal SIMD Compute Time: {metal_simd_compute_time*1000:.2f} ms")

    # Benchmark NumPy (CPU)
    print(f"Running NumPy implementation ({iterations} iterations)...")
    start_time = time.time()
    for _ in range(iterations):
        np.matmul(a_data, b_data)
    end_time = time.time()
    
    numpy_time = (end_time - start_time) / iterations
    print(f"NumPy Average Time: {numpy_time*1000:.2f} ms")
    
    # Calculate GFLOPS
    flops = 2 * M * N * K
    metal_gflops = (flops / metal_simd_compute_time) / 1e9
    numpy_gflops = (flops / numpy_time) / 1e9
    
    print(f"Metal Performance: {metal_gflops:.2f} GFLOPS")
    print(f"NumPy Performance: {numpy_gflops:.2f} GFLOPS")

    print(f"Naive Speedup: {numpy_time / metal_time:.2f}x")
    # print(f"SIMD Speedup (incl. copy): {numpy_time / metal_simd_time:.2f}x") # metal_simd_time is not defined in this scope anymore
    print(f"SIMD Speedup (compute only): {numpy_time / metal_simd_compute_time:.2f}x")

def benchmark_gemv_q4(K, N, iterations=100):
    print(f"\nBenchmarking GEMV Q4_0 [Batch={N}, Hidden={K}]...")
    
    backend = orchard_core.MetalBackend()
    backend.initialize()
    
    # Create tensors
    # Weights: N * (K/2) bytes (Int8 storage)
    t_weights = orchard_core.Tensor(backend, [N, K // 2], orchard_core.DType.Int8)
    t_scales = orchard_core.Tensor(backend, [N, K // 32], orchard_core.DType.Float16)
    t_input = orchard_core.Tensor(backend, [K], orchard_core.DType.Float16)
    t_output = orchard_core.Tensor(backend, [N], orchard_core.DType.Float16)
    
    # Dummy data
    weights_data = np.zeros((N, K // 2), dtype=np.uint8)
    scales_data = np.zeros((N, K // 32), dtype=np.float16)
    input_data = np.zeros((K,), dtype=np.float16)
    
    t_weights.copy_from_host(weights_data)
    t_scales.copy_from_host(scales_data)
    t_input.copy_from_host(input_data)
    
    # Warmup
    for _ in range(10):
        backend.run_gemv_q4_0(t_weights, t_scales, t_input, t_output, K, N)
        
    # Benchmark Metal
    start_time = time.time()
    for _ in range(iterations):
        backend.run_gemv_q4_0(t_weights, t_scales, t_input, t_output, K, N)
    end_time = time.time()
    
    metal_time = (end_time - start_time) / iterations
    print(f"Metal INT4 GEMV Time: {metal_time*1000:.3f} ms")
    
    # Benchmark NumPy FP16 GEMV (Simulated)
    # NumPy doesn't support FP16 dot product well on all platforms, usually promotes to FP32.
    # Also, we want to compare against the "baseline" of running the model in FP16 on CPU.
    
    # Create FP16 weights for NumPy
    np_weights = np.zeros((N, K), dtype=np.float16)
    np_input = np.zeros((K,), dtype=np.float16)
    
    start_time = time.time()
    for _ in range(iterations):
        # NumPy dot: (N, K) @ (K,) -> (N,)
        # Note: NumPy might be slow with float16.
        np.dot(np_weights, np_input)
    end_time = time.time()
    
    numpy_time = (end_time - start_time) / iterations
    print(f"NumPy FP16 GEMV Time: {numpy_time*1000:.3f} ms")
    
    print(f"Speedup vs NumPy FP16: {numpy_time / metal_time:.2f}x")

if __name__ == "__main__":
    # Test with a reasonably large matrix
    benchmark_matmul(2048, 2048, 2048)
    # Test GEMV with Llama-2-7B sizes (Hidden=4096, Output=4096)
    # This simulates a single linear layer (e.g. q_proj)
    benchmark_gemv_q4(4096, 4096)

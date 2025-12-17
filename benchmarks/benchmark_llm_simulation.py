import orchard_core
import numpy as np
import time

def benchmark_llama2_7b_simulation():
    print("\n=== Llama-2-7B Performance Simulation (INT4) ===")
    print("Simulating a single token generation step (decode phase).")
    
    backend = orchard_core.MetalBackend()
    backend.initialize()
    
    if not backend.is_available():
        print("Metal not available.")
        return

    # Llama-2-7B Config
    n_layers = 32
    hidden_dim = 4096
    intermediate_dim = 11008
    n_heads = 32
    head_dim = 128 # 4096 / 32
    vocab_size = 32000
    
    print(f"Config: Layers={n_layers}, Hidden={hidden_dim}, MLP={intermediate_dim}")
    
    # We will simulate ONE layer fully, then multiply by n_layers.
    # We allocate tensors for one layer.
    
    # 1. Attention Weights (Q, K, V, O)
    # Shape: [Hidden, Hidden] -> [4096, 4096]
    # Quantized: [4096, 2048] (Int8) + Scales
    
    print("Allocating layer weights (INT4)...")
    
    # Helper to create quantized weight tensors
    def create_q4_weight(rows, cols):
        # rows = output dim (N), cols = input dim (K)
        # Metal kernel expects: weights [N, K/2], scales [N, K/32]
        w = orchard_core.Tensor(backend, [rows, cols // 2], orchard_core.DType.Int8)
        s = orchard_core.Tensor(backend, [rows, cols // 32], orchard_core.DType.Float16)
        # Fill with dummy data to avoid NaNs/crashes
        # We don't copy from host to save startup time, just use uninitialized (or zeroed by default?)
        # Metal buffers are usually zeroed on creation or contain garbage. 
        # For benchmarking speed, garbage is fine as long as it's valid numbers.
        # But let's zero them to be safe against NaNs slowing things down (denormals).
        # Actually, copying 0s takes time. Let's assume allocation is fast.
        return w, s

    # Attention Projections
    wq_w, wq_s = create_q4_weight(hidden_dim, hidden_dim)
    wk_w, wk_s = create_q4_weight(hidden_dim, hidden_dim)
    wv_w, wv_s = create_q4_weight(hidden_dim, hidden_dim)
    wo_w, wo_s = create_q4_weight(hidden_dim, hidden_dim)
    
    # MLP Projections
    # Gate & Up: [Intermediate, Hidden] -> [11008, 4096]
    w1_w, w1_s = create_q4_weight(intermediate_dim, hidden_dim) # Gate
    w3_w, w3_s = create_q4_weight(intermediate_dim, hidden_dim) # Up
    # Down: [Hidden, Intermediate] -> [4096, 11008]
    w2_w, w2_s = create_q4_weight(hidden_dim, intermediate_dim) # Down
    
    # RMSNorm weights (1D)
    norm_w = orchard_core.Tensor(backend, [hidden_dim], orchard_core.DType.Float32) # Usually FP32 for norm
    
    # Activation Tensors (FP16)
    # We reuse these to simulate memory flow
    input_state = orchard_core.Tensor(backend, [hidden_dim], orchard_core.DType.Float16)
    hidden_state = orchard_core.Tensor(backend, [hidden_dim], orchard_core.DType.Float16)
    mlp_gate = orchard_core.Tensor(backend, [intermediate_dim], orchard_core.DType.Float16)
    mlp_up = orchard_core.Tensor(backend, [intermediate_dim], orchard_core.DType.Float16)
    
    # Initialize input with random data
    input_host = np.random.rand(hidden_dim).astype(np.float16)
    input_state.copy_from_host(input_host)
    
    print("Warming up...")
    # Run a few dummy passes
    for _ in range(5):
        backend.run_gemv_q4_0(wq_w, wq_s, input_state, hidden_state, hidden_dim, hidden_dim)

    print("Running simulation loop (100 iterations)...")
    
    iterations = 100
    start_time = time.time()
    
    for _ in range(iterations):
        # Simulate one layer's operations
        
        # 1. RMSNorm (Pre-Attn) - Mocked (we have the kernel)
        # backend.run_rmsnorm(input_state, norm_w, hidden_state, 1e-5, hidden_dim, 1)
        
        # 2. Attention Projections (Q, K, V)
        # Input: hidden_state, Output: hidden_state (reused for simplicity of benchmark)
        backend.run_gemv_q4_0(wq_w, wq_s, input_state, hidden_state, hidden_dim, hidden_dim)
        backend.run_gemv_q4_0(wk_w, wk_s, input_state, hidden_state, hidden_dim, hidden_dim)
        backend.run_gemv_q4_0(wv_w, wv_s, input_state, hidden_state, hidden_dim, hidden_dim)
        
        # 3. RoPE (Mocked)
        # backend.run_rope(...)
        
        # 4. Attention Score (MatMul) - Skipped (small compared to projections for batch=1)
        
        # 5. Output Projection
        backend.run_gemv_q4_0(wo_w, wo_s, hidden_state, input_state, hidden_dim, hidden_dim)
        
        # 6. RMSNorm (Pre-FFN)
        # backend.run_rmsnorm(...)
        
        # 7. MLP Projections (Gate, Up)
        # Input: input_state, Output: mlp_gate, mlp_up
        backend.run_gemv_q4_0(w1_w, w1_s, input_state, mlp_gate, hidden_dim, intermediate_dim)
        backend.run_gemv_q4_0(w3_w, w3_s, input_state, mlp_up, hidden_dim, intermediate_dim)
        
        # 8. Silu/Mul - Skipped (element-wise, fast)
        
        # 9. Down Projection
        # Input: mlp_up (reused), Output: input_state
        backend.run_gemv_q4_0(w2_w, w2_s, mlp_up, input_state, intermediate_dim, hidden_dim)
        
        # 10. Residual Add - Skipped
        
    # Sync
    # Copy one small result back to force wait
    dummy = np.zeros((hidden_dim,), dtype=np.float16)
    input_state.copy_to_host(dummy)
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_layer = total_time / iterations
    
    # Total time for full model (32 layers)
    total_model_time = avg_time_per_layer * n_layers
    
    tps = 1.0 / total_model_time
    
    print(f"\nResults:")
    print(f"Average Layer Latency: {avg_time_per_layer*1000:.2f} ms")
    print(f"Projected Model Latency (32 layers): {total_model_time*1000:.2f} ms")
    print(f"Projected Throughput: {tps:.2f} tokens/sec")
    
    # Theoretical Bandwidth Check
    # Total weights per layer:
    # Attn: 4 * 4096*4096 * 0.5 bytes = 33.5 MB
    # MLP: 3 * 4096*11008 * 0.5 bytes = 67.6 MB
    # Total: ~101 MB per layer
    # Total Model: 3.2 GB
    
    bytes_per_layer = (4 * hidden_dim * hidden_dim + 3 * hidden_dim * intermediate_dim) * 0.5
    bandwidth = bytes_per_layer / avg_time_per_layer / 1e9 # GB/s
    print(f"Effective Memory Bandwidth: {bandwidth:.2f} GB/s")

if __name__ == "__main__":
    benchmark_llama2_7b_simulation()

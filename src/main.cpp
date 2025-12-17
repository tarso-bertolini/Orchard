#include "platform/metal_backend.h"
#include "runtime/tensor.h"
#include "runtime/model.h"
#include "runtime/kv_cache.h"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

using namespace orchard::platform;
using namespace orchard::runtime;

// Helper to convert float to half (stored as uint16_t for simplicity on host)
// In a real app we'd use a library or _Float16 if available.
uint16_t float_to_half(float x) {
    // Very basic truncation/conversion for testing. 
    // Real implementation needs proper IEEE 754 conversion.
    // For random weights in -1..1, this is "okay" for a viability test 
    // but will be numerically garbage.
    // Let's use a slightly better approximation or just 0 for now if we don't have a library.
    // Actually, let's just use _Float16 if clang supports it (it should on Apple Silicon).
    return 0; 
}

// We will use _Float16 for the benchmark data
using float16_t = _Float16;

// Helper for Q8_0 Quantization
struct Q8_0_Block {
    float16_t scale;
    int8_t data[32];
};

void quantize_q8_0(const std::vector<float>& src, std::vector<int8_t>& dst_weights, std::vector<float16_t>& dst_scales, int K, int N) {
    // Src is [N, K] (Transposed)
    // Dst Weights: [N, K]
    // Dst Scales: [N, K/32]
    
    dst_weights.resize(N * K);
    dst_scales.resize(N * (K / 32));
    
    for (int row = 0; row < N; ++row) {
        for (int k_blk = 0; k_blk < K; k_blk += 32) {
            // Find max abs
            float max_val = 0.0f;
            for (int i = 0; i < 32; ++i) {
                float val = std::abs(src[row * K + k_blk + i]);
                if (val > max_val) max_val = val;
            }
            
            float16_t scale = (float16_t)(max_val / 127.0f);
            if (scale == 0) scale = 1.0f; // Avoid div by zero
            
            dst_scales[row * (K / 32) + (k_blk / 32)] = scale;
            
            for (int i = 0; i < 32; ++i) {
                float val = src[row * K + k_blk + i];
                int8_t q = (int8_t)std::round(val / (float)scale);
                dst_weights[row * K + k_blk + i] = q;
            }
        }
    }
}

void quantize_q4_0(const std::vector<float>& src, std::vector<uint8_t>& dst_weights, std::vector<float16_t>& dst_scales, int K, int N) {
    // Src is [N, K] (Transposed)
    // Dst Weights: [N, K/2] (Packed 2 per byte)
    // Dst Scales: [N, K/32]
    
    dst_weights.resize(N * (K / 2));
    dst_scales.resize(N * (K / 32));
    
    for (int row = 0; row < N; ++row) {
        for (int k_blk = 0; k_blk < K; k_blk += 32) {
            // Find max abs
            float max_val = 0.0f;
            for (int i = 0; i < 32; ++i) {
                float val = std::abs(src[row * K + k_blk + i]);
                if (val > max_val) max_val = val;
            }
            
            float16_t scale = (float16_t)(max_val / 7.0f); // -7..7 range
            if (scale == 0) scale = 1.0f;
            
            dst_scales[row * (K / 32) + (k_blk / 32)] = scale;
            
            for (int i = 0; i < 32; i += 2) {
                // Low nibble
                float val0 = src[row * K + k_blk + i];
                int8_t q0 = (int8_t)std::round(val0 / (float)scale) + 8;
                if (q0 < 0) q0 = 0;
                if (q0 > 15) q0 = 15;
                
                // High nibble
                float val1 = src[row * K + k_blk + i + 1];
                int8_t q1 = (int8_t)std::round(val1 / (float)scale) + 8;
                if (q1 < 0) q1 = 0;
                if (q1 > 15) q1 = 15;
                
                uint8_t packed = (uint8_t)(q0 | (q1 << 4));
                dst_weights[row * (K / 2) + (k_blk / 2) + (i / 2)] = packed;
            }
        }
    }
}

int main() {
    std::cout << "Orchard Runtime - Phase 2: Performance Benchmark" << std::endl;

    MetalBackend backend;
    if (!backend.is_available()) {
        std::cerr << "Error: Metal is not available on this system." << std::endl;
        return 1;
    }
    backend.initialize();

    // Benchmark Dimensions (Llama-2-7B style layer)
    // 4096 hidden dim
    const int M = 4096; // Batch size (let's do a large batch to saturate GPU, or 1 for latency)
    // For "efficiency" testing, we usually want to see throughput on large matrices first.
    // Let's do M=128, N=4096, K=4096
    const int Batch = 128; 
    const int Dim = 4096;
    const int Hidden = 4096;

    std::cout << "Benchmarking Matrix Multiplication (" << Batch << "x" << Dim << " * " << Dim << "x" << Hidden << ")..." << std::endl;

    // 1. Naive FP32 Benchmark
    {
        Tensor t_A(backend, {static_cast<size_t>(Batch), static_cast<size_t>(Dim)}, DType::Float32);
        Tensor t_B(backend, {static_cast<size_t>(Dim), static_cast<size_t>(Hidden)}, DType::Float32);
        Tensor t_C(backend, {static_cast<size_t>(Batch), static_cast<size_t>(Hidden)}, DType::Float32);

        // Warmup
        backend.run_matmul(t_A.data(), t_B.data(), t_C.data(), Batch, Hidden, Dim);

        auto start = std::chrono::high_resolution_clock::now();
        int iters = 10;
        for(int i=0; i<iters; ++i) {
            backend.run_matmul(t_A.data(), t_B.data(), t_C.data(), Batch, Hidden, Dim);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        std::cout << "Naive FP32: " << duration / (double)iters << " ms avg" << std::endl;
    }

    // 2. SIMD FP16 Benchmark
    {
        // Allocate FP16 tensors (size is half)
        // We need to update Tensor class to support Float16 size calculation correctly (it does).
        Tensor t_A(backend, {static_cast<size_t>(Batch), static_cast<size_t>(Dim)}, DType::Float16);
        Tensor t_B(backend, {static_cast<size_t>(Dim), static_cast<size_t>(Hidden)}, DType::Float16);
        Tensor t_C(backend, {static_cast<size_t>(Batch), static_cast<size_t>(Hidden)}, DType::Float16);

        // Warmup
        backend.run_matmul_simd(t_A.data(), t_B.data(), t_C.data(), Batch, Hidden, Dim);

        auto start = std::chrono::high_resolution_clock::now();
        int iters = 100; // Run more iters as it should be faster
        for(int i=0; i<iters; ++i) {
            backend.run_matmul_simd(t_A.data(), t_B.data(), t_C.data(), Batch, Hidden, Dim);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        std::cout << "SIMD FP16:  " << duration / (double)iters << " ms avg" << std::endl;
    }

    // 3. INT8 GEMV Benchmark (Batch=1)
    {
        // For generation (Batch=1), we are memory bound.
        // Compare FP16 GEMV vs INT8 GEMV.
        // We'll use the SIMD kernel for FP16 (it handles Batch=1 too, though maybe not optimally).
        // And our new INT8 kernel.
        
        const int B_gen = 1;
        
        // FP16 Baseline (Batch=1)
        Tensor t_A_fp16(backend, {static_cast<size_t>(B_gen), static_cast<size_t>(Dim)}, DType::Float16);
        Tensor t_B_fp16(backend, {static_cast<size_t>(Dim), static_cast<size_t>(Hidden)}, DType::Float16);
        Tensor t_C_fp16(backend, {static_cast<size_t>(B_gen), static_cast<size_t>(Hidden)}, DType::Float16);
        
        auto start_fp16 = std::chrono::high_resolution_clock::now();
        int iters = 100;
        for(int i=0; i<iters; ++i) {
            backend.run_matmul_simd(t_A_fp16.data(), t_B_fp16.data(), t_C_fp16.data(), B_gen, Hidden, Dim);
        }
        auto end_fp16 = std::chrono::high_resolution_clock::now();
        double ms_fp16 = std::chrono::duration_cast<std::chrono::microseconds>(end_fp16 - start_fp16).count() / 1000.0 / iters;
        std::cout << "FP16 GEMV (Batch=1): " << ms_fp16 << " ms avg" << std::endl;

        // INT8 Quantized (Batch=1)
        // Weights: [Hidden, Dim] (Transposed for GEMV)
        // Scales: [Hidden, Dim/32]
        Tensor t_W_int8(backend, {static_cast<size_t>(Hidden), static_cast<size_t>(Dim)}, DType::Int8);
        Tensor t_S_int8(backend, {static_cast<size_t>(Hidden), static_cast<size_t>(Dim/32)}, DType::Float16);
        Tensor t_In_fp16(backend, {static_cast<size_t>(Dim)}, DType::Float16); // Flattened input
        Tensor t_Out_fp16(backend, {static_cast<size_t>(Hidden)}, DType::Float16); // Flattened output
        
        // Warmup
        backend.run_gemv_q8_0(t_W_int8.data(), t_S_int8.data(), t_In_fp16.data(), t_Out_fp16.data(), Dim, Hidden);
        
        auto start_int8 = std::chrono::high_resolution_clock::now();
        for(int i=0; i<iters; ++i) {
            backend.run_gemv_q8_0(t_W_int8.data(), t_S_int8.data(), t_In_fp16.data(), t_Out_fp16.data(), Dim, Hidden);
        }
        auto end_int8 = std::chrono::high_resolution_clock::now();
        double ms_int8 = std::chrono::duration_cast<std::chrono::microseconds>(end_int8 - start_int8).count() / 1000.0 / iters;
        std::cout << "INT8 GEMV (Batch=1): " << ms_int8 << " ms avg" << std::endl;
        std::cout << "Speedup: " << ms_fp16 / ms_int8 << "x" << std::endl;
    }

    // 4. INT4 GEMV Benchmark (Batch=1)
    {
        const int B_gen = 1;
        
        // INT4 Quantized (Batch=1)
        // Weights: [Hidden, Dim/2] (Packed)
        // Scales: [Hidden, Dim/32]
        // Note: We use Int8 DType for weights buffer, but size is halved.
        Tensor t_W_int4(backend, {static_cast<size_t>(Hidden), static_cast<size_t>(Dim/2)}, DType::Int8);
        Tensor t_S_int4(backend, {static_cast<size_t>(Hidden), static_cast<size_t>(Dim/32)}, DType::Float16);
        Tensor t_In_fp16(backend, {static_cast<size_t>(Dim)}, DType::Float16);
        Tensor t_Out_fp16(backend, {static_cast<size_t>(Hidden)}, DType::Float16);
        
        // Warmup
        backend.run_gemv_q4_0(t_W_int4.data(), t_S_int4.data(), t_In_fp16.data(), t_Out_fp16.data(), Dim, Hidden);
        
        auto start_int4 = std::chrono::high_resolution_clock::now();
        int iters = 100;
        for(int i=0; i<iters; ++i) {
            backend.run_gemv_q4_0(t_W_int4.data(), t_S_int4.data(), t_In_fp16.data(), t_Out_fp16.data(), Dim, Hidden);
        }
        auto end_int4 = std::chrono::high_resolution_clock::now();
        double ms_int4 = std::chrono::duration_cast<std::chrono::microseconds>(end_int4 - start_int4).count() / 1000.0 / iters;
        std::cout << "INT4 GEMV (Batch=1): " << ms_int4 << " ms avg" << std::endl;
        // We don't have ms_fp16 here easily unless we scope it out, but user can compare manually.
    }

    // 5. Full Transformer Block Simulation (Llama-2-7B style)
    // This simulates the latency of one token generation step for one layer.
    // Note: This is a simplified simulation (no Attention logic, just kernels).
    {
        std::cout << "\nSimulating Llama-2-7B Block (1 Token)..." << std::endl;
        
        // Dimensions
        const int B = 1;
        const int Seq = 1;
        const int H = 4096; // Hidden
        const int Heads = 32;
        const int HeadDim = 128;
        const int Inter = 11008; // Intermediate (MLP)

        // Tensors (FP16)
        Tensor x(backend, {B, H}, DType::Float16);
        Tensor rms_w(backend, {H}, DType::Float16);
        Tensor x_norm(backend, {B, H}, DType::Float16);
        
        // QKV Projections
        Tensor w_q(backend, {H, H}, DType::Float16);
        Tensor w_k(backend, {H, H}, DType::Float16);
        Tensor w_v(backend, {H, H}, DType::Float16);
        Tensor q(backend, {B, H}, DType::Float16);
        Tensor k(backend, {B, H}, DType::Float16);
        Tensor v(backend, {B, H}, DType::Float16);
        
        // RoPE (In-place on q, k)
        Tensor freqs_cos(backend, {HeadDim/2}, DType::Float16);
        Tensor freqs_sin(backend, {HeadDim/2}, DType::Float16);
        
        // Output Proj
        Tensor w_o(backend, {H, H}, DType::Float16);
        Tensor out_attn(backend, {B, H}, DType::Float16);
        
        // MLP
        Tensor w_gate(backend, {H, Inter}, DType::Float16);
        Tensor w_up(backend, {H, Inter}, DType::Float16);
        Tensor w_down(backend, {Inter, H}, DType::Float16);
        Tensor gate(backend, {B, Inter}, DType::Float16);
        Tensor up(backend, {B, Inter}, DType::Float16);
        Tensor down(backend, {B, H}, DType::Float16);

        // Warmup
        backend.run_rmsnorm(x.data(), rms_w.data(), x_norm.data(), 1e-5, H, B);
        
        auto start = std::chrono::high_resolution_clock::now();
        int iters = 50;
        for(int i=0; i<iters; ++i) {
            // 1. RMSNorm
            backend.run_rmsnorm(x.data(), rms_w.data(), x_norm.data(), 1e-5, H, B);
            
            // 2. QKV Proj
            backend.run_matmul_simd(x_norm.data(), w_q.data(), q.data(), B, H, H);
            backend.run_matmul_simd(x_norm.data(), w_k.data(), k.data(), B, H, H);
            backend.run_matmul_simd(x_norm.data(), w_v.data(), v.data(), B, H, H);
            
            // 3. RoPE
            backend.run_rope(q.data(), freqs_cos.data(), freqs_sin.data(), q.data(), HeadDim, Heads, Seq);
            backend.run_rope(k.data(), freqs_cos.data(), freqs_sin.data(), k.data(), HeadDim, Heads, Seq);
            
            // 4. Attention (Skipped - requires complex kernel, assume 0 cost for 1 token or add dummy delay)
            // For 1 token, attention is negligible compared to weights.
            
            // 5. Output Proj
            // Input to O is usually result of attention. Let's use 'q' as dummy.
            backend.run_matmul_simd(q.data(), w_o.data(), out_attn.data(), B, H, H);
            
            // 6. Residual Add (Skipped kernel, assume fast)
            
            // 7. RMSNorm 2
            backend.run_rmsnorm(out_attn.data(), rms_w.data(), x_norm.data(), 1e-5, H, B);
            
            // 8. MLP (Gate, Up, Down)
            backend.run_matmul_simd(x_norm.data(), w_gate.data(), gate.data(), B, Inter, H);
            backend.run_matmul_simd(x_norm.data(), w_up.data(), up.data(), B, Inter, H);
            // Silu/Mul (Skipped)
            backend.run_matmul_simd(gate.data(), w_down.data(), down.data(), B, H, Inter);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        double avg_latency = duration / (double)iters / 1000.0;
        std::cout << "Block Latency: " << avg_latency << " ms" << std::endl;
        std::cout << "Est. Tokens/sec (1 layer): " << 1000.0 / avg_latency << std::endl;
        std::cout << "Est. Tokens/sec (32 layers): " << 1000.0 / (avg_latency * 32) << std::endl;
    }

    return 0;
}

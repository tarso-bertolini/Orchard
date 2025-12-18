#include <metal_stdlib>
using namespace metal;

// Q4_0 Quantization Kernel for GEMM (Batch of Vectors)
// Weights: Packed 4-bit (2 per byte). [N, K/2]
// Scales: half [N, K/32]
// Input: half [B, K]
// Output: half [B, N]

kernel void gemm_q4_0(
    device const uchar* weights [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const half* input [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& B [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // gid.x = Output Row Index (0..N-1)
    // gid.y = Batch Index (0..B-1)
    
    uint n = gid.x;
    uint b = gid.y;
    
    if (n >= N || b >= B) return;
    
    float sum = 0.0f;
    
    // Pointers for this weight row
    device const uchar* w_row = weights + n * (K / 2);
    device const half* s_row = scales + n * (K / 32);
    
    // Pointer for this input row (batch b)
    device const half* in_row = input + b * K;
    
    // Loop over K in blocks of 32
    for (uint k = 0; k < K; k += 32) {
        float scale = (float)s_row[k / 32];
        
        // We process 32 weights (16 bytes) in 4 chunks of 4 bytes (8 weights)
        device const uchar4* w_ptr = (device const uchar4*)(w_row + k/2);
        device const half4* in_ptr = (device const half4*)(in_row + k);
        
        for (int j = 0; j < 4; ++j) {
            // Load 4 bytes (8 weights)
            uchar4 packed = w_ptr[j];
            
            // Unpack 8 weights
            uchar4 lo = packed & 0x0F;
            uchar4 hi = packed >> 4;
            
            float4 w_lo_f = float4(lo) - 8.0f;
            float4 w_hi_f = float4(hi) - 8.0f;
            
            // Load 8 inputs
            half4 in0 = in_ptr[2*j];
            half4 in1 = in_ptr[2*j + 1];
            
            // Accumulate
            sum += dot(w_lo_f * scale, float4(in0));
            sum += dot(w_hi_f * scale, float4(in1));
        }
    }
    
    // Write output
    output[b * N + n] = (half)sum;
}

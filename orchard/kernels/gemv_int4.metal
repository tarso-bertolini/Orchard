#include <metal_stdlib>
using namespace metal;

// Q4_0 Quantization Kernel for GEMV
// Weights: Packed 4-bit (2 per byte). [N, K/2]
// Scales: half [N, K/32]
// Input: half [K]
// Output: half [N]

kernel void gemv_q4_0(
    device const uchar* weights [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const half* input [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= N) return;
    
    float sum = 0.0f;
    
    // Pointers for this row
    device const uchar* w_row = weights + gid * (K / 2);
    device const half* s_row = scales + gid * (K / 32);
    
    // Loop over K in blocks of 32
    for (uint k = 0; k < K; k += 32) {
        float scale = (float)s_row[k / 32];
        
        // We process 32 weights (16 bytes) in 4 chunks of 4 bytes (8 weights)
        device const uchar4* w_ptr = (device const uchar4*)(w_row + k/2);
        device const half4* in_ptr = (device const half4*)(input + k);
        
        for (int j = 0; j < 4; ++j) {
            // Load 4 bytes (8 weights)
            uchar4 packed = w_ptr[j];
            
            // Unpack 8 weights
            // packed.x -> w0, w1 (corresponding to inputs 2*j*4 + 0, 1)
            // packed.y -> w2, w3
            // packed.z -> w4, w5
            // packed.w -> w6, w7
            
            uchar4 lo = packed & 0x0F;
            uchar4 hi = packed >> 4;
            
            float4 w_lo_f = float4(lo) - 8.0f;
            float4 w_hi_f = float4(hi) - 8.0f;
            
            // Load 8 inputs (2 half4 vectors)
            // in_ptr[2*j]   -> x0, x1, x2, x3
            // in_ptr[2*j+1] -> x4, x5, x6, x7
            
            half4 x0_3 = in_ptr[2*j];
            half4 x4_7 = in_ptr[2*j+1];
            
            float4 x_lo = float4(x0_3);
            float4 x_hi = float4(x4_7);
            
            // Shuffle inputs to match weights
            // w_lo contains w0, w2, w4, w6 (corresponding to x0, x2, x4, x6)
            // w_hi contains w1, w3, w5, w7 (corresponding to x1, x3, x5, x7)
            
            float4 x_evens = float4(x_lo.x, x_lo.z, x_hi.x, x_hi.z);
            float4 x_odds  = float4(x_lo.y, x_lo.w, x_hi.y, x_hi.w);
            
            sum += dot(w_lo_f, x_evens) * scale;
            sum += dot(w_hi_f, x_odds) * scale;
        }
    }
    
    output[gid] = (half)sum;
}

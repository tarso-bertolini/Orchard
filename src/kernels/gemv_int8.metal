#include <metal_stdlib>
using namespace metal;

// Q8_0 Quantization Kernel for GEMV
// Assumes weights are stored in Transposed format [N, K] (Row-major in memory)
// This allows contiguous memory access for the dot product of one output element.
//
// Weights: int8_t [N * K]
// Scales: half [N * (K / 32)]
// Input: half [K]
// Output: half [N]

kernel void gemv_q8_0(
    device const char* weights [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const half* input [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= N) return;
    
    float sum = 0.0f;
    
    // Pointers for this row (output element)
    device const char* w_row = weights + gid * K;
    device const half* s_row = scales + gid * (K / 32);
    
    // Loop over K in blocks of 32
    for (uint k = 0; k < K; k += 32) {
        half scale = s_row[k / 32];
        float scale_f = (float)scale;
        
        // Process 32 elements in chunks of 4
        for (uint i = 0; i < 32; i += 4) {
            // Load 4 weights (1 byte each)
            // We use uchar4 load and cast to preserve bit pattern if needed, 
            // but direct char4 pointer cast works for signed bytes.
            char4 w_vec = *((device const char4*)(w_row + k + i));
            
            // Load 4 inputs (2 bytes each)
            half4 x_vec = *((device const half4*)(input + k + i));
            
            // Convert to float for accumulation
            float4 w_f = float4(w_vec);
            float4 x_f = float4(x_vec);
            
            // Dot product
            sum += dot(w_f, x_f) * scale_f;
        }
    }
    
    output[gid] = (half)sum;
}

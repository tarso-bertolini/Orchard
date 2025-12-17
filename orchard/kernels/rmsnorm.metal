#include <metal_stdlib>
using namespace metal;

kernel void rms_norm(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant float& epsilon [[buffer(3)]],
    constant uint& N [[buffer(4)]], // Hidden dimension
    uint2 gid [[thread_position_in_grid]]
) {
    // One thread per row (token) is too slow.
    // One threadgroup per row is better.
    // For simplicity/viability: 1 thread per element, but reducing across row is hard.
    
    // Let's do a simple implementation: 1 thread per row (token).
    // This is NOT optimal for large N (4096), but easy to implement.
    // A better one uses threadgroup reduction.
    
    uint row = gid.x;
    // We assume grid is (Batch, 1, 1)
    
    // Calculate sum of squares
    float sum_sq = 0.0f;
    for (uint i = 0; i < N; ++i) {
        float val = (float)input[row * N + i];
        sum_sq += val * val;
    }
    
    float rms = rsqrt(sum_sq / N + epsilon);
    
    // Normalize and scale
    for (uint i = 0; i < N; ++i) {
        output[row * N + i] = (half)((float)input[row * N + i] * rms * (float)weight[i]);
    }
}

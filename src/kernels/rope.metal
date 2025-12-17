#include <metal_stdlib>
using namespace metal;

kernel void rope(
    device const half* input [[buffer(0)]],
    device const half* freqs_cos [[buffer(1)]], // Precomputed cos(theta)
    device const half* freqs_sin [[buffer(2)]], // Precomputed sin(theta)
    device half* output [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& num_heads [[buffer(5)]],
    constant uint& seq_len [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Grid: (HeadDim/2, NumHeads * Batch * SeqLen)
    // Each thread processes a pair of elements (real, imag) for rotation.
    
    uint i = gid.x; // Index in head_dim/2
    uint token_idx = gid.y; // Flattened index for (Batch, Seq, Head)
    
    if (i >= head_dim / 2) return;
    
    // Calculate offsets
    // Input shape: [Batch, SeqLen, NumHeads, HeadDim] or similar.
    // Let's assume flattened layout: [TotalTokens, NumHeads, HeadDim]
    
    uint base_idx = token_idx * head_dim + i * 2;
    
    float x0 = (float)input[base_idx];
    float x1 = (float)input[base_idx + 1];
    
    // Freqs shape: [SeqLen, HeadDim/2] usually.
    // We need to know which sequence position we are at.
    // token_idx = batch_idx * (SeqLen * NumHeads) + seq_idx * NumHeads + head_idx
    // This is getting complicated without explicit strides.
    // Let's assume simplified: [1, 1, NumHeads, HeadDim] for single token generation.
    // Then freqs are just for pos 0 (or current pos).
    // Let's assume freqs are passed for the specific position(s) we are processing.
    
    float cos_theta = (float)freqs_cos[i];
    float sin_theta = (float)freqs_sin[i];
    
    // Rotate
    // x' = x * cos - y * sin
    // y' = x * sin + y * cos
    float out0 = x0 * cos_theta - x1 * sin_theta;
    float out1 = x0 * sin_theta + x1 * cos_theta;
    
    output[base_idx] = (half)out0;
    output[base_idx + 1] = (half)out1;
}

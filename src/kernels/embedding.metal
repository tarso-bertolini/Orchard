#include <metal_stdlib>
using namespace metal;

kernel void embedding_forward(
    device const int* input_ids [[buffer(0)]],
    device const half* weights [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& hidden_dim [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint token_idx = id / hidden_dim;
    uint dim_idx = id % hidden_dim;
    
    int token_id = input_ids[token_idx];
    
    output[id] = weights[token_id * hidden_dim + dim_idx];
}

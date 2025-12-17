#include <metal_stdlib>
using namespace metal;

kernel void softmax_fp16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint blockDim [[threads_per_threadgroup]]
) {
    threadgroup float shared_data[1024];
    
    uint row_idx = bid;
    device const half* row_in = input + row_idx * N;
    device half* row_out = output + row_idx * N;
    
    // 1. Find Max
    float max_val = -1e38f;
    for (uint i = tid; i < N; i += blockDim) {
        max_val = max(max_val, float(row_in[i]));
    }
    shared_data[tid] = max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction for Max
    for (uint s = blockDim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    max_val = shared_data[0];
    
    // 2. Sum Exp
    float sum_exp = 0.0f;
    for (uint i = tid; i < N; i += blockDim) {
        sum_exp += exp(float(row_in[i]) - max_val);
    }
    shared_data[tid] = sum_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction for Sum
    for (uint s = blockDim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total_sum = shared_data[0];
    
    // 3. Write Output
    for (uint i = tid; i < N; i += blockDim) {
        row_out[i] = half(exp(float(row_in[i]) - max_val) / total_sum);
    }
}

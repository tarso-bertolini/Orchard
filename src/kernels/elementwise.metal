#include <metal_stdlib>
using namespace metal;

kernel void add_fp16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    c[id] = a[id] + b[id];
}

kernel void mul_fp16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* c [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    c[id] = a[id] * b[id];
}

kernel void silu_fp16(
    device const half* in [[buffer(0)]],
    device half* out [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    float x = float(in[id]);
    float sigmoid = 1.0f / (1.0f + exp(-x));
    out[id] = half(x * sigmoid);
}

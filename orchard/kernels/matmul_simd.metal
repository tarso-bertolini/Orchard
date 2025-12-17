#include <metal_stdlib>
using namespace metal;

kernel void matmul_simd_fp16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tg_id [[threadgroup_position_in_grid]]
) {
    // We assume 1 threadgroup = 1 SIMD group = 1 8x8 tile
    // Dispatch: (N/8, M/8, 1)
    
    uint output_row = tg_id.y * 8;
    uint output_col = tg_id.x * 8;
    
    if (output_row >= M || output_col >= N) return;

    simdgroup_matrix<half, 8, 8> acc;
    acc = simdgroup_matrix<half, 8, 8>(0.0h);

    for (uint k = 0; k < K; k += 8) {
        simdgroup_matrix<half, 8, 8> a_frag;
        simdgroup_matrix<half, 8, 8> b_frag;

        // Load A [Row, k]
        // Stride = K
        simdgroup_load(a_frag, A + output_row * K + k, K, ulong2(0, 0), false);

        // Load B [k, Col]
        // Stride = N
        simdgroup_load(b_frag, B + k * N + output_col, N, ulong2(0, 0), false);

        simdgroup_multiply_accumulate(acc, a_frag, b_frag, acc);
    }

    simdgroup_store(acc, C + output_row * N + output_col, N, ulong2(0, 0), false);
}

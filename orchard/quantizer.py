import numpy as np

def quantize_q4_0(weight_fp32, group_size=32):
    """
    Quantize a weight matrix (K, N) or (N, K) to Q4_0 format.
    Returns (packed_weights, scales).
    
    Args:
        weight_fp32: Numpy array of shape (out_features, in_features)
        group_size: Block size for quantization (default 32)
    """
    # Ensure shape is (N, K) where K is input dimension (reduction dim)
    # Our kernels expect row-major weights where each row is a output neuron.
    # weight_fp32 shape: [N, K]
    
    N, K = weight_fp32.shape
    assert K % group_size == 0, f"Input dimension {K} must be divisible by group size {group_size}"
    
    # Reshape to (N, K/group_size, group_size) to process blocks
    # We want to find scale for each block of 32 weights.
    w_reshaped = weight_fp32.reshape(N, K // group_size, group_size)
    
    # Find max absolute value in each block
    max_abs = np.max(np.abs(w_reshaped), axis=2)
    
    # Calculate scales (fp16)
    # We map range [-max, max] to [-7, 7] (integers)
    # scale = max_abs / 7.0
    scales = max_abs / 7.0
    scales = scales.astype(np.float16)
    
    # Avoid divide by zero
    scales[scales == 0] = 1e-5
    
    # Quantize
    # w_quant = round(w / scale) + 8
    # Result is in [1, 15] range usually. 0 is reserved or just unused in this symmetric scheme?
    # Actually Q4_0 in llama.cpp usually maps -8 to 7 or similar.
    # Let's stick to the kernel implementation:
    # float val = (float(packed & 0xF) - 8.0f) * scale;
    # So value 8 -> 0.0.
    # Value 0 -> -8.0 * scale.
    # Value 15 -> 7.0 * scale.
    
    # Broadcast scales back
    scales_expanded = scales[:, :, np.newaxis]
    
    # Divide and round
    w_scaled = w_reshaped / scales_expanded
    w_quant = np.round(w_scaled + 8.0).astype(np.int8)
    
    # Clamp to [0, 15]
    w_quant = np.clip(w_quant, 0, 15).astype(np.uint8)
    
    # Pack 2 weights into 1 byte
    # Layout: Low nibble = even index, High nibble = odd index?
    # Kernel: 
    # uchar4 lo = packed & 0x0F; -> w0
    # uchar4 hi = packed >> 4;   -> w1
    # So low bits are the first weight.
    
    # We need to pack along the last dimension (group_size)
    # w_quant shape: (N, blocks, 32)
    # We want to pack pairs (0,1), (2,3), etc.
    
    # Reshape to (N, blocks, 16, 2)
    w_pairs = w_quant.reshape(N, K // group_size, group_size // 2, 2)
    
    # w0 is w_pairs[..., 0], w1 is w_pairs[..., 1]
    # packed = (w1 << 4) | w0
    low = w_pairs[..., 0]
    high = w_pairs[..., 1]
    packed = (high << 4) | low
    
    # Result shape: (N, blocks, 16) -> flatten to (N, K/2)
    packed_weights = packed.reshape(N, K // 2)
    
    return packed_weights.astype(np.uint8), scales

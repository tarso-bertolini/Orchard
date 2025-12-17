#pragma once

#include "tensor.h"
#include <vector>

namespace orchard {
namespace runtime {

class KVCache {
public:
    KVCache(platform::MetalBackend& backend, 
            size_t max_seq_len, 
            size_t num_layers, 
            size_t num_heads, 
            size_t head_dim, 
            DType dtype);

    // Get the Key tensor for a specific layer
    Tensor* get_key(size_t layer);
    
    // Get the Value tensor for a specific layer
    Tensor* get_value(size_t layer);

    // Update the current sequence length (pointer to where next token goes)
    void step();
    void reset();
    size_t current_pos() const { return current_pos_; }

private:
    platform::MetalBackend& backend_;
    size_t max_seq_len_;
    size_t num_layers_;
    size_t num_heads_;
    size_t head_dim_;
    DType dtype_;
    size_t current_pos_;

    // We store one large tensor per layer or one huge tensor for all?
    // For simplicity, let's do vector of Tensors for now.
    // Shape: [Batch=1, NumHeads, MaxSeqLen, HeadDim]
    // Note: Usually KV cache is [Batch, NumHeads, MaxSeqLen, HeadDim] or similar.
    // We'll assume Batch=1 for local inference.
    std::vector<Tensor> k_cache_;
    std::vector<Tensor> v_cache_;
};

} // namespace runtime
} // namespace orchard

#include "kv_cache.h"

namespace orchard {
namespace runtime {

KVCache::KVCache(platform::MetalBackend& backend, 
                 size_t max_seq_len, 
                 size_t num_layers, 
                 size_t num_heads, 
                 size_t head_dim, 
                 DType dtype)
    : backend_(backend), 
      max_seq_len_(max_seq_len), 
      num_layers_(num_layers), 
      num_heads_(num_heads), 
      head_dim_(head_dim), 
      dtype_(dtype), 
      current_pos_(0) {

    k_cache_.reserve(num_layers);
    v_cache_.reserve(num_layers);

    for (size_t i = 0; i < num_layers; ++i) {
        // Shape: [1, NumHeads, MaxSeqLen, HeadDim]
        // Flattened: NumHeads * MaxSeqLen * HeadDim
        // We use 1 for batch size implicitly
        std::vector<size_t> shape = {1, num_heads, max_seq_len, head_dim};
        
        // We need to construct Tensors in place or move them
        // Since Tensor has no default constructor, we use emplace_back
        k_cache_.emplace_back(backend, shape, dtype);
        v_cache_.emplace_back(backend, shape, dtype);
    }
}

Tensor* KVCache::get_key(size_t layer) {
    if (layer >= k_cache_.size()) return nullptr;
    return &k_cache_[layer];
}

Tensor* KVCache::get_value(size_t layer) {
    if (layer >= v_cache_.size()) return nullptr;
    return &v_cache_[layer];
}

void KVCache::step() {
    if (current_pos_ < max_seq_len_) {
        current_pos_++;
    }
}

void KVCache::reset() {
    current_pos_ = 0;
}

} // namespace runtime
} // namespace orchard

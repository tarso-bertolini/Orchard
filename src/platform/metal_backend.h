#pragma once

#include <vector>
#include <string>

namespace orchard {
namespace platform {

class MetalBackend {
public:
    MetalBackend();
    ~MetalBackend();

    void initialize();
    bool is_available() const;
    std::string get_device_name() const;

    // Basic buffer management for testing
    void* create_buffer(size_t size);
    void copy_to_buffer(void* buffer, const void* data, size_t size);
    void copy_from_buffer(void* buffer, void* data, size_t size);
    void release_buffer(void* buffer);

    // Execute the matmul kernel
    void run_matmul(void* buffer_a, void* buffer_b, void* buffer_c, 
                   uint32_t M, uint32_t N, uint32_t K);

    // Execute the SIMD matmul kernel (FP16)
    void run_matmul_simd(void* buffer_a, void* buffer_b, void* buffer_c, 
                        uint32_t M, uint32_t N, uint32_t K);

    void run_rmsnorm(void* input, void* weight, void* output, float epsilon, uint32_t N, uint32_t count);
    void run_rope(void* input, void* freqs_cos, void* freqs_sin, void* output, 
                 uint32_t head_dim, uint32_t num_heads, uint32_t seq_len);

    void run_gemv_q8_0(void* weights, void* scales, void* input, void* output, 
                      uint32_t K, uint32_t N);

    void run_gemv_q4_0(void* weights, void* scales, void* input, void* output, 
                      uint32_t K, uint32_t N);

private:
    struct Impl;
    Impl* pImpl;
};

} // namespace platform
} // namespace orchard

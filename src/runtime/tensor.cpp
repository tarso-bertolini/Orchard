#include "tensor.h"
#include <numeric>
#include <stdexcept>

namespace orchard {
namespace runtime {

size_t get_dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return 4;
        case DType::Float16: return 2;
        case DType::Int8: return 1;
        default: return 1;
    }
}

Tensor::Tensor(platform::MetalBackend& backend, const std::vector<size_t>& shape, DType dtype)
    : backend_(backend), shape_(shape), dtype_(dtype), buffer_(nullptr) {
    
    size_t bytes = size_bytes();
    buffer_ = backend_.create_buffer(bytes);
    if (!buffer_) {
        throw std::runtime_error("Failed to allocate tensor on GPU");
    }
}

Tensor::~Tensor() {
    if (buffer_) {
        backend_.release_buffer(buffer_);
    }
}

Tensor::Tensor(Tensor&& other) noexcept 
    : backend_(other.backend_), shape_(std::move(other.shape_)), dtype_(other.dtype_), buffer_(other.buffer_) {
    other.buffer_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (buffer_) {
            backend_.release_buffer(buffer_);
        }
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        buffer_ = other.buffer_;
        other.buffer_ = nullptr;
    }
    return *this;
}

size_t Tensor::numel() const {
    if (shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_t>());
}

size_t Tensor::size_bytes() const {
    return numel() * get_dtype_size(dtype_);
}

void Tensor::copy_from_host(const void* data, size_t size) {
    if (size != size_bytes()) {
        throw std::runtime_error("Size mismatch in copy_from_host");
    }
    backend_.copy_to_buffer(buffer_, data, size);
}

void Tensor::copy_to_host(void* data, size_t size) const {
    if (size != size_bytes()) {
        throw std::runtime_error("Size mismatch in copy_to_host");
    }
    backend_.copy_from_buffer(buffer_, data, size);
}

} // namespace runtime
} // namespace orchard

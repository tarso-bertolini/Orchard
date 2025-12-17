#pragma once

#include <vector>
#include <memory>
#include <string>
#include "platform/metal_backend.h"

namespace orchard {
namespace runtime {

enum class DType {
    Float32,
    Float16,
    Int8
};

class Tensor {
public:
    Tensor(platform::MetalBackend& backend, const std::vector<size_t>& shape, DType dtype);
    ~Tensor();

    // No copying, only moving
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    void copy_from_host(const void* data, size_t size);
    void copy_to_host(void* data, size_t size) const;

    void* data() const { return buffer_; }
    const std::vector<size_t>& shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    size_t numel() const;
    size_t size_bytes() const;

private:
    platform::MetalBackend& backend_;
    std::vector<size_t> shape_;
    DType dtype_;
    void* buffer_;
};

} // namespace runtime
} // namespace orchard

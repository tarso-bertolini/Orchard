#pragma once

#include <unordered_map>
#include <string>
#include <memory>
#include "tensor.h"

namespace orchard {
namespace runtime {

class Model {
public:
    Model(platform::MetalBackend& backend);
    
    void add_weight(const std::string& name, Tensor&& tensor);
    Tensor* get_weight(const std::string& name);
    const Tensor* get_weight(const std::string& name) const;

    // In the future, this will load from disk
    // void load(const std::string& path);

private:
    platform::MetalBackend& backend_;
    std::unordered_map<std::string, Tensor> weights_;
};

} // namespace runtime
} // namespace orchard

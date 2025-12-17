#include "model.h"

namespace orchard {
namespace runtime {

Model::Model(platform::MetalBackend& backend) : backend_(backend) {}

void Model::add_weight(const std::string& name, Tensor&& tensor) {
    weights_.emplace(name, std::move(tensor));
}

Tensor* Model::get_weight(const std::string& name) {
    auto it = weights_.find(name);
    if (it != weights_.end()) {
        return &it->second;
    }
    return nullptr;
}

const Tensor* Model::get_weight(const std::string& name) const {
    auto it = weights_.find(name);
    if (it != weights_.end()) {
        return &it->second;
    }
    return nullptr;
}

} // namespace runtime
} // namespace orchard

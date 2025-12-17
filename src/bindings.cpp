#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "platform/metal_backend.h"
#include "runtime/tensor.h"

namespace py = pybind11;
using namespace orchard::platform;
using namespace orchard::runtime;

PYBIND11_MODULE(orchard_core, m) {
    m.doc() = "Orchard Core Metal Runtime";

    // MetalBackend
    py::class_<MetalBackend>(m, "MetalBackend")
        .def(py::init<>())
        .def("initialize", &MetalBackend::initialize)
        .def("is_available", &MetalBackend::is_available)
        .def("get_device_name", &MetalBackend::get_device_name)
        // Expose raw kernels with Tensor wrappers
        .def("run_matmul", [](MetalBackend& self, Tensor& a, Tensor& b, Tensor& c, uint32_t M, uint32_t N, uint32_t K) {
            self.run_matmul(a.data(), b.data(), c.data(), M, N, K);
        })
        .def("run_matmul_simd", [](MetalBackend& self, Tensor& a, Tensor& b, Tensor& c, uint32_t M, uint32_t N, uint32_t K) {
            self.run_matmul_simd(a.data(), b.data(), c.data(), M, N, K);
        })
        .def("run_rmsnorm", [](MetalBackend& self, Tensor& input, Tensor& weight, Tensor& output, float epsilon, uint32_t N, uint32_t count) {
            self.run_rmsnorm(input.data(), weight.data(), output.data(), epsilon, N, count);
        })
        .def("run_rope", [](MetalBackend& self, Tensor& input, Tensor& freqs_cos, Tensor& freqs_sin, Tensor& output, uint32_t head_dim, uint32_t num_heads, uint32_t seq_len) {
            self.run_rope(input.data(), freqs_cos.data(), freqs_sin.data(), output.data(), head_dim, num_heads, seq_len);
        })
        .def("run_gemv_q8_0", [](MetalBackend& self, Tensor& weights, Tensor& scales, Tensor& input, Tensor& output, uint32_t K, uint32_t N) {
            self.run_gemv_q8_0(weights.data(), scales.data(), input.data(), output.data(), K, N);
        })
        .def("run_gemv_q4_0", [](MetalBackend& self, Tensor& weights, Tensor& scales, Tensor& input, Tensor& output, uint32_t K, uint32_t N) {
            self.run_gemv_q4_0(weights.data(), scales.data(), input.data(), output.data(), K, N);
        })
        .def("run_add", [](MetalBackend& self, Tensor& a, Tensor& b, Tensor& c, uint32_t size) {
            self.run_add(a.data(), b.data(), c.data(), size);
        })
        .def("run_mul", [](MetalBackend& self, Tensor& a, Tensor& b, Tensor& c, uint32_t size) {
            self.run_mul(a.data(), b.data(), c.data(), size);
        })
        .def("run_silu", [](MetalBackend& self, Tensor& in, Tensor& out, uint32_t size) {
            self.run_silu(in.data(), out.data(), size);
        })
        .def("run_softmax", [](MetalBackend& self, Tensor& input, Tensor& output, uint32_t rows, uint32_t cols) {
            self.run_softmax(input.data(), output.data(), rows, cols);
        })
        .def("run_embedding", [](MetalBackend& self, Tensor& input_ids, Tensor& weights, Tensor& output, uint32_t num_tokens, uint32_t hidden_dim) {
            self.run_embedding(input_ids.data(), weights.data(), output.data(), num_tokens, hidden_dim);
        });

    // DType
    py::enum_<DType>(m, "DType")
        .value("Float32", DType::Float32)
        .value("Float16", DType::Float16)
        .value("Int8", DType::Int8)
        .export_values();

    // Tensor
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<MetalBackend&, const std::vector<size_t>&, DType>(), py::keep_alive<1, 2>()) // Keep backend alive
        .def("copy_from_host", [](Tensor& self, py::buffer b) {
            py::buffer_info info = b.request();
            self.copy_from_host(info.ptr, info.size * info.itemsize);
        })
        .def("copy_to_host", [](Tensor& self, py::buffer b) {
            py::buffer_info info = b.request();
            size_t expected_size = self.size_bytes();
            size_t buffer_size = info.size * info.itemsize;
            if (buffer_size < expected_size) {
                throw std::runtime_error("Buffer too small");
            }
            self.copy_to_host(info.ptr, expected_size);
        })
        .def("shape", &Tensor::shape)
        .def("numel", &Tensor::numel)
        .def("size_bytes", &Tensor::size_bytes);
}

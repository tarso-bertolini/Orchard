#include "metal_backend.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace orchard {
namespace platform {

struct MetalBackend::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLComputePipelineState> matmulPSO;
    id<MTLComputePipelineState> matmulSimdPSO;
    id<MTLComputePipelineState> rmsNormPSO;
    id<MTLComputePipelineState> ropePSO;
    id<MTLComputePipelineState> gemvInt8PSO;
    id<MTLComputePipelineState> gemvInt4PSO;
    id<MTLComputePipelineState> gemmInt4PSO;
    id<MTLComputePipelineState> addPSO;
    id<MTLComputePipelineState> mulPSO;
    id<MTLComputePipelineState> siluPSO;
    id<MTLComputePipelineState> softmaxPSO;
    id<MTLComputePipelineState> embeddingPSO;

    Impl() {
        device = MTLCreateSystemDefaultDevice();
        if (device) {
            commandQueue = [device newCommandQueue];
        }
    }
};

MetalBackend::MetalBackend() : pImpl(new Impl()) {}

MetalBackend::~MetalBackend() {
    delete pImpl;
}

void MetalBackend::initialize(const std::string& resource_path) {
    if (!pImpl->device) {
        std::cerr << "Failed to create Metal device" << std::endl;
        return;
    }

    // Helper to read file
    auto read_file = [&](const std::string& filename) -> std::string {
        std::string full_path = resource_path + "/" + filename;
        std::ifstream file(full_path);
        if (!file.is_open()) {
            std::cerr << "Failed to open kernel source file: " << full_path << std::endl;
            return "";
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    };

    // Load the library from source
    NSError* error = nil;
    
    std::string source = read_file("matmul.metal");
    std::string source_simd = read_file("matmul_simd.metal");
    std::string source_rms = read_file("rmsnorm.metal");
    std::string source_rope = read_file("rope.metal");
    std::string source_gemv = read_file("gemv_int8.metal");
    std::string source_gemv4 = read_file("gemv_int4.metal");
    std::string source_gemm4 = read_file("gemm_int4.metal");
    std::string source_ew = read_file("elementwise.metal");
    std::string source_sm = read_file("softmax.metal");
    std::string source_emb = read_file("embedding.metal");
    
    // Combine sources
    std::string combined_source = source + "\n" + source_simd + "\n" + source_rms + "\n" + source_rope + "\n" + source_gemv + "\n" + source_gemv4 + "\n" + source_gemm4 + "\n" + source_ew + "\n" + source_sm + "\n" + source_emb;
    
    NSString* librarySource = [NSString stringWithUTF8String:combined_source.c_str()];
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    options.languageVersion = MTLLanguageVersion3_0; // Ensure Metal 3.0 for SIMD
    
    id<MTLLibrary> library = [pImpl->device newLibraryWithSource:librarySource options:options error:&error];
    if (!library) {
        std::cerr << "Failed to compile library: " << [[error localizedDescription] UTF8String] << std::endl;
        return;
    }

    // Load Naive Kernel
    id<MTLFunction> kernelFunction = [library newFunctionWithName:@"matmul_naive"];
    if (!kernelFunction) {
        std::cerr << "Failed to find function 'matmul_naive'" << std::endl;
        return;
    }

    pImpl->matmulPSO = [pImpl->device newComputePipelineStateWithFunction:kernelFunction error:&error];
    if (!pImpl->matmulPSO) {
        std::cerr << "Failed to create PSO: " << [[error localizedDescription] UTF8String] << std::endl;
        return;
    }

    // Load SIMD Kernel
    id<MTLFunction> kernelFunctionSimd = [library newFunctionWithName:@"matmul_simd_fp16"];
    if (!kernelFunctionSimd) {
        std::cerr << "Failed to find function 'matmul_simd_fp16'" << std::endl;
        return;
    }

    pImpl->matmulSimdPSO = [pImpl->device newComputePipelineStateWithFunction:kernelFunctionSimd error:&error];
    if (!pImpl->matmulSimdPSO) {
        std::cerr << "Failed to create SIMD PSO: " << [[error localizedDescription] UTF8String] << std::endl;
        return;
    }

    // Load RMSNorm Kernel
    id<MTLFunction> kernelFunctionRMS = [library newFunctionWithName:@"rms_norm"];
    if (kernelFunctionRMS) {
        pImpl->rmsNormPSO = [pImpl->device newComputePipelineStateWithFunction:kernelFunctionRMS error:&error];
    }

    // Load RoPE Kernel
    id<MTLFunction> kernelFunctionRoPE = [library newFunctionWithName:@"rope"];
    if (kernelFunctionRoPE) {
        pImpl->ropePSO = [pImpl->device newComputePipelineStateWithFunction:kernelFunctionRoPE error:&error];
    }

    // Load GEMV INT8 Kernel
    id<MTLFunction> kernelFunctionGemv = [library newFunctionWithName:@"gemv_q8_0"];
    if (kernelFunctionGemv) {
        pImpl->gemvInt8PSO = [pImpl->device newComputePipelineStateWithFunction:kernelFunctionGemv error:&error];
    }

    // Load GEMV INT4 Kernel
    id<MTLFunction> kernelFunctionGemv4 = [library newFunctionWithName:@"gemv_q4_0"];
    if (kernelFunctionGemv4) {
        pImpl->gemvInt4PSO = [pImpl->device newComputePipelineStateWithFunction:kernelFunctionGemv4 error:&error];
    }

    // Load GEMM INT4 Kernel
    id<MTLFunction> kernelFunctionGemm4 = [library newFunctionWithName:@"gemm_q4_0"];
    if (kernelFunctionGemm4) {
        pImpl->gemmInt4PSO = [pImpl->device newComputePipelineStateWithFunction:kernelFunctionGemm4 error:&error];
    }

    // Load Elementwise Kernels
    id<MTLFunction> kernelFunctionAdd = [library newFunctionWithName:@"add_fp16"];
    if (kernelFunctionAdd) {
        pImpl->addPSO = [pImpl->device newComputePipelineStateWithFunction:kernelFunctionAdd error:&error];
    }
    id<MTLFunction> kernelFunctionMul = [library newFunctionWithName:@"mul_fp16"];
    if (kernelFunctionMul) {
        pImpl->mulPSO = [pImpl->device newComputePipelineStateWithFunction:kernelFunctionMul error:&error];
    }
    id<MTLFunction> kernelFunctionSilu = [library newFunctionWithName:@"silu_fp16"];
    if (kernelFunctionSilu) {
        pImpl->siluPSO = [pImpl->device newComputePipelineStateWithFunction:kernelFunctionSilu error:&error];
    }

    // Load Softmax Kernel
    id<MTLFunction> kernelFunctionSoftmax = [library newFunctionWithName:@"softmax_fp16"];
    if (kernelFunctionSoftmax) {
        pImpl->softmaxPSO = [pImpl->device newComputePipelineStateWithFunction:kernelFunctionSoftmax error:&error];
    }

    // Load Embedding Kernel
    id<MTLFunction> kernelFunctionEmb = [library newFunctionWithName:@"embedding_forward"];
    if (kernelFunctionEmb) {
        pImpl->embeddingPSO = [pImpl->device newComputePipelineStateWithFunction:kernelFunctionEmb error:&error];
    }
}

bool MetalBackend::is_available() const {
    return pImpl->device != nil;
}

std::string MetalBackend::get_device_name() const {
    if (pImpl->device) {
        return std::string([[pImpl->device name] UTF8String]);
    }
    return "Unknown";
}

void* MetalBackend::create_buffer(size_t size) {
    if (!pImpl->device) return nullptr;
    id<MTLBuffer> buffer = [pImpl->device newBufferWithLength:size options:MTLResourceStorageModeShared];
    return (__bridge_retained void*)buffer;
}

void MetalBackend::copy_to_buffer(void* buffer, const void* data, size_t size) {
    id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
    memcpy([mtlBuffer contents], data, size);
}

void MetalBackend::copy_from_buffer(void* buffer, void* data, size_t size) {
    id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
    memcpy(data, [mtlBuffer contents], size);
}

void MetalBackend::release_buffer(void* buffer) {
    if (buffer) {
        CFRelease(buffer);
    }
}

void MetalBackend::run_matmul(void* buffer_a, void* buffer_b, void* buffer_c, 
                             uint32_t M, uint32_t N, uint32_t K) {
    if (!pImpl->commandQueue || !pImpl->matmulPSO) return;

    id<MTLCommandBuffer> commandBuffer = [pImpl->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:pImpl->matmulPSO];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)buffer_a offset:0 atIndex:0];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)buffer_b offset:0 atIndex:1];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)buffer_c offset:0 atIndex:2];
    
    [computeEncoder setBytes:&M length:sizeof(uint32_t) atIndex:3];
    [computeEncoder setBytes:&N length:sizeof(uint32_t) atIndex:4];
    [computeEncoder setBytes:&K length:sizeof(uint32_t) atIndex:5];

    MTLSize gridSize = MTLSizeMake(N, M, 1);
    NSUInteger threadGroupSize = pImpl->matmulPSO.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > N * M) {
        threadGroupSize = N * M;
    }
    
    // Simple 2D threadgroup calculation (naive)
    NSUInteger w = pImpl->matmulPSO.threadExecutionWidth;
    NSUInteger h = pImpl->matmulPSO.maxTotalThreadsPerThreadgroup / w;
    MTLSize threadgroupSize = MTLSizeMake(w, h, 1);

    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [computeEncoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void MetalBackend::run_matmul_simd(void* buffer_a, void* buffer_b, void* buffer_c, 
                                  uint32_t M, uint32_t N, uint32_t K) {
    if (!pImpl->commandQueue || !pImpl->matmulSimdPSO) return;

    id<MTLCommandBuffer> commandBuffer = [pImpl->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:pImpl->matmulSimdPSO];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)buffer_a offset:0 atIndex:0];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)buffer_b offset:0 atIndex:1];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)buffer_c offset:0 atIndex:2];
    
    [computeEncoder setBytes:&M length:sizeof(uint32_t) atIndex:3];
    [computeEncoder setBytes:&N length:sizeof(uint32_t) atIndex:4];
    [computeEncoder setBytes:&K length:sizeof(uint32_t) atIndex:5];

    // For SIMD kernel, we launch 1 thread per 8x8 output tile (simplified)
    // Actually, simdgroup_matrix works per SIMD group (32 threads).
    // So we need 1 SIMD group per 8x8 tile.
    // Threads per threadgroup: 32 (1 simdgroup)
    // Grid size: (N/8) * (M/8) * 32 threads total?
    // No, dispatchThreads is total threads.
    // We want (N/8) groups in X, (M/8) groups in Y.
    // Each group has 32 threads.
    
    // Let's use dispatchThreadgroups instead for explicit control
    NSUInteger threadGroupWidth = 32;
    NSUInteger threadGroupHeight = 1;
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadGroupWidth, threadGroupHeight, 1);
    
    // Number of threadgroups needed
    // Each threadgroup (1 simdgroup) processes one 8x8 tile.
    NSUInteger groupsX = (N + 7) / 8;
    NSUInteger groupsY = (M + 7) / 8;
    MTLSize threadgroupsPerGrid = MTLSizeMake(groupsX, groupsY, 1);

    [computeEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [computeEncoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void MetalBackend::run_rmsnorm(void* input, void* weight, void* output, float epsilon, uint32_t N, uint32_t count) {
    if (!pImpl->commandQueue || !pImpl->rmsNormPSO) return;

    id<MTLCommandBuffer> commandBuffer = [pImpl->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:pImpl->rmsNormPSO];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:0];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)weight offset:0 atIndex:1];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:2];
    [computeEncoder setBytes:&epsilon length:sizeof(float) atIndex:3];
    [computeEncoder setBytes:&N length:sizeof(uint32_t) atIndex:4];

    // Dispatch 1 thread per row (count)
    MTLSize gridSize = MTLSizeMake(count, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(std::min((uint32_t)count, 32u), 1, 1); // Simple
    
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [computeEncoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void MetalBackend::run_rope(void* input, void* freqs_cos, void* freqs_sin, void* output, 
                           uint32_t head_dim, uint32_t num_heads, uint32_t seq_len) {
    if (!pImpl->commandQueue || !pImpl->ropePSO) return;

    id<MTLCommandBuffer> commandBuffer = [pImpl->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:pImpl->ropePSO];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:0];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)freqs_cos offset:0 atIndex:1];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)freqs_sin offset:0 atIndex:2];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:3];
    [computeEncoder setBytes:&head_dim length:sizeof(uint32_t) atIndex:4];
    [computeEncoder setBytes:&num_heads length:sizeof(uint32_t) atIndex:5];
    [computeEncoder setBytes:&seq_len length:sizeof(uint32_t) atIndex:6];

    // Grid: (HeadDim/2, NumHeads * SeqLen)
    // Assuming Batch=1 for now
    uint32_t total_tokens = num_heads * seq_len;
    MTLSize gridSize = MTLSizeMake(head_dim / 2, total_tokens, 1);
    MTLSize threadgroupSize = MTLSizeMake(std::min(head_dim/2, 32u), 1, 1);

    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [computeEncoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void MetalBackend::run_gemv_q8_0(void* weights, void* scales, void* input, void* output, 
                                uint32_t K, uint32_t N) {
    if (!pImpl->commandQueue || !pImpl->gemvInt8PSO) return;

    id<MTLCommandBuffer> commandBuffer = [pImpl->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:pImpl->gemvInt8PSO];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)weights offset:0 atIndex:0];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)scales offset:0 atIndex:1];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:2];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:3];
    [computeEncoder setBytes:&K length:sizeof(uint32_t) atIndex:4];
    [computeEncoder setBytes:&N length:sizeof(uint32_t) atIndex:5];

    // Dispatch 1 thread per output element (N)
    // We can use larger threadgroups for better occupancy
    MTLSize gridSize = MTLSizeMake(N, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(std::min((uint32_t)N, 128u), 1, 1);

    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [computeEncoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void MetalBackend::run_gemv_q4_0(void* weights, void* scales, void* input, void* output, 
                                uint32_t K, uint32_t N) {
    if (!pImpl->commandQueue || !pImpl->gemvInt4PSO) return;

    id<MTLCommandBuffer> commandBuffer = [pImpl->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:pImpl->gemvInt4PSO];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)weights offset:0 atIndex:0];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)scales offset:0 atIndex:1];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:2];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:3];
    [computeEncoder setBytes:&K length:sizeof(uint32_t) atIndex:4];
    [computeEncoder setBytes:&N length:sizeof(uint32_t) atIndex:5];

    // Dispatch 1 thread per output element (N)
    MTLSize gridSize = MTLSizeMake(N, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(std::min((uint32_t)N, 128u), 1, 1);

    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [computeEncoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void MetalBackend::run_gemm_q4_0(void* weights, void* scales, void* input, void* output, 
                                uint32_t K, uint32_t N, uint32_t B) {
    if (!pImpl->commandQueue || !pImpl->gemmInt4PSO) return;

    id<MTLCommandBuffer> commandBuffer = [pImpl->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:pImpl->gemmInt4PSO];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)weights offset:0 atIndex:0];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)scales offset:0 atIndex:1];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:2];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:3];
    [computeEncoder setBytes:&K length:sizeof(uint32_t) atIndex:4];
    [computeEncoder setBytes:&N length:sizeof(uint32_t) atIndex:5];
    [computeEncoder setBytes:&B length:sizeof(uint32_t) atIndex:6];

    // Grid: (N, B)
    MTLSize gridSize = MTLSizeMake(N, B, 1);
    MTLSize threadgroupSize = MTLSizeMake(std::min((uint32_t)N, 128u), 1, 1);

    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [computeEncoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void MetalBackend::run_add(void* a, void* b, void* c, uint32_t size) {
    if (!pImpl->commandQueue || !pImpl->addPSO) return;
    id<MTLCommandBuffer> commandBuffer = [pImpl->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:pImpl->addPSO];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:0 atIndex:0];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)b offset:0 atIndex:1];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)c offset:0 atIndex:2];
    MTLSize gridSize = MTLSizeMake(size, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(std::min(size, 1024u), 1, 1);
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void MetalBackend::run_mul(void* a, void* b, void* c, uint32_t size) {
    if (!pImpl->commandQueue || !pImpl->mulPSO) return;
    id<MTLCommandBuffer> commandBuffer = [pImpl->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:pImpl->mulPSO];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)a offset:0 atIndex:0];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)b offset:0 atIndex:1];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)c offset:0 atIndex:2];
    MTLSize gridSize = MTLSizeMake(size, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(std::min(size, 1024u), 1, 1);
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void MetalBackend::run_silu(void* in, void* out, uint32_t size) {
    if (!pImpl->commandQueue || !pImpl->siluPSO) return;
    id<MTLCommandBuffer> commandBuffer = [pImpl->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:pImpl->siluPSO];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)in offset:0 atIndex:0];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:1];
    MTLSize gridSize = MTLSizeMake(size, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(std::min(size, 1024u), 1, 1);
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void MetalBackend::run_softmax(void* input, void* output, uint32_t rows, uint32_t cols) {
    if (!pImpl->commandQueue || !pImpl->softmaxPSO) return;
    id<MTLCommandBuffer> commandBuffer = [pImpl->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:pImpl->softmaxPSO];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:0];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:1];
    [computeEncoder setBytes:&cols length:sizeof(uint32_t) atIndex:2];
    
    // 1 threadgroup per row
    MTLSize gridSize = MTLSizeMake(1024, rows, 1); // Assuming blockDim=1024
    MTLSize threadgroupSize = MTLSizeMake(1024, 1, 1);
    
    [computeEncoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:threadgroupSize];
    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void MetalBackend::run_embedding(void* input_ids, void* weights, void* output, uint32_t num_tokens, uint32_t hidden_dim) {
    if (!pImpl->commandQueue || !pImpl->embeddingPSO) return;
    id<MTLCommandBuffer> commandBuffer = [pImpl->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:pImpl->embeddingPSO];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)input_ids offset:0 atIndex:0];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)weights offset:0 atIndex:1];
    [computeEncoder setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:2];
    [computeEncoder setBytes:&hidden_dim length:sizeof(uint32_t) atIndex:3];
    
    uint32_t total_elements = num_tokens * hidden_dim;
    MTLSize gridSize = MTLSizeMake(total_elements, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(std::min(total_elements, 1024u), 1, 1);
    
    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

} // namespace platform
} // namespace orchard

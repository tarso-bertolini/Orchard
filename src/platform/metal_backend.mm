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

void MetalBackend::initialize() {
    if (!pImpl->device) {
        std::cerr << "Failed to create Metal device" << std::endl;
        return;
    }

    // Load the library from source
    NSError* error = nil;
    
    // Read source file
    std::ifstream file("src/kernels/matmul.metal");
    if (!file.is_open()) {
        std::cerr << "Failed to open kernel source file: src/kernels/matmul.metal" << std::endl;
        return;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string source = buffer.str();

    // Read SIMD kernel source
    std::ifstream file_simd("src/kernels/matmul_simd.metal");
    if (!file_simd.is_open()) {
        std::cerr << "Failed to open kernel source file: src/kernels/matmul_simd.metal" << std::endl;
        return;
    }
    std::stringstream buffer_simd;
    buffer_simd << file_simd.rdbuf();
    std::string source_simd = buffer_simd.str();

    // Read RMSNorm kernel source
    std::ifstream file_rms("src/kernels/rmsnorm.metal");
    std::stringstream buffer_rms;
    buffer_rms << file_rms.rdbuf();
    std::string source_rms = buffer_rms.str();

    // Read RoPE kernel source
    std::ifstream file_rope("src/kernels/rope.metal");
    std::stringstream buffer_rope;
    buffer_rope << file_rope.rdbuf();
    std::string source_rope = buffer_rope.str();

    // Read GEMV INT8 kernel source
    std::ifstream file_gemv("src/kernels/gemv_int8.metal");
    std::stringstream buffer_gemv;
    buffer_gemv << file_gemv.rdbuf();
    std::string source_gemv = buffer_gemv.str();

    // Read GEMV INT4 kernel source
    std::ifstream file_gemv4("src/kernels/gemv_int4.metal");
    std::stringstream buffer_gemv4;
    buffer_gemv4 << file_gemv4.rdbuf();
    std::string source_gemv4 = buffer_gemv4.str();
    
    // Combine sources
    std::string combined_source = source + "\n" + source_simd + "\n" + source_rms + "\n" + source_rope + "\n" + source_gemv + "\n" + source_gemv4;
    
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

} // namespace platform
} // namespace orchard

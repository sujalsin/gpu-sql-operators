#include "gpu_operators.cuh"

namespace gpu_sql {

cudaError_t MemoryManager::allocateDeviceMemory(void** ptr, size_t size) {
    return cudaMalloc(ptr, size);
}

cudaError_t MemoryManager::freeDeviceMemory(void* ptr) {
    return cudaFree(ptr);
}

cudaError_t MemoryManager::copyHostToDevice(
    void* dst,
    const void* src,
    size_t size,
    cudaStream_t stream
) {
    return cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
}

cudaError_t MemoryManager::copyDeviceToHost(
    void* dst,
    const void* src,
    size_t size,
    cudaStream_t stream
) {
    return cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
}

} // namespace gpu_sql

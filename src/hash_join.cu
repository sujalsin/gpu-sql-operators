#include "gpu_operators.cuh"
#include <cooperative_groups.h>

namespace gpu_sql {

namespace cg = cooperative_groups;

// Hash function for keys
__device__ inline uint32_t hash_function(SqlKey key) {
    key = ((key >> 16) ^ key) * 0x45d9f3b;
    key = ((key >> 16) ^ key) * 0x45d9f3b;
    key = (key >> 16) ^ key;
    return key;
}

// Structure for hash table entry
struct HashEntry {
    SqlKey key;
    SqlValue value;
    bool valid;
};

__global__ void build_hash_table(
    const TableEntry* build_table,
    size_t build_size,
    HashEntry* hash_table,
    size_t hash_table_size
) {
    const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= build_size) return;

    const TableEntry entry = build_table[tid];
    const uint32_t hash = hash_function(entry.key);
    
    // Linear probing
    size_t pos = hash % hash_table_size;
    while (true) {
        bool expected = false;
        if (atomicCAS((unsigned int*)&hash_table[pos].valid, 0, 1) == 0) {
            hash_table[pos].key = entry.key;
            hash_table[pos].value = entry.value;
            break;
        }
        pos = (pos + 1) % hash_table_size;
    }
}

__global__ void probe_hash_table(
    const TableEntry* probe_table,
    size_t probe_size,
    const HashEntry* hash_table,
    size_t hash_table_size,
    TableEntry* output,
    size_t* output_size
) {
    const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= probe_size) return;

    const TableEntry probe_entry = probe_table[tid];
    const uint32_t hash = hash_function(probe_entry.key);
    
    // Linear probing for matching entries
    size_t pos = hash % hash_table_size;
    size_t attempts = 0;
    
    while (attempts < hash_table_size) {
        const HashEntry hash_entry = hash_table[pos];
        if (!hash_entry.valid) break;
        
        if (hash_entry.key == probe_entry.key) {
            const size_t output_idx = atomicAdd(output_size, 1);
            output[output_idx].key = probe_entry.key;
            output[output_idx].value = hash_entry.value;
            break;
        }
        
        pos = (pos + 1) % hash_table_size;
        attempts++;
    }
}

cudaError_t HashJoin::execute(
    const TableEntry* build_table,
    size_t build_size,
    const TableEntry* probe_table,
    size_t probe_size,
    TableEntry* output,
    size_t* output_size,
    cudaStream_t stream
) {
    cudaError_t cuda_status;

    // Calculate hash table size (with load factor consideration)
    const size_t hash_table_size = (build_size * 100) / HASH_TABLE_LOAD_FACTOR;
    
    // Allocate hash table on device
    HashEntry* d_hash_table;
    cuda_status = cudaMalloc(&d_hash_table, hash_table_size * sizeof(HashEntry));
    if (cuda_status != cudaSuccess) return cuda_status;
    
    // Initialize hash table
    cuda_status = cudaMemsetAsync(d_hash_table, 0, hash_table_size * sizeof(HashEntry), stream);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_hash_table);
        return cuda_status;
    }

    // Build phase
    const dim3 build_blocks((build_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    build_hash_table<<<build_blocks, BLOCK_SIZE, 0, stream>>>(
        build_table,
        build_size,
        d_hash_table,
        hash_table_size
    );
    
    // Initialize output size
    size_t* d_output_size;
    cuda_status = cudaMalloc(&d_output_size, sizeof(size_t));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_hash_table);
        return cuda_status;
    }
    
    cuda_status = cudaMemsetAsync(d_output_size, 0, sizeof(size_t), stream);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_hash_table);
        cudaFree(d_output_size);
        return cuda_status;
    }

    // Probe phase
    const dim3 probe_blocks((probe_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    probe_hash_table<<<probe_blocks, BLOCK_SIZE, 0, stream>>>(
        probe_table,
        probe_size,
        d_hash_table,
        hash_table_size,
        output,
        d_output_size
    );
    
    // Copy output size back to host
    cuda_status = cudaMemcpyAsync(output_size, d_output_size, sizeof(size_t),
                                 cudaMemcpyDeviceToHost, stream);
    
    // Cleanup
    cudaFree(d_hash_table);
    cudaFree(d_output_size);
    
    return cuda_status;
}

} // namespace gpu_sql

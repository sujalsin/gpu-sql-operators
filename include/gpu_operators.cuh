#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <cstdint>

namespace gpu_sql {

// Data types for SQL operations
using SqlKey = int64_t;
using SqlValue = int64_t;

struct TableEntry {
    SqlKey key;
    SqlValue value;
};

// Hash Join operator
class HashJoin {
public:
    static cudaError_t execute(
        const TableEntry* build_table,
        size_t build_size,
        const TableEntry* probe_table,
        size_t probe_size,
        TableEntry* output,
        size_t* output_size,
        cudaStream_t stream = nullptr
    );

private:
    static constexpr size_t BLOCK_SIZE = 256;
    static constexpr size_t HASH_TABLE_LOAD_FACTOR = 75;
};

// Aggregation operator
class Aggregation {
public:
    enum class AggType {
        SUM,
        COUNT,
        MIN,
        MAX,
        AVG
    };

    static cudaError_t execute(
        const TableEntry* input,
        size_t input_size,
        TableEntry* output,
        size_t* output_size,
        AggType agg_type,
        cudaStream_t stream = nullptr
    );

private:
    static constexpr size_t BLOCK_SIZE = 256;
    static constexpr size_t SHARED_MEMORY_SIZE = 48 * 1024; // 48KB
};

// Filter operator
class Filter {
public:
    enum class Predicate {
        EQUAL,
        NOT_EQUAL,
        GREATER,
        LESS,
        GREATER_EQUAL,
        LESS_EQUAL
    };

    static cudaError_t execute(
        const TableEntry* input,
        size_t input_size,
        TableEntry* output,
        size_t* output_size,
        Predicate pred_type,
        SqlValue pred_value,
        cudaStream_t stream = nullptr
    );

private:
    static constexpr size_t BLOCK_SIZE = 256;
};

// Memory management utilities
class MemoryManager {
public:
    static cudaError_t allocateDeviceMemory(void** ptr, size_t size);
    static cudaError_t freeDeviceMemory(void* ptr);
    static cudaError_t copyHostToDevice(void* dst, const void* src, size_t size, cudaStream_t stream = nullptr);
    static cudaError_t copyDeviceToHost(void* dst, const void* src, size_t size, cudaStream_t stream = nullptr);
};

} // namespace gpu_sql

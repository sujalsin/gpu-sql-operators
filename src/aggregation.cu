#include "gpu_operators.cuh"
#include <cooperative_groups.h>

namespace gpu_sql {

namespace cg = cooperative_groups;

template<typename T>
__device__ inline void warp_reduce(T& val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
}

__global__ void compute_aggregation(
    const TableEntry* input,
    size_t input_size,
    TableEntry* output,
    size_t* output_size,
    Aggregation::AggType agg_type
) {
    extern __shared__ SqlValue shared_mem[];
    
    const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t lane_id = threadIdx.x % warpSize;
    const size_t warp_id = threadIdx.x / warpSize;
    
    SqlValue thread_value = 0;
    size_t count = 0;

    // Each thread processes multiple elements
    for (size_t i = tid; i < input_size; i += blockDim.x * gridDim.x) {
        SqlValue current = input[i].value;
        
        switch (agg_type) {
            case Aggregation::AggType::SUM:
                thread_value += current;
                break;
            case Aggregation::AggType::COUNT:
                thread_value += 1;
                break;
            case Aggregation::AggType::MIN:
                thread_value = (i == tid) ? current : min(thread_value, current);
                break;
            case Aggregation::AggType::MAX:
                thread_value = (i == tid) ? current : max(thread_value, current);
                break;
            case Aggregation::AggType::AVG:
                thread_value += current;
                count++;
                break;
        }
    }

    // Warp-level reduction
    warp_reduce(thread_value);
    if (agg_type == Aggregation::AggType::AVG) {
        warp_reduce(count);
    }

    // Write warp results to shared memory
    if (lane_id == 0) {
        shared_mem[warp_id] = thread_value;
        if (agg_type == Aggregation::AggType::AVG) {
            shared_mem[warp_id + blockDim.x/warpSize] = count;
        }
    }
    
    __syncthreads();

    // Final block reduction (single warp)
    if (warp_id == 0) {
        thread_value = (lane_id < blockDim.x/warpSize) ? shared_mem[lane_id] : 0;
        if (agg_type == Aggregation::AggType::AVG) {
            count = (lane_id < blockDim.x/warpSize) ? shared_mem[lane_id + blockDim.x/warpSize] : 0;
        }
        
        warp_reduce(thread_value);
        if (agg_type == Aggregation::AggType::AVG) {
            warp_reduce(count);
        }

        if (lane_id == 0) {
            const size_t output_idx = atomicAdd(output_size, 1);
            output[output_idx].key = 0;  // Single group for now
            
            if (agg_type == Aggregation::AggType::AVG) {
                output[output_idx].value = count > 0 ? thread_value / count : 0;
            } else {
                output[output_idx].value = thread_value;
            }
        }
    }
}

cudaError_t Aggregation::execute(
    const TableEntry* input,
    size_t input_size,
    TableEntry* output,
    size_t* output_size,
    AggType agg_type,
    cudaStream_t stream
) {
    cudaError_t cuda_status;

    // Initialize output size
    size_t* d_output_size;
    cuda_status = cudaMalloc(&d_output_size, sizeof(size_t));
    if (cuda_status != cudaSuccess) return cuda_status;
    
    cuda_status = cudaMemsetAsync(d_output_size, 0, sizeof(size_t), stream);
    if (cuda_status != cudaSuccess) {
        cudaFree(d_output_size);
        return cuda_status;
    }

    // Calculate grid dimensions
    const int num_blocks = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int shared_mem_size = (BLOCK_SIZE / warpSize) * sizeof(SqlValue) * 
                               (agg_type == AggType::AVG ? 2 : 1);

    // Launch kernel
    compute_aggregation<<<num_blocks, BLOCK_SIZE, shared_mem_size, stream>>>(
        input,
        input_size,
        output,
        d_output_size,
        agg_type
    );

    // Copy output size back to host
    cuda_status = cudaMemcpyAsync(output_size, d_output_size, sizeof(size_t),
                                 cudaMemcpyDeviceToHost, stream);
    
    cudaFree(d_output_size);
    return cuda_status;
}

} // namespace gpu_sql

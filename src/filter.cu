#include "gpu_operators.cuh"
#include <cooperative_groups.h>

namespace gpu_sql {

namespace cg = cooperative_groups;

__device__ inline bool evaluate_predicate(
    SqlValue value,
    Filter::Predicate pred_type,
    SqlValue pred_value
) {
    switch (pred_type) {
        case Filter::Predicate::EQUAL:
            return value == pred_value;
        case Filter::Predicate::NOT_EQUAL:
            return value != pred_value;
        case Filter::Predicate::GREATER:
            return value > pred_value;
        case Filter::Predicate::LESS:
            return value < pred_value;
        case Filter::Predicate::GREATER_EQUAL:
            return value >= pred_value;
        case Filter::Predicate::LESS_EQUAL:
            return value <= pred_value;
        default:
            return false;
    }
}

__global__ void filter_kernel(
    const TableEntry* input,
    size_t input_size,
    TableEntry* output,
    size_t* output_size,
    Filter::Predicate pred_type,
    SqlValue pred_value
) {
    const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (tid < input_size) {
        const TableEntry entry = input[tid];
        if (evaluate_predicate(entry.value, pred_type, pred_value)) {
            const size_t output_idx = atomicAdd(output_size, 1);
            output[output_idx] = entry;
        }
    }
}

__global__ void filter_with_prefix_sum(
    const TableEntry* input,
    size_t input_size,
    TableEntry* output,
    size_t* output_size,
    Filter::Predicate pred_type,
    SqlValue pred_value,
    int* prefix_sum
) {
    extern __shared__ int shared_flags[];
    
    const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t local_tid = threadIdx.x;
    
    // Step 1: Evaluate predicate and store result
    int flag = 0;
    if (tid < input_size) {
        flag = evaluate_predicate(input[tid].value, pred_type, pred_value) ? 1 : 0;
    }
    shared_flags[local_tid] = flag;
    
    __syncthreads();
    
    // Step 2: Compute prefix sum within block
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int temp = 0;
        if (local_tid >= stride) {
            temp = shared_flags[local_tid - stride];
        }
        __syncthreads();
        if (local_tid >= stride) {
            shared_flags[local_tid] += temp;
        }
        __syncthreads();
    }
    
    // Step 3: Store block sum and update global prefix sum
    if (local_tid == blockDim.x - 1) {
        if (blockIdx.x > 0) {
            prefix_sum[blockIdx.x] = shared_flags[local_tid];
        }
        if (blockIdx.x == gridDim.x - 1) {
            *output_size = shared_flags[local_tid];
        }
    }
    
    __syncthreads();
    
    // Step 4: Write output
    if (tid < input_size && flag) {
        const size_t output_idx = (blockIdx.x > 0 ? prefix_sum[blockIdx.x - 1] : 0) + 
                                 shared_flags[local_tid] - 1;
        output[output_idx] = input[tid];
    }
}

cudaError_t Filter::execute(
    const TableEntry* input,
    size_t input_size,
    TableEntry* output,
    size_t* output_size,
    Predicate pred_type,
    SqlValue pred_value,
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

    const dim3 block_dim(BLOCK_SIZE);
    const dim3 grid_dim((input_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // For small inputs, use simple atomic approach
    if (input_size <= 1000000) {
        filter_kernel<<<grid_dim, block_dim, 0, stream>>>(
            input,
            input_size,
            output,
            d_output_size,
            pred_type,
            pred_value
        );
    } else {
        // For larger inputs, use prefix sum approach for better performance
        int* d_prefix_sum;
        cuda_status = cudaMalloc(&d_prefix_sum, grid_dim.x * sizeof(int));
        if (cuda_status != cudaSuccess) {
            cudaFree(d_output_size);
            return cuda_status;
        }

        filter_with_prefix_sum<<<grid_dim, block_dim, BLOCK_SIZE * sizeof(int), stream>>>(
            input,
            input_size,
            output,
            d_output_size,
            pred_type,
            pred_value,
            d_prefix_sum
        );

        cudaFree(d_prefix_sum);
    }

    // Copy output size back to host
    cuda_status = cudaMemcpyAsync(output_size, d_output_size, sizeof(size_t),
                                 cudaMemcpyDeviceToHost, stream);
    
    cudaFree(d_output_size);
    return cuda_status;
}

} // namespace gpu_sql

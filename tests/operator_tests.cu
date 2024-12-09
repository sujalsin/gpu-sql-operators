#include "gpu_operators.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <cassert>

using namespace gpu_sql;

// Test utilities
void generateRandomData(std::vector<TableEntry>& data, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<SqlKey> key_dist(1, 1000000);
    std::uniform_int_distribution<SqlValue> value_dist(1, 1000);

    data.resize(size);
    for (size_t i = 0; i < size; ++i) {
        data[i].key = key_dist(gen);
        data[i].value = value_dist(gen);
    }
}

// Test hash join
void testHashJoin() {
    std::cout << "Testing Hash Join..." << std::endl;

    const size_t build_size = 1000000;
    const size_t probe_size = 500000;

    std::vector<TableEntry> build_table, probe_table;
    generateRandomData(build_table, build_size);
    generateRandomData(probe_table, probe_size);

    // Allocate device memory
    TableEntry *d_build_table, *d_probe_table, *d_output;
    cudaMalloc(&d_build_table, build_size * sizeof(TableEntry));
    cudaMalloc(&d_probe_table, probe_size * sizeof(TableEntry));
    cudaMalloc(&d_output, probe_size * sizeof(TableEntry));

    // Copy data to device
    cudaMemcpy(d_build_table, build_table.data(), build_size * sizeof(TableEntry),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_probe_table, probe_table.data(), probe_size * sizeof(TableEntry),
               cudaMemcpyHostToDevice);

    // Execute join
    size_t output_size;
    cudaError_t status = HashJoin::execute(
        d_build_table,
        build_size,
        d_probe_table,
        probe_size,
        d_output,
        &output_size
    );

    assert(status == cudaSuccess);
    std::cout << "Hash Join completed. Output size: " << output_size << std::endl;

    // Cleanup
    cudaFree(d_build_table);
    cudaFree(d_probe_table);
    cudaFree(d_output);
}

// Test aggregation
void testAggregation() {
    std::cout << "Testing Aggregation..." << std::endl;

    const size_t input_size = 1000000;
    std::vector<TableEntry> input;
    generateRandomData(input, input_size);

    // Allocate device memory
    TableEntry *d_input, *d_output;
    cudaMalloc(&d_input, input_size * sizeof(TableEntry));
    cudaMalloc(&d_output, sizeof(TableEntry));

    // Copy data to device
    cudaMemcpy(d_input, input.data(), input_size * sizeof(TableEntry),
               cudaMemcpyHostToDevice);

    // Test each aggregation type
    const Aggregation::AggType agg_types[] = {
        Aggregation::AggType::SUM,
        Aggregation::AggType::COUNT,
        Aggregation::AggType::MIN,
        Aggregation::AggType::MAX,
        Aggregation::AggType::AVG
    };

    for (auto agg_type : agg_types) {
        size_t output_size;
        cudaError_t status = Aggregation::execute(
            d_input,
            input_size,
            d_output,
            &output_size,
            agg_type
        );

        assert(status == cudaSuccess);
        
        TableEntry result;
        cudaMemcpy(&result, d_output, sizeof(TableEntry),
                   cudaMemcpyDeviceToHost);

        std::cout << "Aggregation type " << static_cast<int>(agg_type)
                  << " result: " << result.value << std::endl;
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
}

// Test filter
void testFilter() {
    std::cout << "Testing Filter..." << std::endl;

    const size_t input_size = 1000000;
    std::vector<TableEntry> input;
    generateRandomData(input, input_size);

    // Allocate device memory
    TableEntry *d_input, *d_output;
    cudaMalloc(&d_input, input_size * sizeof(TableEntry));
    cudaMalloc(&d_output, input_size * sizeof(TableEntry));

    // Copy data to device
    cudaMemcpy(d_input, input.data(), input_size * sizeof(TableEntry),
               cudaMemcpyHostToDevice);

    // Test each predicate type
    const Filter::Predicate pred_types[] = {
        Filter::Predicate::EQUAL,
        Filter::Predicate::GREATER,
        Filter::Predicate::LESS
    };

    const SqlValue pred_value = 500;  // Test value in middle of range

    for (auto pred_type : pred_types) {
        size_t output_size;
        cudaError_t status = Filter::execute(
            d_input,
            input_size,
            d_output,
            &output_size,
            pred_type,
            pred_value
        );

        assert(status == cudaSuccess);
        std::cout << "Filter predicate " << static_cast<int>(pred_type)
                  << " output size: " << output_size << std::endl;
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    // Initialize CUDA
    cudaFree(0);  // Force CUDA context initialization

    try {
        testHashJoin();
        testAggregation();
        testFilter();
        
        std::cout << "All tests completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

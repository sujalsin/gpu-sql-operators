cmake_minimum_required(VERSION 3.18)
project(gpu_sql_operators CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDA REQUIRED)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add source files
add_library(gpu_sql_operators STATIC
    src/hash_join.cu
    src/aggregation.cu
    src/filter.cu
    src/utils/memory_manager.cu
)

# Set compile options
target_compile_options(gpu_sql_operators PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -lineinfo
    >
)

# Create test executable
add_executable(operator_tests
    tests/operator_tests.cu
)

target_link_libraries(operator_tests
    gpu_sql_operators
    ${CUDA_LIBRARIES}
)

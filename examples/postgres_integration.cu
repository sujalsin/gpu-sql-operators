#include "gpu_operators.cuh"
#include "postgres.h"
#include "fmgr.h"
#include "executor/executor.h"
#include "utils/array.h"
#include "utils/builtins.h"

PG_MODULE_MAGIC;

using namespace gpu_sql;

// Utility function to convert PostgreSQL tuple to TableEntry
static void tuple_to_table_entry(HeapTuple tuple, TupleDesc tupdesc, 
                               TableEntry* entry) {
    bool isnull;
    Datum key = heap_getattr(tuple, 1, tupdesc, &isnull);
    Datum value = heap_getattr(tuple, 2, tupdesc, &isnull);
    
    entry->key = DatumGetInt64(key);
    entry->value = DatumGetInt64(value);
}

// Example: GPU-accelerated hash join for PostgreSQL
extern "C" {
PG_FUNCTION_INFO_V1(gpu_hash_join);

Datum gpu_hash_join(PG_FUNCTION_ARGS) {
    // Get input relations
    Relation build_rel = (Relation) PG_GETARG_POINTER(0);
    Relation probe_rel = (Relation) PG_GETARG_POINTER(1);
    
    // Scan input relations
    TableScanDesc build_scan = table_beginscan(build_rel, SnapshotAny, 0, NULL);
    TableScanDesc probe_scan = table_beginscan(probe_rel, SnapshotAny, 0, NULL);
    
    // Count tuples
    size_t build_size = 0;
    size_t probe_size = 0;
    HeapTuple tuple;
    
    while ((tuple = heap_getnext(build_scan, ForwardScanDirection)) != NULL)
        build_size++;
    while ((tuple = heap_getnext(probe_scan, ForwardScanDirection)) != NULL)
        probe_size++;
    
    // Allocate host memory
    std::vector<TableEntry> build_table(build_size);
    std::vector<TableEntry> probe_table(probe_size);
    
    // Read tuples into host arrays
    table_rescan(build_scan, NULL);
    table_rescan(probe_scan, NULL);
    
    size_t i = 0;
    while ((tuple = heap_getnext(build_scan, ForwardScanDirection)) != NULL)
        tuple_to_table_entry(tuple, build_rel->rd_att, &build_table[i++]);
    
    i = 0;
    while ((tuple = heap_getnext(probe_scan, ForwardScanDirection)) != NULL)
        tuple_to_table_entry(tuple, probe_rel->rd_att, &probe_table[i++]);
    
    // Allocate device memory
    TableEntry *d_build_table, *d_probe_table, *d_output;
    cudaMalloc(&d_build_table, build_size * sizeof(TableEntry));
    cudaMalloc(&d_probe_table, probe_size * sizeof(TableEntry));
    cudaMalloc(&d_output, probe_size * sizeof(TableEntry));
    
    // Copy to device
    cudaMemcpy(d_build_table, build_table.data(), 
               build_size * sizeof(TableEntry), cudaMemcpyHostToDevice);
    cudaMemcpy(d_probe_table, probe_table.data(),
               probe_size * sizeof(TableEntry), cudaMemcpyHostToDevice);
    
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
    
    if (status != cudaSuccess) {
        ereport(ERROR,
                (errcode(ERRCODE_GP_INTERNAL_ERROR),
                 errmsg("GPU hash join failed: %s", 
                        cudaGetErrorString(status))));
    }
    
    // Allocate result array
    std::vector<TableEntry> output(output_size);
    
    // Copy results back
    cudaMemcpy(output.data(), d_output,
               output_size * sizeof(TableEntry), cudaMemcpyDeviceToHost);
    
    // Clean up
    cudaFree(d_build_table);
    cudaFree(d_probe_table);
    cudaFree(d_output);
    
    table_endscan(build_scan);
    table_endscan(probe_scan);
    
    // Create and return result array
    ArrayType* result = construct_array((Datum*)output.data(),
                                      output_size,
                                      INT8OID,
                                      sizeof(TableEntry),
                                      true,
                                      'd');
    
    PG_RETURN_ARRAYTYPE_P(result);
}
}

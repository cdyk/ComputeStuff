#pragma once
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>


namespace ComputeStuff {

  namespace MC {

    enum struct FieldFormat : uint32_t
    {
      UInt8,
      UInt16,
      Float
    };

    struct Tables
    {
      const uint8_t* index_count = nullptr;
      const uint8_t* index_table = nullptr;
    };

    struct Context
    {
      const Tables* tables = nullptr;
      uint8_t* index_cases_d = nullptr;           // 800 * chunk count
      uint4* pyramid = nullptr;           // baselevel is full grid
      uint32_t* sidebands[2] = {
        nullptr,                          // baselevel / 5
        nullptr                           // baselevel / 5*5
      };
      uint32_t* sum_h = nullptr;
      uint32_t* sum_d = nullptr;

      uint3 cells;
      uint3 chunks;
      uint32_t chunk_total = 0;
      uint32_t levels = 0;
      uint32_t level_sizes[32];   // nunber of uvec4's
      uint32_t level_offsets[32]; // offsets in uvec4's
      uint32_t total_size;
      cudaEvent_t countWritten;
    };

    void getCounts(Context* ctx, uint32_t* vertices, uint32_t* indices);

    Tables* createTables(cudaStream_t stream);

    Context* createContext(const Tables* tables,
                           uint3 cells,
                           cudaStream_t stream);

    void buildPN(Context* ctx,
                 void* output_buffer,
                 size_t output_buffer_size,
                 size_t field_row_stride,
                 size_t field_slice_stride,
                 uint3 field_offset,
                 uint3 field_size,
                 const float* field,
                 const float threshold,
                 cudaStream_t stream,
                 bool skipPyramid,
                 bool alwaysExtract);

    void destroyContext(Context* ctx);

  }
}
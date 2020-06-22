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
      uint32_t* buffer = nullptr;
      uint32_t* sidebands[2] = { nullptr, nullptr };
      uint3 cells;
      uint3 chunks;
      uint32_t levels = 0;
      uint32_t level_sizes[32];
      uint32_t level_offsets[32];
      uint32_t total_size;
    };

    Tables* createTables(cudaStream_t stream);

    Context* createContext(const Tables* tables,
                           uint3 cells,
                           cudaStream_t stream);

    uint32_t buildP3(Context* ctx,
                     uint3 offset,
                     uint3 field_size,
                     const float* field,
                     const float threshold,
                     cudaStream_t stream);

    void destroyContext(Context* ctx);

  }
}
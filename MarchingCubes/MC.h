#pragma once
// This file is part of ComputeStuff copyright (C) 2020 Christopher Dyken.
// Released under the MIT license, please see LICENSE file for details.

#include <vector>
#include <cstdint>
#include <cuda_runtime.h>


namespace ComputeStuff {

  namespace MC {

    struct Tables
    {
      const uint8_t* index_count = nullptr;
      const uint8_t* index_table = nullptr;
    };

    struct Context
    {
      const Tables* tables = nullptr;

      uint8_t*      index_cases_d = nullptr;           // 800 * chunk count
      uint4*        index_pyramid = nullptr;           // baselevel is full grid
      uint32_t*     index_sidebands[2] = { nullptr, nullptr };

      uint8_t*      vertex_cases_d = nullptr;           // 800 * chunk count
      uint4*        vertex_pyramid = nullptr;           // baselevel is full grid
      uint32_t*     vertex_sidebands[2] = { nullptr, nullptr };

      uint32_t*     sum_h = nullptr;
      uint32_t*     sum_d = nullptr;
      uint3         grid_size;
      uint3         chunks;
      uint32_t      chunk_total = 0;
      uint32_t      levels = 0;
      uint32_t      level_sizes[16];   // nunber of uvec4's
      uint32_t      level_offsets[16]; // offsets in uvec4's
      uint32_t      total_size;

      bool          indexed = false;
    };

    void getCounts(Context* ctx, uint32_t* vertices, uint32_t* indices, cudaStream_t stream);

    Tables* createTables(cudaStream_t stream);

    Context* createContext(const Tables* tables,
                           uint3 cells,
                           bool indexed,
                           cudaStream_t stream);

    void freeContext(Context* ctx, cudaStream_t stream);

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
#pragma once
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

    struct Tables;

    struct HistoPyramid;

    Tables* createTables(cudaStream_t streaam);

    HistoPyramid* createHistoPyramid(cudaStream_t stream, Tables* tables, uint32_t nx, uint32_t ny, uint32_t nz);

    void buildHistoPyramid(cudaStream_t stream, HistoPyramid* hp, float iso);

  }
}
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace ComputeStuff {
  namespace Scan {

    template<typename T>
    size_t scratchByteSize(T N);

    size_t levels(size_t N);

    void calcOffsets(uint32_t* offsets_d,
                     uint32_t* sum_d,
                     uint32_t* scratch_d,
                     const uint32_t* counts_d,
                     size_t N,
                     cudaStream_t stream = 0);

    void calcOffsets(uint32_t* offsets_d,
                     uint32_t* scratch_d,
                     const uint32_t* counts_d,
                     size_t N,
                     cudaStream_t stream = 0);

  }
}
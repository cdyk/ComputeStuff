#pragma once
// This file is part of ComputeStuff copyright (C) 2017 Christopher Dyken.
// Released under the MIT license, please see LICENSE file for details.

#include <cstdint>
#include <cuda_runtime.h>

namespace ComputeStuff {
  namespace Scan {

    /// Get the size of the scratch buffer (in bytes) for the scan functions.
    ///
    /// \param N  Problem size.
    /// \returns  Number of bytes for the scratch buffer.
    uint32_t scratchByteSize(uint32_t N);

    void exclusiveScan(uint32_t* output_d,
                       uint32_t* scratch_d,
                       const uint32_t* input_d,
                       uint32_t N,
                       cudaStream_t stream = (cudaStream_t)0);

    void inclusiveScan(uint32_t* output_d,
                       uint32_t* scratch_d,
                       const uint32_t* input_d,
                       uint32_t N,
                       cudaStream_t stream = (cudaStream_t)0);

    void calcOffsets(uint32_t* output_d,
                     uint32_t* sum_d,
                     uint32_t* scratch_d,
                     const uint32_t* input_d,
                     uint32_t N,
                     cudaStream_t stream = 0);

    void calcOffsets(uint32_t* output_d,
                     uint32_t* scratch_d,
                     const uint32_t* input_d,
                     uint32_t N,
                     cudaStream_t stream = 0);

  }
}
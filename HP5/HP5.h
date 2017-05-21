#pragma once
// This file is part of ComputeStuff copyright (C) 2017 Christopher Dyken.
// Released under the MIT license, please see LICENSE file for details.

#include <cstdint>
#include <cuda_runtime.h>

namespace ComputeStuff {
  namespace HP5 {

    /// Get the size of the scratch buffer (in bytes) for the HP5 functions.
    ///
    /// \param N  Problem size.
    /// \returns  Number of bytes for the scratch buffer.
    size_t scratchByteSize(uint32_t N);

    void compact(uint32_t* out_d,
                 uint32_t* sum_d,
                 uint32_t* scratch_d,
                 const uint32_t* in_d,
                 uint32_t N,
                 cudaStream_t stream = (cudaStream_t)0);
  }
}

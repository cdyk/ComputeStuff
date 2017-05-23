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

    /// Extract indices where the corresponding input has a non-zero value (a.k.a. subset, stream-compact and copy-if)
    ///
    /// If input is
    ///    [a, 0, 0, b, 0, c, 0, d, 0, 0, e],
    /// where a,b,c,d,e are non-zero values, then the result is
    ///    [0, 3, 5, 7, 10],
    /// that is, the indices of a, b, c, d, and e in the input array, and sum is 5, the number of
    /// non-zero entries in input (and thus the number of entries in the output array).
    ///
    /// Also known as subset, stream-compact, and copy-if.
    ///
    /// \note Supports in-place operation, i.e. output_d == input_d.
    /// 
    /// \param out_d Device memory pointer to where the result of maximally N elements are stored.
    /// \param sum_d Device memory pointer to where the total sum.
    /// \param scratch_d Device memory pointer to scratch area, size given by \ref scratchByteSize.
    /// \param in_d Device memory pointer to the N input elements.
    /// \param N The number of input elements.
    /// \param stream The CUDA stream to use.
    void compact(uint32_t* out_d,
                 uint32_t* sum_d,
                 uint32_t* scratch_d,
                 const uint32_t* in_d,
                 uint32_t N,
                 cudaStream_t stream = (cudaStream_t)0);
  }
}

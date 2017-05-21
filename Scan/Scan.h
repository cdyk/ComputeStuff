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

    /// Calculate the exclusive prefix-sum (scan)
    ///
    /// If input is
    ///   [a, b, c, d],
    /// then the result is
    ///   [0, a, a+b, a+b+c].
    ///
    /// \note Supports in-place operation, i.e. output_d == input_d.
    /// 
    /// \param output_d Device memory pointer to where the result of N elements is stored.
    /// \param scratch_d Device memory pointer to scratch area, size given by \ref scratchByteSize.
    /// \param input_d Device memory pointer to the N input elements.
    /// \param N The number of input elements.
    /// \param stream The CUDA stream to use.
    void exclusiveScan(uint32_t* output_d,
                       uint32_t* scratch_d,
                       const uint32_t* input_d,
                       uint32_t N,
                       cudaStream_t stream = (cudaStream_t)0);

    /// Calculate the inclusive prefix-sum (scan)
    ///
    /// If input is
    ///   [a, b, c, d],
    /// then the result is
    ///   [a, a+b, a+b+c, a+b+c+d].
    ///
    /// \note Supports in-place operation, i.e. output_d == input_d.
    /// 
    /// \param output_d Device memory pointer to where the result of N elements is stored.
    /// \param scratch_d Device memory pointer to scratch area, size given by \ref scratchByteSize.
    /// \param input_d Device memory pointer to the N input elements.
    /// \param N The number of input elements.
    /// \param stream The CUDA stream to use.
    void inclusiveScan(uint32_t* output_d,
                       uint32_t* scratch_d,
                       const uint32_t* input_d,
                       uint32_t N,
                       cudaStream_t stream = (cudaStream_t)0);

    /// Calculate an offset-table (exclusive prefix-sum with total concatenated on end).
    ///
    /// If input is
    ///   [a, b, c, d],
    /// then the result is
    ///   [0, a, a+b, a+b+c, a+b+c+d],
    /// that is, the output array is one element larger than the input.
    ///
    /// Thus, output[i] gives the offset for i, and output[i+1]-output[i] gives input[i].
    ///
    /// \note Supports in-place operation, i.e. output_d == input_d.
    /// 
    /// \param output_d Device memory pointer to where the result of (N+1) elements is stored.
    /// \param scratch_d Device memory pointer to scratch area, size given by \ref scratchByteSize.
    /// \param input_d Device memory pointer to the N input elements.
    /// \param N The number of input elements.
    /// \param stream The CUDA stream to use.
    void calcOffsets(uint32_t* output_d,
                     uint32_t* scratch_d,
                     const uint32_t* input_d,
                     uint32_t N,
                     cudaStream_t stream = 0);

    /// Calculate an offset-table (exclusive prefix-sum with total concatenated on end), and write total sum.
    ///
    /// If input is
    ///   [a, b, c, d],
    /// then the result is
    ///   [0, a, a+b, a+b+c, a+b+c+d],
    /// that is, the output array is one element larger than the input.
    ///
    /// Thus, output[i] gives the offset for i, and output[i+1]-output[i] gives input[i].
    ///
    /// In addition, the total (i.e. last element of output) is also written to given device pointer. A use
    /// for this is when a subsequent grid depends on the total sum, and this total sum is zero-copied to
    /// host memory and is available as soon as the kernel is finished. See test application for an example.
    ///
    /// \note Supports in-place operation, i.e. output_d == input_d.
    /// 
    /// \param output_d Device memory pointer to where the result of (N+1) elements are stored.
    /// \param sum_d Device memory pointer to where the total sum.
    /// \param scratch_d Device memory pointer to scratch area, size given by \ref scratchByteSize.
    /// \param input_d Device memory pointer to the N input elements.
    /// \param N The number of input elements.
    /// \param stream The CUDA stream to use.
    void calcOffsets(uint32_t* output_d,
                     uint32_t* sum_d,
                     uint32_t* scratch_d,
                     const uint32_t* input_d,
                     uint32_t N,
                     cudaStream_t stream = 0);

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
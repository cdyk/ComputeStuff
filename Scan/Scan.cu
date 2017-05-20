// This file is part of ComputeStuff copyright (C) 2017 Christopher Dyken.
// Released under the MIT license, please see LICENSE file for details.

#include <limits>
#include <vector>
#include <algorithm>
#include "Scan.h"

#define WARPSIZE 32
#define SCAN_WARPS 4

namespace {

  template<bool subset>
  __global__
  __launch_bounds__(SCAN_WARPS * WARPSIZE)
  void
  reduce(uint32_t* output,
         const uint32_t* input,
         uint32_t N)
  {
    const uint32_t lane = threadIdx.x % WARPSIZE;
    const uint32_t warp = threadIdx.x / WARPSIZE;
    const uint32_t threadOffset = 4 * (SCAN_WARPS * WARPSIZE * blockIdx.x + threadIdx.x);

    // Fetch
    uint4 a;
    if (threadOffset + 3 < N) {
      a = *reinterpret_cast<const uint4*>(input + threadOffset);
    }
    else if (N <= threadOffset) {
      a = make_uint4(0, 0, 0, 0);
    }
    else {
      a.x = input[threadOffset];
      a.y = threadOffset + 1 < N ? input[threadOffset + 1] : 0;
      a.z = threadOffset + 2 < N ? input[threadOffset + 2] : 0;
      a.w = 0;
    }
    if (subset) {
      a.x = a.x != 0 ? 1 : 0;
      a.y = a.y != 0 ? 1 : 0;
      a.z = a.z != 0 ? 1 : 0;
      a.w = a.w != 0 ? 1 : 0;
    }
    uint32_t s = a.x + a.y + a.z + a.w;

    // Per-warp reduce
    #pragma unroll
    for (uint32_t i = 1; i < WARPSIZE; i *= 2) {
      uint32_t t = __shfl_up(s, i);
      if (i <= lane) {
        s += t;
      }
    }

    __shared__ uint32_t warpSum[SCAN_WARPS];
    if (lane == (WARPSIZE - 1)) {
      warpSum[warp] = s;
    }

    __syncthreads();

    // Aggregate warp sums and write total.
    if (threadIdx.x == 0) {
      auto a = warpSum[0];

      #pragma unroll
      for (uint32_t i = 1; i < SCAN_WARPS; i++) {
        a += warpSum[i];
      }

      output[blockIdx.x] = a;
    }
  }

  template<bool subset, bool inclusive, bool writeSum0, bool writeSum1, bool readOffset>
  __global__
  __launch_bounds__(SCAN_WARPS * WARPSIZE)
  void
  scan(uint32_t* output,
       uint32_t* sum0,
       uint32_t* sum1,
       const uint32_t* input,
       const uint32_t* offset,
       uint32_t N)
  {
    const uint32_t lane = threadIdx.x % WARPSIZE;
    const uint32_t warp = threadIdx.x / WARPSIZE;
    const uint32_t threadOffset = 4 * (SCAN_WARPS * WARPSIZE * blockIdx.x + threadIdx.x);

    // Fetch
    uint4 a;
    if (threadOffset + 3 < N) {
      a = *reinterpret_cast<const uint4*>(input + threadOffset);
    }
    else if (N <= threadOffset) {
      a = make_uint4(0, 0, 0, 0);
    }
    else {
      a.x = input[threadOffset];
      a.y = threadOffset + 1 < N ? input[threadOffset + 1] : 0;
      a.z = threadOffset + 2 < N ? input[threadOffset + 2] : 0;
      a.w = 0;
    }
    if (subset) {
      a.x = a.x != 0 ? 1 : 0;
      a.y = a.y != 0 ? 1 : 0;
      a.z = a.z != 0 ? 1 : 0;
      a.w = a.w != 0 ? 1 : 0;
    }
    uint32_t s = a.x + a.y + a.z + a.w;
    uint32_t q = s;

    // Per-warp reduce
    #pragma unroll
    for (uint32_t i = 1; i < WARPSIZE; i *= 2) {
      uint32_t t = __shfl_up(s, i);
      if (i <= lane) {
        s += t;
      }
    }

    __shared__ uint32_t warpSum[SCAN_WARPS];
    if (lane == (WARPSIZE - 1)) {
      warpSum[warp] = s;
    }

    __syncthreads();

    #pragma unroll
    for (uint32_t w = 0; w < SCAN_WARPS - 1; w++) {
      if (w < warp) s += warpSum[w];
    }

    if (threadIdx.x == (SCAN_WARPS*WARPSIZE - 1)) {
      if (writeSum0) *sum0 = s;
      if (writeSum1) *sum1 = s;
    }

    s -= q;

    if (readOffset) {
      s += offset[blockIdx.x];
    }

    // Store
    uint4 r;
    if (inclusive) {
      r = make_uint4(s + a.x,
                     s + a.x + a.y,
                     s + a.x + a.y + a.z,
                     s + a.x + a.y + a.z + a.w);
    }
    else {
      r = make_uint4(s,
                     s + a.x,
                     s + a.x + a.y,
                     s + a.x + a.y + a.z);
    }

    if (threadOffset + 3 < N) {
      *reinterpret_cast<uint4*>(output + threadOffset) = r;
    }
    else if (threadOffset < N) {
      output[threadOffset + 0] = r.x;
      if (threadOffset + 1 < N) output[threadOffset + 1] = r.y;
      if (threadOffset + 2 < N) output[threadOffset + 2] = r.z;
    }
  }

  void calcLevelSizes(std::vector<uint32_t>& levels, uint32_t N)
  {
    // Amount of reduction per level.
    uint32_t R = 4 * SCAN_WARPS*WARPSIZE;

    levels.clear();
    while (1 < N) {
      N = (N + R - 1) / R;
      levels.push_back(N);
    }

    // Remove to-one reduction that is always present.
    if (!levels.empty()) levels.pop_back();
  }


  // Helper func to run all scan variants.
  template<bool subset, bool inclusive, bool extraElement, bool writeSum>
  void runKernels(uint32_t* output_d,
                  uint32_t* sum_d,
                  uint32_t* scratch_d,
                  const uint32_t* input_d,
                  uint32_t N,
                  cudaStream_t stream)
  {
    if (N == 0) return;

    std::vector<uint32_t> levels;
    calcLevelSizes(levels, N);

    std::vector<uint32_t> offsets;
    offsets.push_back(0);
    uint32_t alignment = 128 / sizeof(uint32_t);
    for (size_t i = 0; i < levels.size(); i++) {
      auto levelSize = (levels[i] + alignment - 1) & ~(alignment - 1);
      offsets.push_back(offsets[i] + levelSize);
    }

    uint32_t blockSize = SCAN_WARPS * WARPSIZE;

    if (levels.empty()) {

      ::scan<subset, inclusive, extraElement, writeSum, false><<<1, blockSize, 0, stream >>>(output_d,
                                                                                             output_d + N,
                                                                                             sum_d,
                                                                                             input_d,
                                                                                             nullptr,
                                                                                             N);

    }
    else {

      uint32_t L = static_cast<uint32_t>(levels.size());

      // From input, populate level 0
      ::reduce<subset><<<levels[0], blockSize, 0, stream >>>(scratch_d + offsets[0],
                                                     input_d,
                                                     N);

      // From level i-1, populate level i, up to including L-1.
      for (uint32_t i = 1; i < L; i++) {
        ::reduce<false><<<levels[i], blockSize, 0, stream >>>(scratch_d + offsets[i],
                                                       scratch_d + offsets[i - 1],
                                                       levels[i - 1]);
      }

      // Run scan on last level L-1, and write off total sum to last element of output (offsets_d+N).
      ::scan<false, false, extraElement, writeSum, false><<<1, blockSize, 0, stream >>>(scratch_d + offsets[L - 1],
                                                                                        output_d + N,
                                                                                        sum_d,
                                                                                        scratch_d + offsets[L - 1],
                                                                                        nullptr,
                                                                                        levels[L - 1]);

      // Now, level L-1 is processed, scan levels L-2...0 pulling start offsets from the level above.
      for (uint32_t i = L - 1u; 0 < i; i--) {
        ::scan<false, false, false, false, true><<<levels[i], blockSize, 0, stream >>>(scratch_d + offsets[i - 1],
                                                                                       nullptr,
                                                                                       nullptr,
                                                                                       scratch_d + offsets[i - 1],
                                                                                       scratch_d + offsets[i],
                                                                                       levels[i - 1]);
      }

      // Now, level 0 is processed, scan input writing to output, pulling offsets from level 0.
      ::scan<subset, inclusive, false, false, true><<<levels[0], blockSize, 0, stream >>>(output_d,
                                                                                          nullptr,
                                                                                          nullptr,
                                                                                          input_d,
                                                                                          scratch_d + offsets[0],
                                                                                          N);
    }
  }

}


uint32_t ComputeStuff::Scan::scratchByteSize(uint32_t N)
{
  std::vector<uint32_t> levels;
  calcLevelSizes(levels, N);

  uint32_t size = 0;
  uint32_t alignment = 128/sizeof(uint32_t);
  for (auto & level : levels) {
    size += (level + alignment - 1) & ~(alignment - 1);
  }
  return sizeof(uint32_t)*size;
}


void ComputeStuff::Scan::exclusiveScan(uint32_t* output_d,
                                       uint32_t* scratch_d,
                                       const uint32_t* input_d,
                                       uint32_t N,
                                       cudaStream_t stream)
{
  runKernels<false, false, false, false>(output_d, nullptr, scratch_d, input_d, N, stream);
}

void ComputeStuff::Scan::inclusiveScan(uint32_t* output_d,
                                       uint32_t* scratch_d,
                                       const uint32_t* input_d,
                                       uint32_t N,
                                       cudaStream_t stream)
{
  runKernels<false, true, false, false>(output_d, nullptr, scratch_d, input_d, N, stream);
}

void ComputeStuff::Scan::calcOffsets(uint32_t* offsets_d,
                                     uint32_t* sum_d,
                                     uint32_t* scratch_d,
                                     const uint32_t* counts_d,
                                     uint32_t N,
                                     cudaStream_t stream)
{
  runKernels<false, false, true, true>(offsets_d, sum_d, scratch_d, counts_d, N, stream);
}

void ComputeStuff::Scan::calcOffsets(uint32_t* offsets_d,
                                     uint32_t* scratch_d,
                                     const uint32_t* counts_d,
                                     uint32_t N,
                                     cudaStream_t stream)
{
  runKernels<false, false, true, false>(offsets_d, nullptr, scratch_d, counts_d, N, stream);
}

void ComputeStuff::Scan::compact(uint32_t* out_d,
                                 uint32_t* sum_d,
                                 uint32_t* scratch_d,
                                 const uint32_t* in_d,
                                 uint32_t N,
                                 cudaStream_t stream)
{
  runKernels<true, false, false, true>(out_d, sum_d, scratch_d, in_d, N, stream);
}
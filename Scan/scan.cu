// This file is part of ComputeStuff copyright (C) 2017 Christopher Dyken.
// Released under the MIT license, please see LICENSE file for details.

#include <limits>
#include <vector>
#include <algorithm>
#include "Scan.h"
#include "scan_kernels.cuh"

namespace {

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
  template<bool inclusive, bool extraElement, bool writeSum>
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

      ComputeStuff::Scan::Kernels::scan<false, extraElement, writeSum, false><<<1, blockSize, 0, stream >>>(output_d,
                                                                                                            output_d + N,
                                                                                                            sum_d,
                                                                                                            input_d,
                                                                                                            nullptr,
                                                                                                            N);

    }
    else {

      uint32_t L = static_cast<uint32_t>(levels.size());

      // From input, populate level 0
      ComputeStuff::Scan::Kernels::reduce<<<levels[0], blockSize, 0, stream >>>(scratch_d + offsets[0],
                                                                                input_d,
                                                                                N);

      // From level i-1, populate level i, up to including L-1.
      for (uint32_t i = 1; i < L; i++) {
        ComputeStuff::Scan::Kernels::reduce<<<levels[i], blockSize, 0, stream >>>(scratch_d + offsets[i],
                                                                                  scratch_d + offsets[i - 1],
                                                                                  levels[i - 1]);
      }

      // Run scan on last level L-1, and write off total sum to last element of output (offsets_d+N).
      ComputeStuff::Scan::Kernels::scan<false, extraElement, writeSum, false><<<1, blockSize, 0, stream >>>(scratch_d + offsets[L - 1],
                                                                                                            output_d + N,
                                                                                                            sum_d,
                                                                                                            scratch_d + offsets[L - 1],
                                                                                                            nullptr,
                                                                                                            levels[L - 1]);

      // Now, level L-1 is processed, scan levels L-2...0 pulling start offsets from the level above.
      for (uint32_t i = L - 1u; 0 < i; i--) {
        ComputeStuff::Scan::Kernels::scan<false, false, false, true><<<levels[i], blockSize, 0, stream >>>(scratch_d + offsets[i - 1],
                                                                                                           nullptr,
                                                                                                           nullptr,
                                                                                                           scratch_d + offsets[i - 1],
                                                                                                           scratch_d + offsets[i],
                                                                                                           levels[i - 1]);
      }

      // Now, level 0 is processed, scan input writing to output, pulling offsets from level 0.
      ComputeStuff::Scan::Kernels::scan<inclusive, false, false, true><<<levels[0], blockSize, 0, stream >>>(output_d,
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
  runKernels<false, false, false>(output_d, nullptr, scratch_d, input_d, N, stream);
}

void ComputeStuff::Scan::inclusiveScan(uint32_t* output_d,
                                       uint32_t* scratch_d,
                                       const uint32_t* input_d,
                                       uint32_t N,
                                       cudaStream_t stream)
{
  runKernels<true, false, false>(output_d, nullptr, scratch_d, input_d, N, stream);
}

void ComputeStuff::Scan::calcOffsets(uint32_t* offsets_d,
                                     uint32_t* sum_d,
                                     uint32_t* scratch_d,
                                     const uint32_t* counts_d,
                                     uint32_t N,
                                     cudaStream_t stream)
{
  runKernels<false, true, true>(offsets_d, sum_d, scratch_d, counts_d, N, stream);
}

void ComputeStuff::Scan::calcOffsets(uint32_t* offsets_d,
                                     uint32_t* scratch_d,
                                     const uint32_t* counts_d,
                                     uint32_t N,
                                     cudaStream_t stream)
{
  runKernels<false, true, false>(offsets_d, nullptr, scratch_d, counts_d, N, stream);
}
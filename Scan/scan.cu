#include <limits>
#include <vector>
#include <algorithm>
#include "Scan.h"

#define WARPSIZE 32
#define SCAN_WARPS 4

namespace {

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



  template<bool inclusive, bool writeSum0, bool writeSum1, bool readOffset>
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

    if (inclusive == false) {
      s -= q; // exclusive scan
    }

    if (readOffset) {
      s += offset[blockIdx.x];
    }

    // Store
    if (threadOffset + 3 < N) {
      *reinterpret_cast<uint4*>(output + threadOffset) = make_uint4(s,
                                                                    s + a.x,
                                                                    s + a.x + a.y,
                                                                    s + a.x + a.y + a.z);
    }
    else if(threadOffset < N) {
      output[threadOffset + 0] = s;
      s += a.x;
      if (threadOffset + 1 < N) output[threadOffset + 1] = s;
      s += a.y;
      if (threadOffset + 2 < N) output[threadOffset + 2] = s;
    }

    
  }

  void calcLevelSizes(std::vector<uint32_t>& levels, uint32_t N)
  {
    uint32_t R = 4 * SCAN_WARPS*WARPSIZE; // Amount of reduction per level.
    levels.clear();
    while (1 < N) {
      N = (N + R - 1) / R;
      levels.push_back(N);
    }
    if (!levels.empty()) levels.pop_back(); // Remove to-one reduction that is always present.
  }

}


uint32_t ComputeStuff::Scan::levels(uint32_t N)
{
  std::vector<uint32_t> levels;
  calcLevelSizes(levels, N);

  return static_cast<uint32_t>(levels.size());
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


void ComputeStuff::Scan::calcOffsets(uint32_t* offsets_d,
                                     uint32_t* sum_d,
                                     uint32_t* scratch_d,
                                     const uint32_t* counts_d,
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

    if (sum_d == nullptr) {
      scan<false, true, false, false> << <1, blockSize, 0, stream >> > (offsets_d,
                                                                        offsets_d + N,
                                                                        sum_d,
                                                                        counts_d,
                                                                        nullptr,
                                                                        N);
    }
    else {
      scan<false, true, true, false> << <1, blockSize, 0, stream >> > (offsets_d,
                                                                       offsets_d + N,
                                                                       sum_d,
                                                                       counts_d,
                                                                       nullptr,
                                                                       N);
    }

  }
  else {
    uint32_t L = static_cast<uint32_t>(levels.size());

    // From input, populate level 0
    reduce << <levels[0], blockSize, 0, stream >> > (scratch_d + offsets[0],
                                                     counts_d,
                                                     N);
    // From level i-1, populate level i, up to including L-1.
    for (uint32_t i = 1; i < L; i++) {
      reduce << <levels[i], blockSize, 0, stream >> > (scratch_d + offsets[i],
                                                       scratch_d + offsets[i - 1],
                                                       levels[i - 1]);
    }

    // Run scan on last level L-1, and write off total sum to last element of output (offsets_d+N).
    if (sum_d == nullptr) {
      scan<false, true, false, false> << <1, blockSize, 0, stream >> > (scratch_d + offsets[L - 1],
                                                                        offsets_d + N,
                                                                        nullptr,
                                                                        scratch_d + offsets[L - 1],
                                                                        nullptr,
                                                                        levels[L - 1]);
    }
    else {
      scan<false, true, true, false> << <1, blockSize, 0, stream >> > (scratch_d + offsets[L - 1],
                                                                       offsets_d + N,
                                                                       sum_d,
                                                                       scratch_d + offsets[L - 1],
                                                                       nullptr,
                                                                       levels[L - 1]);
    }

    // Now, level L-1 is processed, scan levels L-2...0 pulling start offsets from the level above.
    for (uint32_t i = L - 1u; 0 < i; i--) {
      scan<false, false, false, true> << <levels[i], blockSize, 0, stream >> > (scratch_d + offsets[i - 1],
                                                                                nullptr,
                                                                                nullptr,
                                                                                scratch_d + offsets[i - 1],
                                                                                scratch_d + offsets[i],
                                                                                levels[i - 1]);
    }
    // Now, level 0 is processed, scan input writing to output, pulling offsets from level 0.

    scan<false, false, false, true> << <levels[0], blockSize, 0, stream >> > (offsets_d,
                                                                              nullptr,
                                                                              nullptr,
                                                                              counts_d,
                                                                              scratch_d + offsets[0],
                                                                              N);
  }


}

void ComputeStuff::Scan::calcOffsets(uint32_t* offsets_d,
                                     uint32_t* scratch_d,
                                     const uint32_t* counts_d,
                                     uint32_t N,
                                     cudaStream_t stream)
{
  calcOffsets(offsets_d,
              nullptr,
              scratch_d,
              counts_d,
              N,
              stream);
}
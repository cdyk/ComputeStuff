#include <limits>
#include <vector>
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



  template<bool inclusive, bool writeSum0, bool writeSum1>
  __global__
  __launch_bounds__(SCAN_WARPS * WARPSIZE)
  void
  scan(uint32_t* output,
       uint32_t* sum0,
       uint32_t* sum1,
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
    uint32_t q = s;

    // Per-warp reduce
    #pragma unroll
    for (uint32_t i = 1; i < WARPSIZE; i*=2) {
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


  void calcLevelSizes(std::vector<size_t>& levels, size_t N)
  {
    size_t R = 4 * SCAN_WARPS*WARPSIZE; // Amount of reduction per level.
    levels.clear();
    while (1 < N) {
      N = (N + R - 1) / R;
      levels.push_back(N);
    }
    if (!levels.empty()) levels.pop_back(); // Remove to-one reduction that is always present.
  }

}



template<>
size_t ComputeStuff::Scan::scratchByteSize<uint32_t>(size_t N)
{
  std::vector<size_t> levels;
  calcLevelSizes(levels, N);

  size_t size = 0;
  size_t alignment = 128/sizeof(uint32_t);
  for (auto & level : levels) {
    size += (level + alignment - 1) & ~alignment;
  }
  return sizeof(uint32_t)*size;
}

void ComputeStuff::Scan::calcOffsets(uint32_t* offsets_d,
                                     uint32_t* sum_d,
                                     uint32_t* scratch_d,
                                     const uint32_t* counts_d,
                                     size_t N,
                                     cudaStream_t stream)
{
  std::vector<size_t> levels;
  calcLevelSizes(levels, N);

  std::vector<size_t> offsets;
  offsets.push_back(0);
  size_t alignment = 128 / sizeof(uint32_t);
  for (size_t i = 0; i < levels.size(); i++) {
    size_t levelSize = (levels[i] + alignment - 1) & ~alignment;
    offsets.push_back(offsets[i] + levelSize);
  }

  if (N <= std::numeric_limits<uint32_t>::max()) {
    uint32_t blockSize = SCAN_WARPS * WARPSIZE;
    uint32_t n = static_cast<uint32_t>(N);

    if (levels.empty()) {

      if (sum_d == nullptr) {
        scan<false, true, false> <<<1, blockSize, 0, stream>>> (offsets_d,
                                                                offsets_d + N,
                                                                sum_d,
                                                                counts_d,
                                                                n);
      }
      else {
        scan<false, true, true> <<<1, blockSize, 0, stream>>> (offsets_d,
                                                               offsets_d + N,
                                                               sum_d,
                                                               counts_d,
                                                               n);
      }

    }
    else {

      reduce<<<static_cast<uint32_t>(levels[0]), blockSize, 0, stream>>>(scratch_d + offsets[0],
                                                                         counts_d,
                                                                         n);
      for (size_t i = 1; i < levels.size(); i++) {
        reduce<<<static_cast<uint32_t>(levels[i]), blockSize, 0, stream>>>(scratch_d + offsets[i],
                                                                           scratch_d + offsets[i - 1],
                                                                           levels[i - 1]);
      }

      auto l = levels.size() - 1;
      if (sum_d == nullptr) {
        scan<false, true, false><<<1, blockSize, 0, stream>>>(scratch_d + offsets[l],
                                                              offsets_d + N,
                                                              nullptr,
                                                              scratch_d + offsets[l],
                                                              levels[l]);
      }
      else {
        scan<false, true, true><<<1, blockSize, 0, stream>>>(scratch_d + offsets[l],
                                                             offsets_d + N,
                                                             sum_d,
                                                             scratch_d + offsets[l],
                                                             levels[l]);
      }


    }


  }
}

void ComputeStuff::Scan::calcOffsets(uint32_t* offsets_d,
                                     uint32_t* scratch_d,
                                     const uint32_t* counts_d,
                                     size_t N,
                                     cudaStream_t stream)
{
  calcOffsets(offsets_d,
              nullptr,
              scratch_d,
              counts_d,
              N,
              stream);
}
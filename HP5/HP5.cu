#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "HP5.h"

#define HP5_WARP_COUNT 5
#define HP5_WARP_SIZE 32

namespace {

  // Reads 160 values, outputs HP level of 128 values, and 32 sideband values.

  template<bool mask_input>
  __global__
  __launch_bounds__(HP5_WARP_COUNT * HP5_WARP_SIZE)
  void
  reduce1(uint4* __restrict__ hp1_d,
          uint32_t* __restrict__ sb1_d,
          const uint32_t* __restrict__ sb0_d,
          uint32_t N)
  {
    const uint32_t threadOffset = HP5_WARP_COUNT * HP5_WARP_SIZE * blockIdx.x + threadIdx.x;

    // Idea, each warp reads 32 values. read instead 32/4 uint4's.

    __shared__ uint32_t sb[HP5_WARP_COUNT * HP5_WARP_SIZE];
    uint32_t a = threadOffset < N ? sb0_d[threadOffset] : 0;
    if (mask_input) {
      a = a != 0 ? 1 : 0;
    }
    sb[threadIdx.x] = a;

    __syncthreads();
    if (threadIdx.x < HP5_WARP_SIZE) { // First warp
      uint4 hp = make_uint4(sb[5 * threadIdx.x + 0],
                            sb[5 * threadIdx.x + 1],
                            sb[5 * threadIdx.x + 2],
                            sb[5 * threadIdx.x + 3]);
      hp1_d[32 * blockIdx.x + threadIdx.x] = hp;
      sb1_d[32 * blockIdx.x + threadIdx.x] = hp.x + hp.y + hp.z + hp.w + sb[5 * threadIdx.x + 4];
    }
  }

  template<bool mask_input, bool write_sum>
  __global__
  __launch_bounds__(128)
  void
  reduceApex(uint4* __restrict__ apex_d,
             uint32_t* sum_d,
             const uint32_t* in_d,
             uint32_t N)
  {
    // 0 : sum + 3 padding
    // 1 : 1 uvec4 of level 0.
    // 2 : 5 values of level 0 (top)
    // 7 : 25 values of level 1
    // 32: total sum.

    // Fetch up to 125 elements from in_d.
    uint32_t a = threadIdx.x < N ? in_d[threadIdx.x] : 0;
    if (mask_input) {
      a = a != 0 ? 1 : 0;
    }
    __shared__ uint32_t sb[125 + 25];
    sb[threadIdx.x] = a;

    // Store 5x5 uint4's at uint4 offset 0 (25x4=100 elements, corresponding to 125 inputs).
    __syncthreads();
    if (threadIdx.x < 25) {
      uint32_t e0 = sb[5 * threadIdx.x + 0];
      uint32_t e1 = sb[5 * threadIdx.x + 1];
      uint32_t e2 = sb[5 * threadIdx.x + 2];
      uint32_t e3 = sb[5 * threadIdx.x + 3];
      uint32_t e4 = sb[5 * threadIdx.x + 4];
      apex_d[7 + threadIdx.x] = make_uint4(e0,
                                       e0 + e1,
                                       e0 + e1 + e2,
                                       e0 + e1 + e2 + e3);

      sb[125 + threadIdx.x] = e0 + e1 + e2 + e3 + e4;
    }

    // Store 5 uint4's at uint4 offset 25 (5x4=20 elements, corresponding to 25 inputs).
    __syncthreads();
    if (threadIdx.x < 5) {
      uint32_t e0 = sb[125 + 5 * threadIdx.x + 0];
      uint32_t e1 = sb[125 + 5 * threadIdx.x + 1];
      uint32_t e2 = sb[125 + 5 * threadIdx.x + 2];
      uint32_t e3 = sb[125 + 5 * threadIdx.x + 3];
      uint32_t e4 = sb[125 + 5 * threadIdx.x + 4];
      apex_d[2 + threadIdx.x] = make_uint4(e0,
                                            e0 + e1,
                                            e0 + e1 + e2,
                                            e0 + e1 + e2 + e3);

      sb[threadIdx.x] = e0 + e1 + e2 + e3 + e4;
    }

    // Store 1 uint4 at uint4 offset 30 (1x4=4 elements, corresponding to 5 inputs)
    // Store total at uint4 offset 31
    __syncthreads();
    if (threadIdx.x < 1) {
      uint32_t e0 = sb[0];
      uint32_t e1 = sb[1];
      uint32_t e2 = sb[2];
      uint32_t e3 = sb[3];
      uint32_t e4 = sb[4];
      apex_d[1 + threadIdx.x] = make_uint4(e0,
                                            e0 + e1,
                                            e0 + e1 + e2,
                                            e0 + e1 + e2 + e3);
      uint32_t s = e0 + e1 + e2 + e3 + e4;
      *reinterpret_cast<uint32_t*>(apex_d) = s;
      if (write_sum) {
        *sum_d = s;
      }
    }

  }

  void scratchLayout(std::vector<uint32_t>& levels, std::vector<uint32_t>& offsets, uint32_t N)
  {
    if (N == 0) return;

    // Apex-level is always present.
    // Levels below apex, reduction is done in 160 -> 32 blocks.
    while (125 < N) {
      levels.push_back((N + 159) / 160);  // Number of blocks per level
      N = 32 * levels.back();
    }


    offsets.resize(levels.size() + 4);

    uint32_t o = 0;
    offsets[levels.size()] = o;  // Apex
    o += 128;

    for (int i = static_cast<int>(levels.size()) - 1; 0 <= i; i--) {
      offsets[i] = o; // HP level i
      o += 32 * 4 * levels[i];
    }
    offsets[levels.size() + 1] = o; // Large sideband buffer
    o += 32 * (levels.empty() ? 0 : levels[0]);

    offsets[levels.size() + 2] = o; // Small sideband buffer
    o += 32 * (levels.size() < 2 ? 0 : levels[1]);

    offsets[levels.size() + 3] = o; // Final size
  }

}

size_t ComputeStuff::HP5::scratchByteSize(uint32_t N)
{
  std::vector<uint32_t> levels;
  std::vector<uint32_t> offsets;
  scratchLayout(levels, offsets, N);
  return sizeof(uint32_t)*offsets.back();
}

void ComputeStuff::HP5::compact(uint32_t* out_d,
                                uint32_t* sum_d,
                                uint32_t* scratch_d,
                                const uint32_t* in_d,
                                uint32_t N,
                                cudaStream_t stream)
{
  if (N == 0) return;

  std::vector<uint32_t> levels;
  std::vector<uint32_t> offsets;
  scratchLayout(levels, offsets, N);

  if (levels.empty()) {
    reduceApex<true, true> << <1, 128, 0, stream >> > (reinterpret_cast<uint4*>(scratch_d), sum_d, in_d, N);
  }
  else {
    abort();
  }
  cudaStreamSynchronize(stream);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) {
    abort();
  }


/*

  auto L = static_cast<uint32_t>(levels.size());



  ::reduce1<true><<<(N+159)/160, 160, 0, stream>>>(reinterpret_cast<uint4*>(scratch_d + offsets[0]),
                                                   scratch_d + offsets[L + 1],
                                                   in_d,
                                                   N);
  for (uint32_t l = 1; l < L; l++) {
    ::reduce1<false><<<(levels[l - 1] + 159) / 160, 160, 0, stream>>>(reinterpret_cast<uint4*>(scratch_d + offsets[L + 1 + ((l + 1) & 1)]),
                                                                      scratch_d + offsets[L + 1 + (l & 1)],
                                                                      in_d,
                                                                      levels[l - 1]);
  }

  */
}

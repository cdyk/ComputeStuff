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


  void scratchLayout(std::vector<uint32_t>& levels, std::vector<uint32_t>& offsets, uint32_t N)
  {
    uint32_t alignment = 128 / sizeof(uint32_t);

    do {
      levels.push_back((N + 4) / 5);  // Number of quintuplets
      N = levels.back();
    } while (1 < N);

    offsets.push_back(0);
    for (auto & q : levels) {
      offsets.push_back(offsets.back() + 4 * q); // Each HP level uses one uint4 per 5 values.
    }
    // One uint for the total sum + padding
    offsets.push_back((offsets.back() + 1 + alignment - 1) & ~alignment);

    // Sideband ping-pong area 1
    offsets.push_back((offsets.back() + levels[0] + 1 + alignment - 1) & ~alignment);
    if (1 < levels.size()) {
      // Sideband ping-pong area 2
      offsets.push_back((offsets.back() + levels[1] + 1 + alignment - 1) & ~alignment);
    }


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

  cudaStreamSynchronize(stream);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) {
    abort();
  }

}

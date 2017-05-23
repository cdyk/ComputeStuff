#include <array>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cassert>
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

  /*template<uint32_t L>
  struct LevelOffsets
  {
    __host__ LevelOffsets(const LevelOffsets&) = default;
    __host__ LevelOffsets(uint32_t* v) { for (uint32_t i = 0; i < L; i++) value[i] = v[i]; }
    uint32_t value[L];
  };*/

  __device__
  inline uint32_t processHistoElement(uint32_t& key, uint32_t offset, const uint4 element)
  {
    assert(element.x <= element.y);
    assert(element.y <= element.z);
    assert(element.z <= element.w);

    if (key < element.x) {
    }
    else if (key < element.y) {
      key -= element.x;
      offset += 1;
    }
    else if (key < element.z) {
      key -= element.y;
      offset += 2;
    }
    else if (key < element.w) {
      key -= element.z;
      offset += 3;
    }
    else  {
      key -= element.w;
      offset += 4;
    }
    return offset;
  }

  __device__
  inline uint32_t processDataElement(uint32_t& key, uint32_t offset, const uint4 element)
  {
    if (element.x <= key) {
      key -= element.x;
      offset++;
      if (element.y <= key) {
        key -= element.y;
        offset++;
        if (element.z <= key) {
          key -= element.z;
          offset++;
          if (element.w <= key) {
            key -= element.w;
            offset++;
          }
        }
      }
    }
    return offset;
  }

  struct OffsetBlob
  {
    uint32_t data0;
    uint32_t data1;
    uint32_t data2;
    uint32_t data3;
    uint32_t data4;
    uint32_t data5;
    uint32_t data6;
    uint32_t data7;
    uint32_t data8;
    uint32_t data9;
    uint32_t dataA;
    uint32_t dataB;
  };

  template<uint32_t L>
  __global__
  __launch_bounds__(128)
  void extract(uint32_t* __restrict__ out_d,
               const uint4* __restrict__ hp_d,
               OffsetBlob offsetBlob)
  {
    const uint32_t index = 128 * blockIdx.x + threadIdx.x;
    if (hp_d[0].x <= index) return;

    __shared__ uint32_t offsets[12];
    if (0 < L) offsets[0] = offsetBlob.data0;
    if (1 < L) offsets[1] = offsetBlob.data1;
    if (2 < L) offsets[2] = offsetBlob.data2;
    if (3 < L) offsets[3] = offsetBlob.data3;
    if (4 < L) offsets[4] = offsetBlob.data4;
    if (5 < L) offsets[5] = offsetBlob.data5;
    if (6 < L) offsets[6] = offsetBlob.data6;
    if (7 < L) offsets[7] = offsetBlob.data7;
    if (8 < L) offsets[8] = offsetBlob.data8;
    if (9 < L) offsets[9] = offsetBlob.data9;
    if (10 < L) offsets[10] = offsetBlob.dataA;
    if (11 < L) offsets[11] = offsetBlob.dataB;

    // Traverse apex.
    uint32_t offset = 0;
    uint32_t key = index;
    offset = processHistoElement(key, 5 * offset, hp_d[1]);
    offset = processHistoElement(key, 5 * offset, hp_d[2 + offset]);
    offset = processHistoElement(key, 5 * offset, hp_d[7 + offset]);
    for (uint32_t i = L; 0 < i; i--) {
      offset = processDataElement(key, 5 * offset, hp_d[offsets[i - 1] + offset]);
    }
    out_d[index] = offset;
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
    o += 32;

    for (int i = static_cast<int>(levels.size()) - 1; 0 <= i; i--) {
      offsets[i] = o; // HP level i
      o += 32 * levels[i];
    }
    offsets[levels.size() + 1] = o; // Large sideband buffer
    o += (32 / 4) * (levels.empty() ? 0 : levels[0]);

    offsets[levels.size() + 2] = o; // Small sideband buffer
    o += (32 / 4) * (levels.size() < 2 ? 0 : levels[1]);

    offsets[levels.size() + 3] = o; // Final size
  }

}

size_t ComputeStuff::HP5::scratchByteSize(uint32_t N)
{
  std::vector<uint32_t> levels;
  std::vector<uint32_t> offsets;
  scratchLayout(levels, offsets, N);
  return 4 * sizeof(uint32_t)*offsets.back();
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

  auto L = levels.size();
  if (L == 0) {
    reduceApex<true, true><<<1, 128, 0, stream>>>(reinterpret_cast<uint4*>(scratch_d),
                                                  sum_d,
                                                  in_d,
                                                  N);
  }
  else {
    ::reduce1<true><<<levels[0], 160, 0, stream>>>(reinterpret_cast<uint4*>(scratch_d) + offsets[0],
                                                   scratch_d + 4 * offsets[L + 1],
                                                   in_d,
                                                   N);
    for (size_t i = 1; i < levels.size(); i++) {
      ::reduce1<false><<<levels[i], 160, 0, stream>>>(reinterpret_cast<uint4*>(scratch_d) + offsets[i],
                                                      scratch_d + 4*offsets[L + 1 + ((i + 0) & 1)],
                                                      scratch_d + 4*offsets[L + 1 + ((i + 1) & 1)],
                                                      32 * levels[i - 1]);

    }
    ::reduceApex<false, true><<<1, 128, 0, stream>>>(reinterpret_cast<uint4*>(scratch_d),
                                                     sum_d,
                                                     scratch_d + 4*offsets[L + 1 + ((L + 1) & 1)],
                                                     32 * levels[L - 1]);
  }


  OffsetBlob offsetBlob;
  std::memcpy(&offsetBlob.data0, offsets.data(), sizeof(uint32_t)*std::min(size_t(12), L));

  switch (L)
  {
  case 0: ::extract<0><<<(N + 127) / 128, 128, 0, stream>>>(out_d, reinterpret_cast<uint4*>(scratch_d), offsetBlob); break;
  case 1: ::extract<1><<<(N + 127) / 128, 128, 0, stream>>>(out_d, reinterpret_cast<uint4*>(scratch_d), offsetBlob); break;
  case 2: ::extract<2><<<(N + 127) / 128, 128, 0, stream>>>(out_d, reinterpret_cast<uint4*>(scratch_d), offsetBlob); break;
  case 3: ::extract<3><<<(N + 127) / 128, 128, 0, stream>>>(out_d, reinterpret_cast<uint4*>(scratch_d), offsetBlob); break;
  case 4: ::extract<4><<<(N + 127) / 128, 128, 0, stream>>>(out_d, reinterpret_cast<uint4*>(scratch_d), offsetBlob); break;
  case 5: ::extract<5><<<(N + 127) / 128, 128, 0, stream>>>(out_d, reinterpret_cast<uint4*>(scratch_d), offsetBlob); break;
  case 6: ::extract<6><<<(N + 127) / 128, 128, 0, stream>>>(out_d, reinterpret_cast<uint4*>(scratch_d), offsetBlob); break;
  case 7: ::extract<7><<<(N + 127) / 128, 128, 0, stream>>>(out_d, reinterpret_cast<uint4*>(scratch_d), offsetBlob); break;
  case 8: ::extract<8><<<(N + 127) / 128, 128, 0, stream>>>(out_d, reinterpret_cast<uint4*>(scratch_d), offsetBlob); break;
  case 9: ::extract<9><<<(N + 127) / 128, 128, 0, stream>>>(out_d, reinterpret_cast<uint4*>(scratch_d), offsetBlob); break;
  case 10: ::extract<10><<<(N + 127) / 128, 128, 0, stream>>>(out_d, reinterpret_cast<uint4*>(scratch_d), offsetBlob); break;
  case 11: ::extract<11><<<(N + 127) / 128, 128, 0, stream>>>(out_d, reinterpret_cast<uint4*>(scratch_d), offsetBlob); break;
  default:
    abort();
    break;
  }

#if 1
  cudaStreamSynchronize(stream);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) {
    abort();
  }
#endif
}

#include <array>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cassert>
#include <device_launch_parameters.h>

#include "HP5.h"

#if defined(__LP64__) || defined(_WIN64)
#define PTR_REG "l"
#define PTR_REG_DST "=l"
#else
#define PTR_REG "r"
#define PTR_REG_DST "=r"
#endif

#define HP5_WARP_COUNT 5
#define HP5_WARP_SIZE 32

namespace {


  __global__ __launch_bounds__(128) void reduceBase(uint32_t* __restrict__ hp_d,
                                                    uint32_t* __restrict__ sb_d,
                                                    const uint32_t n1,
                                                    const uint32_t* __restrict__ src,
                                                    const uint32_t n0)
  {
    const uint32_t offset0 = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t lane = threadIdx.x %  HP5_WARP_SIZE;
    const uint32_t value = offset0 < n0 ? src[offset0] : 0;
    const uint32_t warpMask = __ballot(value != 0);
    if (lane == 0) {
      const uint32_t offset1 = offset0 / HP5_WARP_SIZE;
      if (offset1 < n1) {
        hp_d[offset1] = warpMask;
        sb_d[offset1] = __popc(warpMask);
      }
    }
  }

  __device__ __forceinline__ uint32_t _bfe(uint32_t val, uint32_t o, uint32_t n)
  {
    uint32_t rv;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(rv) : "r"(val), "r"(o), "r"(n));
    return rv;
  }

  __global__ __launch_bounds__(256) void reduceBase2b(uint32_t* __restrict__ hp2_d,
                                                      uint32_t* __restrict__ sb2_d,
                                                      const uint32_t n2,
                                                      uint32_t* __restrict__ hp1_d,
                                                      const uint32_t n1,
                                                      const uint32_t* __restrict__ sb0_d,
                                                      const uint32_t n0)
  {
    const uint32_t offset0_base = 32 * 160 * blockIdx.x + 4 * threadIdx.x;
    const uint32_t offset1_base = 160 * blockIdx.x;

    const uint32_t lane = threadIdx.x %  HP5_WARP_SIZE;
    const uint32_t warp = threadIdx.x / HP5_WARP_SIZE;

    __shared__ uint32_t sb1[160];

    // Each warp reads 32*4=128 bytes

    for (uint32_t i = 0; i < 5; i++) {
      const uint32_t offset0 = offset0_base + 4 * 256 * i;

      uint4 value;
      if (offset0 + 3 < n0) {
        asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w) : PTR_REG(sb0_d + offset0));
      }
      else {
        value = make_uint4(0, 0, 0, 0);
        if (offset0 < n0) {
          asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(value.x) : PTR_REG(sb0_d + offset0));
          if (offset0 + 1 < n0) {
            asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(value.y) : PTR_REG(sb0_d + offset0 + 1));
            if (offset0 + 2 < n0) {
              asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(value.z) : PTR_REG(sb0_d + offset0 + 2));
            }
          }
        }
      }

      // Prefetch data for next row into L2
      if (i + 1 < 5) {
        asm volatile("prefetch.global.L2 [%0];"::PTR_REG(sb0_d + offset0 + 4 * 256));
      }

      const uint32_t warpMaskA = __ballot(value.x != 0);
      const uint32_t warpMaskB = __ballot(value.y != 0);
      const uint32_t warpMaskC = __ballot(value.z != 0);
      const uint32_t warpMaskD = __ballot(value.w != 0);

      if (lane < 4) {
        uint32_t warpMaskASub = _bfe(warpMaskA, 8 * lane, 8);
        uint32_t warpMaskBSub = _bfe(warpMaskB, 8 * lane, 8);
        uint32_t warpMaskCSub = _bfe(warpMaskC, 8 * lane, 8);
        uint32_t warpMaskDSub = _bfe(warpMaskD, 8 * lane, 8);

        uint32_t warpMask = warpMaskDSub<<24 | warpMaskCSub<<16 | warpMaskBSub<<8 | warpMaskASub;

        warpMask = warpMask & 0xF0F00F0F | (warpMask & 0x0000F0F0) << 12 | (warpMask & 0x0F0F0000) >> 12;
        warpMask = warpMask & 0xCC33CC33 | (warpMask & 0x00CC00CC) << 6 | (warpMask & 0x33003300) >> 6;
        warpMask = warpMask & 0xA5A5A5A5 | (warpMask & 0x0A0A0A0A) << 3 | (warpMask & 0x50505050) >> 3;
        warpMask = warpMask & 0x99999999 | (warpMask & 0x22222222) << 1 | (warpMask & 0x44444444) >> 1;

        const uint32_t offset1 = offset1_base + 32 * i + 4 * warp + lane;
        if (offset1 < n1) {
          asm volatile("st.global.cs.u32 [%0], %1;"::PTR_REG(hp1_d + offset1), "r"(warpMask));
        }
        sb1[32 * i + 4 * warp + lane] = __popc(warpMask);
      }
    }

    __syncthreads();
    if (threadIdx.x < HP5_WARP_SIZE) { // First warp
      const uint32_t offset2 = 32 * blockIdx.x + threadIdx.x;
      if (offset2 < n2) {
        uint4 hp = make_uint4(sb1[5 * threadIdx.x + 0],
                              sb1[5 * threadIdx.x + 1],
                              sb1[5 * threadIdx.x + 2],
                              sb1[5 * threadIdx.x + 3]);
#if 1
        // Streaming store
        asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};"
                     ::PTR_REG((uint4*)hp2_d + offset2), "r"(hp.x), "r"(hp.y), "r"(hp.z), "r"(hp.w));
#else
        ((uint4*)hp2_d)[offset2] = hp;
#endif
        sb2_d[offset2] = hp.x + hp.y + hp.z + hp.w + sb1[5 * threadIdx.x + 4];
      }
    }

  }

  __global__ __launch_bounds__(256) void reduceBase2(uint32_t* __restrict__ hp2_d,
                                                     uint32_t* __restrict__ sb2_d,
                                                     const uint32_t n2,
                                                     uint32_t* __restrict__ hp1_d,
                                                     const uint32_t n1,
                                                     const uint32_t* __restrict__ sb0_d,
                                                     const uint32_t n0)
  {
    const uint32_t offset0_base = 32 * 160 * blockIdx.x + threadIdx.x;
    const uint32_t offset1_base = 160 * blockIdx.x;

    const uint32_t lane = threadIdx.x %  HP5_WARP_SIZE;
    const uint32_t warp = threadIdx.x / HP5_WARP_SIZE;

    __shared__ uint32_t sb1[160];

    // Each warp reads 32*4=128 bytes

    for (uint32_t i = 0; i < 20; i++) {
      const uint32_t offset0 = offset0_base + 256 * i;
#if 1
      // Streaming load
      uint32_t value;
      if (offset0 < n0) {
        asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(value) : PTR_REG(sb0_d + offset0));
      }
      else {
        value = 0;
      }
#else
      // regular load 
      // ld.global.nc.u32 / LDG.E.CI
      const uint32_t value = offset0 < n0 ? sb0_d[offset0] : 0;
#endif

#if 1
      // Prefetch data for next row into L2
      if (i + 1 < 20) {
        asm volatile("prefetch.global.L1 [%0];"::PTR_REG(sb0_d + offset0 + 256));
      }
#endif

      const uint32_t warpMask = __ballot(value != 0);
      if (lane == 0) {
        const uint32_t offset1 = offset1_base + 8 * i + warp;
        if (offset1 < n1) {
#if 1
          asm volatile("st.global.cs.u32 [%0], %1;"::PTR_REG(hp1_d + offset1), "r"(warpMask));
#else
          hp1_d[offset1] = warpMask;
#endif
        }
        sb1[8 * i + warp] = __popc(warpMask);
      }
    }

    __syncthreads();
    if (threadIdx.x < HP5_WARP_SIZE) { // First warp
      const uint32_t offset2 = 32 * blockIdx.x + threadIdx.x;
      if (offset2 < n2) {
        uint4 hp = make_uint4(sb1[5 * threadIdx.x + 0],
                              sb1[5 * threadIdx.x + 1],
                              sb1[5 * threadIdx.x + 2],
                              sb1[5 * threadIdx.x + 3]);
#if 1
        // Streaming store
        asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};"
                     :: PTR_REG((uint4*)hp2_d + offset2), "r"(hp.x), "r"(hp.y), "r"(hp.z), "r"(hp.w));
#else
        ((uint4*)hp2_d)[offset2] = hp;
#endif
        sb2_d[offset2] = hp.x + hp.y + hp.z + hp.w + sb1[5 * threadIdx.x + 4];
      }
    }

  }

  // Reads 160 values, outputs HP level of 128 values, and 32 sideband values.
  __global__
  __launch_bounds__(HP5_WARP_COUNT * HP5_WARP_SIZE)
  void
  reduce1(uint4* __restrict__ hp1_d,
          uint32_t* __restrict__ sb1_d,
          const uint32_t n1,
          const uint32_t* __restrict__ sb0_d,
          const uint32_t n0)
  {
    const uint32_t offset0 = HP5_WARP_COUNT * HP5_WARP_SIZE * blockIdx.x + threadIdx.x;

    // Idea, each warp reads 32 values. read instead 32/4 uint4's.

    __shared__ uint32_t sb[HP5_WARP_COUNT * HP5_WARP_SIZE];
    sb[threadIdx.x] = offset0 < n0 ? sb0_d[offset0] : 0;

    __syncthreads();
    if (threadIdx.x < HP5_WARP_SIZE) { // First warp
      const uint32_t offset1 = 32 * blockIdx.x + threadIdx.x;
      if (offset1 < n1) {
        uint4 hp = make_uint4(sb[5 * threadIdx.x + 0],
                              sb[5 * threadIdx.x + 1],
                              sb[5 * threadIdx.x + 2],
                              sb[5 * threadIdx.x + 3]);
        hp1_d[offset1] = hp;
        sb1_d[offset1] = hp.x + hp.y + hp.z + hp.w + sb[5 * threadIdx.x + 4];
      }
    }
  }

  __global__  __launch_bounds__(5 * HP5_WARP_SIZE)
  void reduce2(uint4* __restrict__ hp2_d,
               uint32_t* __restrict__ sb2_d,
               const uint32_t n2,
               uint4* __restrict__ hp1_d,
               const uint32_t n1,
               const uint32_t* __restrict__ sb0_d,
               const uint32_t n0)
  {
    const uint32_t offset0 = 5 * 5 * HP5_WARP_SIZE * blockIdx.x;

    __shared__ uint32_t sb[5 * 5 * HP5_WARP_SIZE];
    uint32_t o = threadIdx.x;
    for (uint32_t i = 0; i < 5; i++) {
      uint32_t q = offset0 + o;
      sb[o] = q < n0 ? sb0_d[q] : 0;
      o += 5 * HP5_WARP_SIZE;
    }

    __syncthreads();

    __shared__ uint32_t sb1[5 * HP5_WARP_SIZE];
    const uint32_t offset1 = 5 * HP5_WARP_SIZE * blockIdx.x + threadIdx.x;
    uint4 hp = make_uint4(sb[5 * threadIdx.x + 0],
                          sb[5 * threadIdx.x + 1],
                          sb[5 * threadIdx.x + 2],
                          sb[5 * threadIdx.x + 3]);
    if (offset1 < n1) {
      hp1_d[offset1] = hp;
    }
    sb1[threadIdx.x] = hp.x + hp.y + hp.z + hp.w + sb[5 * threadIdx.x + 4];

    __syncthreads();

    if (threadIdx.x < HP5_WARP_SIZE) { // First warp
      const uint32_t offset2 = HP5_WARP_SIZE * blockIdx.x + threadIdx.x;
      if (offset2 < n2) {
        uint4 hp = make_uint4(sb1[5 * threadIdx.x + 0],
                              sb1[5 * threadIdx.x + 1],
                              sb1[5 * threadIdx.x + 2],
                              sb1[5 * threadIdx.x + 3]);
        hp2_d[offset2] = hp;
        sb2_d[offset2] = hp.x + hp.y + hp.z + hp.w + sb1[5 * threadIdx.x + 4];
      }
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

  __device__ __forceinline__ uint32_t processHistoElement(uint32_t& key, uint32_t offset, const uint4 element)
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

  __device__ __forceinline__ uint32_t processDataElement(uint32_t& key, uint32_t offset, const uint4 element)
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

  __device__ __forceinline__ uint32_t processMaskElement(uint32_t& key, uint32_t offset, uint32_t mask)
  {
    const uint32_t m16 = 0xffffu;
    const uint32_t c16 = __popc(mask & m16);
    if (c16 <= key) {  // Key is in upper 16 bits
      key -= c16;
      offset += 16;
      mask = mask >> 16;
    }
    const uint32_t m8 = 0xffu;
    const uint32_t c8 = __popc(mask & m8);
    if(c8 <= key) { // Key is in upper 8 bits
      key -= c8;
      offset += 8;
      mask = mask >> 8;
    }
    const uint32_t m4 = 0xfu;
    const uint32_t c4 = __popc(mask & m4);
    if (c4 <= key) { // Key is in upper 4 bits
      key -= c4;
      offset += 4;
      mask = mask >> 4;
    }
    const uint32_t m2 = 0x3u;
    const uint32_t c2 = __popc(mask & m2);
    if (c2 <= key) { // Key is in upper 2 bits
      key -= c2;
      offset += 2;
      mask = mask >> 2;
    }

    if ((mask & 0x1) <= key) {
      offset++;
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
               const uint32_t* __restrict__ hp_d,
               const OffsetBlob offsetBlob)
  {

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

    uint32_t N = hp_d[0];
    uint4 T = *(const uint4*)(hp_d + 4);

    for (uint32_t k = blockDim.x * blockIdx.x; k < N; k += gridDim.x * blockDim.x) {
      const uint32_t index = k + threadIdx.x;
      if (index < N) {
        uint32_t offset = 0;
        uint32_t key = index;
        offset = processHistoElement(key, 5 * offset, T);
        offset = processHistoElement(key, 5 * offset, *(const uint4*)(hp_d + 8 + 4 * offset));
        offset = processHistoElement(key, 5 * offset, *(const uint4*)(hp_d + 28 + 4 * offset));
        for (uint32_t i = L; 1 < i; i--) {
          offset = processDataElement(key, 5 * offset, *(const uint4*)(hp_d + offsets[i - 1] + 4 * offset));
        }
        out_d[index] = processMaskElement(key, 32 * offset, hp_d[offsets[0] + offset]);
      }
    }
  }


  template<uint32_t L>
  void runExtract(uint32_t* out_d, const std::vector<uint32_t>& offsets, const uint32_t* hp_d, uint32_t N, cudaStream_t stream)
  {
    OffsetBlob offsetBlob;
    std::memcpy(&offsetBlob.data0, offsets.data(), sizeof(uint32_t)*std::min(uint32_t(12), L));

    int minGridSize = 0;
    int blockSize = 0;
    auto rv = cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, extract<L>);
    assert(rv == cudaSuccess);

    auto blocks = std::min(minGridSize, int((N + blockSize - 1) / blockSize));
    ::extract<L><<<blocks, blockSize, 0, stream>>>(out_d, hp_d, offsetBlob);
  }

  void scratchLayout(std::vector<uint32_t>& levels, std::vector<uint32_t>& offsets, uint32_t N)
  {
    if (N == 0) return;
    
    levels.clear();
    levels.push_back((N + 31) / 32);
    while (125 < levels.back())
    {
      levels.push_back((levels.back() + 4) / 5);
    }
    
    offsets.resize(levels.size() + 4);

    uint32_t o = 0;
    offsets[levels.size()] = o;  // Apex
    o += 32 * 4;

    for (int i = static_cast<int>(levels.size()) - 1; 0 < i; i--) {
      offsets[i] = o; // HP level i
      o += 4 * levels[i];
    }

    // level zero
    offsets[0] = o;
    o += (levels[0] + 3) & ~3;

    offsets[levels.size() + 1] = o; // Large sideband buffer
    o += levels.empty() ? 0 : (levels[0] + 3) & ~3;

    offsets[levels.size() + 2] = o; // Small sideband buffer
    o += levels.size() < 2 ? 0 : (levels[1] + 3) & ~3;

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
    assert(false);
  }
  else {
    bool sb = false;

    size_t i = 0;
    if (1 < L) {
      ::reduceBase2b<<<(levels[1] + 31) / 32, 8 * 32, 0, stream>>>(scratch_d + offsets[1],
                                                                  scratch_d + offsets[L + 1 + (sb ? 1 : 0)],
                                                                  levels[1],
                                                                  scratch_d + offsets[0],
                                                                  levels[0],
                                                                  in_d,
                                                                  N);
      i += 2;
      sb = !sb;
    }
    else if (0 < L) {
      ::reduceBase << <(levels[0] + 3) / 4, 4 * 32, 0, stream >> > (scratch_d + offsets[0],
                                                                    scratch_d + offsets[L + 1 + (sb ? 1 : 0)],
                                                                    levels[0],
                                                                    in_d,
                                                                    N);
      i += 1;
      sb = !sb;
    }

    for (; i + 1 < L; i += 2) {
      ::reduce2 << <(levels[i + 1] + 31) / 32, 160, 0, stream >> > (reinterpret_cast<uint4*>(scratch_d + offsets[i + 1]),
                                                                    scratch_d + offsets[L + 1 + (sb ? 1 : 0)],
                                                                    levels[i + 1],
                                                                    reinterpret_cast<uint4*>(scratch_d + offsets[i]),
                                                                    levels[i],
                                                                    scratch_d + offsets[L + 1 + (sb ? 0 : 1)],
                                                                    levels[i - 1]);
      sb = !sb;

    }

    for (; i < L; i++) {
      ::reduce1<<<(levels[i] + 31)/32, 160, 0, stream>>>(reinterpret_cast<uint4*>(scratch_d + offsets[i]),
                                                         scratch_d + offsets[L + 1 + (sb ? 1 : 0)],
                                                         levels[i],
                                                         scratch_d + offsets[L + 1 + (sb ? 0 : 1)],
                                                         levels[i - 1]);
      sb = !sb;
    }
    ::reduceApex<false, true><<<1, 128, 0, stream>>>(reinterpret_cast<uint4*>(scratch_d),
                                                     sum_d,
                                                     scratch_d + offsets[L + 1 + (sb ? 0 : 1)],
                                                     levels[L - 1]);
  }



 

  // No readback, no dynamic parallelism approach: Create enough blocks s.t. all multiprocessors have enough warps,
  // but let problem size beyond this be handled by looping. 

  switch (L)
  {
  case 0: ::runExtract<0>(out_d, offsets, scratch_d, N, stream); break;
  case 1: ::runExtract<1>(out_d, offsets, scratch_d, N, stream); break;
  case 2: ::runExtract<2>(out_d, offsets, scratch_d, N, stream); break;
  case 3: ::runExtract<3>(out_d, offsets, scratch_d, N, stream); break;
  case 4: ::runExtract<4>(out_d, offsets, scratch_d, N, stream); break;
  case 5: ::runExtract<5>(out_d, offsets, scratch_d, N, stream); break;
  case 6: ::runExtract<6>(out_d, offsets, scratch_d, N, stream); break;
  case 7: ::runExtract<7>(out_d, offsets, scratch_d, N, stream); break;
  case 8: ::runExtract<8>(out_d, offsets, scratch_d, N, stream); break;
  case 9: ::runExtract<9>(out_d, offsets, scratch_d, N, stream); break;
  case 10: ::runExtract<10>(out_d, offsets, scratch_d, N, stream); break;
  case 11: ::runExtract<11>(out_d, offsets, scratch_d, N, stream); break;
  case 12: ::runExtract<12>(out_d, offsets, scratch_d, N, stream); break;
  default:
    abort();
    break;
  }

#if 0
  cudaStreamSynchronize(stream);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) {
    abort();
  }
#endif
}

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

    for (uint32_t i = 0; i < 20; i++) {
      const uint32_t offset0 = offset0_base + 256 * i;
      const uint32_t value = offset0 < n0 ? sb0_d[offset0] : 0;
      const uint32_t warpMask = __ballot(value != 0);
      if (lane == 0) {
        const uint32_t offset1 = offset1_base + 8 * i + warp;
        if (offset1 < n1) {
          hp1_d[offset1] = warpMask;
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
        ((uint4*)hp2_d)[offset2] = hp;
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

  template<bool mask_input, bool write_sum>
  __global__
  __launch_bounds__(128)
  void
  reduceApex(uint4* __restrict__ apex_d,
             uint32_t* sum_d,
             const uint32_t* in_d,
             uint32_t N,
             uint32_t offset0,
             uint32_t offset1,
             uint32_t offset2,
             uint32_t offset3,
             uint32_t offset4,
             uint32_t offset5,
             uint32_t offset6,
             uint32_t offset7,
             uint32_t offset8,
             uint32_t offset9,
             uint32_t offsetA,
             uint32_t offsetB,
             uint32_t offsetC)
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
    volatile __shared__ uint32_t sb[125 + 25];
    sb[threadIdx.x] = a;

    // Store 5x5 uint4's at uint4 offset 0 (25x4=100 elements, corresponding to 125 inputs).
    __syncthreads();

    if (threadIdx.x < 32) {

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
      //__syncthreads();
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
      //__syncthreads();
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
    else if (threadIdx.x == 32) {
      uint32_t * t = (uint32_t*)(apex_d + 32);

      if (offset0 != 0) t[0] = offset0;
      if (offset1 != 0) t[1] = offset1;
      if (offset2 != 0) t[2] = offset2;
      if (offset3 != 0) t[3] = offset3;
      if (offset4 != 0) t[4] = offset4;
      if (offset5 != 0) t[5] = offset5;
      if (offset6 != 0) t[6] = offset6;
      if (offset7 != 0) t[7] = offset7;
      if (offset8 != 0) t[8] = offset8;
      if (offset9 != 0) t[9] = offset9;
      if (offsetA != 0) t[10] = offsetA;
      if (offsetB != 0) t[11] = offsetB;
      if (offsetC != 0) t[12] = offsetC;
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

  __device__ __forceinline__ uint32_t _ldu(const uint32_t *ptr)
  {
    uint32_t rv;
    asm ("ldu.global.u32 %0, [%1];" : "=r"(rv) :
#if defined(__LP64__) || defined(_WIN64)
                  "l"(ptr)
#else
                  "r"(ptr)
#endif
    );
    return rv;
  }

  __device__ __forceinline__ uint4 _ldu(const uint4 *ptr)
  {
    uint4 rv;
    asm("ldu.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(rv.x), "=r"(rv.y), "=r"(rv.z), "=r"(rv.w) :
#if defined(__LP64__) || defined(_WIN64)
        "l"(ptr)
#else
        "r"(ptr)
#endif
    );
    return rv;
  }


  template<uint32_t L>
  __global__
  __launch_bounds__(128)
  void extract(uint32_t* __restrict__ out_d,
               const uint32_t* __restrict__ hp_d)
  {
    uint32_t N = hp_d[0];
    uint4 T = *((const uint4*)(hp_d + 4));

    for (uint32_t k = 4 * blockDim.x * blockIdx.x; k < N; k += 4 * gridDim.x * blockDim.x) {
      uint32_t index = k + 4 * threadIdx.x;
      if (index < N) {
        uint4 offset = make_uint4(0, 0, 0, 0);
        uint4 key = make_uint4(index, index + 1, index + 2, index + 3);
        if (N <= key.w) {
          key.y = min(key.y, N - 1);
          key.z = min(key.z, N - 1);
          key.w = min(key.w, N - 1);
        }

        offset.x = processHistoElement(key.x, 5 * offset.x, T);
        offset.y = processHistoElement(key.y, 5 * offset.y, T);
        offset.z = processHistoElement(key.z, 5 * offset.z, T);
        offset.w = processHistoElement(key.w, 5 * offset.w, T);

        offset.x = processHistoElement(key.x, 5 * offset.x, *(const uint4*)(hp_d + 8 + 4 * offset.x));
        offset.y = processHistoElement(key.y, 5 * offset.y, *(const uint4*)(hp_d + 8 + 4 * offset.y));
        offset.z = processHistoElement(key.z, 5 * offset.z, *(const uint4*)(hp_d + 8 + 4 * offset.z));
        offset.w = processHistoElement(key.w, 5 * offset.w, *(const uint4*)(hp_d + 8 + 4 * offset.w));

        offset.x = processHistoElement(key.x, 5 * offset.x, *(const uint4*)(hp_d + 28 + 4 * offset.x));
        offset.y = processHistoElement(key.y, 5 * offset.y, *(const uint4*)(hp_d + 28 + 4 * offset.y));
        offset.z = processHistoElement(key.z, 5 * offset.z, *(const uint4*)(hp_d + 28 + 4 * offset.z));
        offset.w = processHistoElement(key.w, 5 * offset.w, *(const uint4*)(hp_d + 28 + 4 * offset.w));

        for (uint32_t i = L; 1 < i; i--) {
          uint32_t offseti = *(hp_d + 32 * 4 + i - 1);
          offset.x = processDataElement(key.x, 5 * offset.x, *(const uint4*)(hp_d + offseti + 4 * offset.x));
          offset.y = processDataElement(key.y, 5 * offset.y, *(const uint4*)(hp_d + offseti + 4 * offset.y));
          offset.z = processDataElement(key.z, 5 * offset.z, *(const uint4*)(hp_d + offseti + 4 * offset.z));
          offset.w = processDataElement(key.w, 5 * offset.w, *(const uint4*)(hp_d + offseti + 4 * offset.w));
        }
        uint32_t offset0 = *(hp_d + 32 * 4);
        uint4 val;
        val.x = processMaskElement(key.x, 32 * offset.x, hp_d[offset0 + offset.x]);
        val.y = processMaskElement(key.y, 32 * offset.y, hp_d[offset0 + offset.y]);
        val.z = processMaskElement(key.z, 32 * offset.z, hp_d[offset0 + offset.z]);
        val.w = processMaskElement(key.w, 32 * offset.w, hp_d[offset0 + offset.w]);

        if (index + 3 < N) {
          *(uint4*)(out_d + index) = val;
        }
        else {
          out_d[index] = val.x;
          if (index + 1 < N) {
            out_d[index + 1] = val.y;
            if (index + 2 < N) {
              out_d[index + 2] = val.z;
            }
          }
        }
        //uint32_t * dst = out_d + index;
//#if 0
//        *dst = val;
//#else
//        asm("st.global.cs.u32 [%0], %1;" ::  "l"(dst), "r"(val));
//#endif

      }
    }
  }


  template<uint32_t L>
  void runExtract(uint32_t* out_d, const std::vector<uint32_t>& offsets, const uint32_t* hp_d, uint32_t N, cudaStream_t stream)
  {
    int minGridSize = 0;
    int blockSize = 0;
    auto rv = cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, extract<L>);
    //assert(rv == cudaSuccess);

    auto blocks = std::min(minGridSize, int((N + 4 * blockSize - 1) / (4 * blockSize)));
    ::extract<L><<<blocks, blockSize, 0, stream>>>(out_d, hp_d);
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
    offsets[levels.size()] = o;                 // Apex
    o += 32 * 4;                                // Size of apex
    o += static_cast<uint32_t>(levels.size());  // Space for level offsets
    o = (o + 3) & ~3;                           // Align to uvec4

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
      ::reduceBase2<<<(levels[1] + 31) / 32, 8 * 32, 0, stream>>>(scratch_d + offsets[1],
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
                                                     levels[L - 1],
                                                     0 < L ? offsets[0] : 0,
                                                     1 < L ? offsets[1] : 0,
                                                     2 < L ? offsets[2] : 0,
                                                     3 < L ? offsets[3] : 0,
                                                     4 < L ? offsets[4] : 0,
                                                     5 < L ? offsets[5] : 0,
                                                     6 < L ? offsets[6] : 0,
                                                     7 < L ? offsets[7] : 0,
                                                     8 < L ? offsets[8] : 0,
                                                     9 < L ? offsets[9] : 0,
                                                     10 < L ? offsets[10] : 0,
                                                     11 < L ? offsets[11] : 0,
                                                     12 < L ? offsets[12] : 0);
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

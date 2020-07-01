// This file is part of ComputeStuff copyright (C) 2020 Christopher Dyken.
// Released under the MIT license, please see LICENSE file for details.

#include "MC.h"
#include <cassert>
#include <cmath>

namespace {

  // Fetch 32[x] x 5[y] samples from the scalar field, used for the indexed
// baselevel buildup. A bit more care testing for valid Y so samples outside
// will have a GL_CLAMP-ish behaviour, which we rely on to not produce
// edge intersections (which produces vertices) outside of the domiain. 
  template<typename T>
  __device__ float2 fetch2(const T* ptr, const size_t stride, const uint32_t y, const uint32_t max_y, const float t)
  {
    float t0 = (ptr[0] < t) ? 1.f : 0.f; ptr += (y + 1 <= max_y) ? stride : 0;
    float t1 = (ptr[0] < t) ? 1.f : 0.f; ptr += (y + 2 <= max_y) ? stride : 0;
    float t2 = (ptr[0] < t) ? 1.f : 0.f; ptr += (y + 3 <= max_y) ? stride : 0;
    float t3 = (ptr[0] < t) ? 1.f : 0.f; ptr += (y + 4 <= max_y) ? stride : 0;
    float t4 = (ptr[0] < t) ? 1.f : 0.f; ptr += (y + 5 <= max_y) ? stride : 0;
    float t5 = (ptr[0] < t) ? 1.f : 0.f;

    // Simd-in-float3, packing 6 uint8 bitsets into 2 float32.
    float r0 = __fmaf_rn(float(1 << 16), t2, __fmaf_rn(float(1 << 8), t1, t0));
    float r1 = __fmaf_rn(float(1 << 16), t5, __fmaf_rn(float(1 << 8), t4, t3));
    return make_float2(r0, r1);
  }

  // Given a MC case (8-bits), create a mask for the (0,0,0)-(1,0,0), (0,0,0)-(0,1,0),
  // and (0,0,0)-(0,0,1) axes where a bit is set if the bits of the edge end-points in
  // the MC case are different (and thus edge pierces the iso-surface).
  //
  // This apporachs works by broadcasting the bit of (0,0,0) to all bits, and xor this
  // with the case, and mask out the relevant bits.
  __device__ __forceinline__ uint32_t piercingAxesFromCase(uint32_t c)
  {
    uint32_t a;
    asm("{\n"
        "  .reg .u32 t1, t2, t3;\n"
        "  bfe.s32 t1, %1, 0, 1;\n"     // Sign-extend bit 0 (0,0,0) over whole word
        "  xor.b32 t2, %1, t1;\n"       // XOR with (0,0,0) to see which corners change sign wrt (0,0,0).
        "  and.b32 %0, t2, 0b10110;\n"  // Mask out (1,0,0), (0,1,0) and (0,0,1).
        "}"
        : "=r"(a) : "r"(c));
    return a;
  }

  // Count the number of edge-intersections of edges belonging to that cell (0-3).
  // We just count the number of set bits in a bitmask.
  __device__ __forceinline__ uint32_t axisCountFromCase(const uint32_t c)
  {
    uint32_t a = piercingAxesFromCase(c);
    uint32_t n;
    asm("popc.b32 %0, %1;": "=r"(n) : "r"(a));         // Count number of 1's (should be 0..3).
    return n;
  }

  // Fetch samples from scalar-field, deduce MC cases for cells, and write out
  // first level ot the HistoPyramid and sideband.
  //
  // Since cells needs surrounding samples, 32 samples are fetched to produce
  // 31 cells. This wastes 1/32 of the processing, but yields pretty coalesced
  // fetches. Profiling shows a bit too many memory transactions, which I guess
  // is due to fetches not being aligned (first warp fetches values 0..31, next
  // warp fetches 31...62, etc, so requests gets split).
  //
  // The cell grid is decomposed into chunks of 32 x 5 x 5. The last x-value is,
  // as mentioned, discarded, so the extent is actually 31 x 5 x 5, but the
  // full set of values are written.
  //
  // I have tried (RTX2080 + CUDA 11) to just let the cache system handle this
  // and create a full set of 32 cells, but that was slower. Also tried to add
  // extra end-fetch for the last thread to get the missing samnple for a full
  // set of 32 cells, but that was slower as well.
  //
  // In addition to calculating the MC cell case, it also calculates the number
  // of unique vertices belong to that cell (0-3). This grid is one cell-layer
  // larger along X,Y and Z than the unindexed grid to have the simple relation-
  // ship of each cell having 0-3 vertices.
  //
  // But this requires careful masking to avoid producing extra vertices, we
  // basically implement GL_CLAMP-behavior.

  template<class FieldType, bool indexed>
  __device__ void reduceBaseKernel(uint8_t* __restrict__ index_case_ptr,
                                   uchar4* __restrict__ out_index_level0_ptr,
                                   uchar4* __restrict__ out_vertex_level0_ptr,
                                   uint32_t* __restrict__ out_index_sideband_ptr,
                                   uint32_t* __restrict__ out_vertex_sideband_ptr,
                                   const uint8_t* __restrict__   index_count,
                                   const FieldType* __restrict__ ptr,
                                   size_t field_row_stride,
                                   size_t field_slice_stride,
                                   uint3 cell,
                                   uint3 grid_max_index,
                                   unsigned thread,
                                   float threshold)
  {

    // First fetch initial z-slice, and then fetch 5 additional slices, to
    // produce 5 slices of cells.
    float2 prev = fetch2(ptr, field_row_stride, cell.y, grid_max_index.y, threshold);
    ptr += cell.z < grid_max_index.z ? field_slice_stride : 0;
    for (unsigned z = 0; z < 5; z++) {

      unsigned isum = 0;
      unsigned vsum = 0;
      bool zmask = cell.z <= grid_max_index.z;
      if (zmask) {
        float2 next = fetch2(ptr, field_row_stride, cell.y, grid_max_index.y, threshold);
        ptr += cell.z + 1 < grid_max_index.z ? field_slice_stride : 0;

        // Merge bits from previous and current slice.
        float2 t0 = make_float2(__fmaf_rn(float(1 << 4), next.x, prev.x),
                                __fmaf_rn(float(1 << 4), next.y, prev.y));

        prev = next;

        // Fetch case from x+1 using warp-shuffle. Last thread has garbage data,
        // but that result is masked out.
        float2 tt = make_float2(__shfl_down_sync(0xffffffff, t0.x, 1),
                                __shfl_down_sync(0xffffffff, t0.y, 1));

        if ((indexed ? cell.x <= grid_max_index.x : cell.x < grid_max_index.x) && thread < 31) {
          // Merge in results from the warp shuffle. 
          uint32_t g0 = static_cast<uint32_t>(__fmaf_rn(2.f, tt.x, t0.x));
          uint32_t g1 = static_cast<uint32_t>(__fmaf_rn(2.f, tt.y, t0.y));
          uint32_t s0 = __byte_perm(g0, g1, (4 << 12) | (2 << 8) | (1 << 4) | (0 << 0));
          g0 = g0 | (s0 >> 6);
          g1 = g1 | (g1 >> 6);

          // Extract cases for the 5-element y-column of cases that each thread has
          uint32_t case_y0 = __byte_perm(g0, 0, (4 << 12) | (4 << 8) | (4 << 4) | (0 << 0));
          uint32_t case_y1 = __byte_perm(g0, 0, (4 << 12) | (4 << 8) | (4 << 4) | (1 << 0));
          uint32_t case_y2 = __byte_perm(g0, 0, (4 << 12) | (4 << 8) | (4 << 4) | (2 << 0));
          uint32_t case_y3 = __byte_perm(g1, 0, (4 << 12) | (4 << 8) | (4 << 4) | (0 << 0));
          uint32_t case_y4 = __byte_perm(g1, 0, (4 << 12) | (4 << 8) | (4 << 4) | (1 << 0));

          // Look up index count. Set count for cells outside of domain to zero. Careful masking.
          uint32_t ic_y0 = (cell.x < grid_max_index.x) && (cell.z < grid_max_index.z) && ((cell.y + 0u) < grid_max_index.y) ? index_count[case_y0] : 0u;
          uint32_t ic_y1 = (cell.x < grid_max_index.x) && (cell.z < grid_max_index.z) && ((cell.y + 1u) < grid_max_index.y) ? index_count[case_y1] : 0u;
          uint32_t ic_y2 = (cell.x < grid_max_index.x) && (cell.z < grid_max_index.z) && ((cell.y + 2u) < grid_max_index.y) ? index_count[case_y2] : 0u;
          uint32_t ic_y3 = (cell.x < grid_max_index.x) && (cell.z < grid_max_index.z) && ((cell.y + 3u) < grid_max_index.y) ? index_count[case_y3] : 0u;
          uint32_t ic_y4 = (cell.x < grid_max_index.x) && (cell.z < grid_max_index.z) && ((cell.y + 4u) < grid_max_index.y) ? index_count[case_y4] : 0u;

          // Look up vertex count. Set count for cells outside of domain to zero.
          uint32_t vc_y0, vc_y1, vc_y2, vc_y3, vc_y4;
          if (indexed) {
            vc_y0 = axisCountFromCase(case_y0);
            vc_y1 = cell.y + 1 <= grid_max_index.y ? axisCountFromCase(case_y1) : 0;
            vc_y2 = cell.y + 2 <= grid_max_index.y ? axisCountFromCase(case_y2) : 0;
            vc_y3 = cell.y + 3 <= grid_max_index.y ? axisCountFromCase(case_y3) : 0;
            vc_y4 = cell.y + 4 <= grid_max_index.y ? axisCountFromCase(case_y4) : 0;
            vsum = vc_y0 + vc_y1 + vc_y2 + vc_y3 + vc_y4;
          }
          else {
            vc_y0 = 0;
            vc_y1 = 0;
            vc_y2 = 0;
            vc_y3 = 0;
            vc_y4 = 0;
            vsum = 0;
          }

          isum = ic_y0 + ic_y1 + ic_y2 + ic_y3 + ic_y4;
          if (isum | vsum) {
            index_case_ptr[5 * (32 * z) + 0] = case_y0;
            index_case_ptr[5 * (32 * z) + 1] = case_y1;
            index_case_ptr[5 * (32 * z) + 2] = case_y2;
            index_case_ptr[5 * (32 * z) + 3] = case_y3;
            index_case_ptr[5 * (32 * z) + 4] = case_y4;
            out_index_level0_ptr[32 * z] = make_uchar4(ic_y0,
                                                       ic_y0 + ic_y1,
                                                       ic_y0 + ic_y1 + ic_y2,
                                                       ic_y0 + ic_y1 + ic_y2 + ic_y3);
          }
          if (indexed && vsum) {
            out_vertex_level0_ptr[32 * z] = make_uchar4(vc_y0,
                                                        vc_y0 + vc_y1,
                                                        vc_y0 + vc_y1 + vc_y2,
                                                        vc_y0 + vc_y1 + vc_y2 + vc_y3);

          }

        }
      }
      cell.z++;
      // Store sideband data.
      out_index_sideband_ptr[32 * z] = isum;
      if (indexed) {
        out_vertex_sideband_ptr[32 * z] = vsum;
      }
    }
  }


  template<class FieldType>
  __global__ __launch_bounds__(32) void reduceBase(uint8_t* __restrict__           index_cases_d,
                                                   uchar4* __restrict__            out_index_level0_d,
                                                   uint32_t* __restrict__          out_index_sideband_d,
                                                   const uint8_t* __restrict__     index_count,
                                                   const FieldType* __restrict__   field_d,
                                                   const size_t                    field_row_stride,
                                                   const size_t                    field_slice_stride,
                                                   const float                     threshold,
                                                   const uint3                     chunks,
                                                   const uint3                     grid_max_index)
  {
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t thread = threadIdx.x % 32;
    const uint32_t chunk_ix = blockIdx.x + warp;

    uint3 chunk = make_uint3(chunk_ix % chunks.x,
                             (chunk_ix / chunks.x) % chunks.y,
                             (chunk_ix / chunks.x) / chunks.y);
    uint3 cell = make_uint3(31 * chunk.x + thread,
                            5 * chunk.y,
                            5 * chunk.z);

    reduceBaseKernel<FieldType, false>(index_cases_d        + static_cast<size_t>(5 * 32 * 5 * blockIdx.x + 5 * thread), // Index doesn't need more than 32 bits
                                       out_index_level0_d   + static_cast<size_t>(    32 * 5 * blockIdx.x +     thread),
                                       nullptr,
                                       out_index_sideband_d + static_cast<size_t>(    32 * 5 * blockIdx.x +     thread),
                                       nullptr,
                                       index_count,
                                       field_d + cell.z * field_slice_stride
                                               + cell.y * field_row_stride
                                               + min(grid_max_index.x, cell.x),
                                       field_row_stride,
                                       field_slice_stride,
                                       cell,
                                       grid_max_index,
                                       thread,
                                       threshold);

  }

  template<class FieldType>
  __global__ __launch_bounds__(32) void reduceBaseIndexed(uint8_t* __restrict__           index_cases_d,
                                                          uchar4* __restrict__            out_vertex_level0_d,
                                                          uint32_t* __restrict__          out_vertex_sideband_d,
                                                          uchar4* __restrict__            out_index_level0_d,
                                                          uint32_t* __restrict__          out_index_sideband_d,
                                                          const uint8_t* __restrict__     index_count,
                                                          const FieldType* __restrict__   field_d,
                                                          const size_t                    field_row_stride,
                                                          const size_t                    field_slice_stride,
                                                          const float                     threshold,
                                                          const uint3                     chunks,
                                                          const uint3                     grid_max_index)
  {
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t thread = threadIdx.x % 32;
    const uint32_t chunk_ix = blockIdx.x + warp;

    // Figure out which chunk we are in.
    // FIXME: Try to use grid extents to mach chunk grid to avoid all these modulo/divisions.
    uint3 chunk = make_uint3(chunk_ix % chunks.x,
                             (chunk_ix / chunks.x) % chunks.y,
                             (chunk_ix / chunks.x) / chunks.y);
    uint3 cell = make_uint3(31 * chunk.x + thread,
                            5 * chunk.y,
                            5 * chunk.z);

    reduceBaseKernel<FieldType, true>(index_cases_d         + static_cast<size_t>(5 * 32 * 5 * blockIdx.x + 5 * thread), // Index doesn't need more than 32 bits
                                      out_index_level0_d    + static_cast<size_t>(    32 * 5 * blockIdx.x +     thread),
                                      out_vertex_level0_d   + static_cast<size_t>(    32 * 5 * blockIdx.x +     thread),
                                      out_index_sideband_d  + static_cast<size_t>(    32 * 5 * blockIdx.x +     thread),
                                      out_vertex_sideband_d + static_cast<size_t>(32 * 5 * blockIdx.x + thread),
                                      index_count,
                                      field_d + cell.z * field_slice_stride
                                              + cell.y * field_row_stride
                                              + min(grid_max_index.x, cell.x),
                                      field_row_stride,
                                      field_slice_stride,
                                      cell,
                                      grid_max_index,
                                      thread,
                                      threshold);
  }

  



  // Reads 160 values, outputs HP level of 128 values, and 32 sideband values.
  __global__  __launch_bounds__(5 * 32) void reduce1(uint4* __restrict__          hp1_d,  //< Each block will write 32 uvec4's into this
                                                     uint32_t* __restrict__       sb1_d,  //< Each block will write 32 values into this.
                                                     const uint32_t               n1,     //< Number of uvec4's in hp1_d
                                                     const uint32_t* __restrict__ sb0_d,  //< Each block will read 5*32=160 values from here
                                                     const uint32_t               n0)     //< Number of elements in sb0_d
  {
    const uint32_t offset0 = 5 * 32 * blockIdx.x + threadIdx.x;

    // FIXME: Test idea, each warp reads 32 values. read instead 32/4 uint4's.
    __shared__ uint32_t sb[5 * 32];
    sb[threadIdx.x] = offset0 < n0 ? sb0_d[offset0] : 0;
    __syncthreads();
    if (threadIdx.x < 32) { // First warp
      const uint32_t offset1 = 32 * blockIdx.x + threadIdx.x;
      if (offset1 < n1) {
        uint4 hp = make_uint4(sb[5 * threadIdx.x + 0],
                              sb[5 * threadIdx.x + 1],
                              sb[5 * threadIdx.x + 2],
                              sb[5 * threadIdx.x + 3]);
        uint32_t sum = hp.x + hp.y + hp.z + hp.w + sb[5 * threadIdx.x + 4];
        if (sum) {
          hp1_d[offset1] = make_uint4(hp.x,
                                      hp.x + hp.y,
                                      hp.x + hp.y + hp.z,
                                      hp.x + hp.y + hp.z + hp.w);
        }
        sb1_d[offset1] = sum;
      }
    }
  }

  // Build 3 top levels (=apex), which are tiny.
  __global__ __launch_bounds__(128) void reduceApex(uint4* __restrict__ apex_d,
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
      *sum_d = s;
    }
  }



  __device__ __forceinline__ uint32_t processHP5Item(uint32_t& key, uint32_t offset, const uint4 item)
  {
    if (key < item.x) {
    }
    else if (key < item.y) {
      key -= item.x;
      offset += 1;
    }
    else if (key < item.z) {
      key -= item.y;
      offset += 2;
    }
    else if (key < item.w) {
      key -= item.z;
      offset += 3;
    }
    else {
      key -= item.w;
      offset += 4;
    }
    return offset;
  }

  struct hp5_result
  {
    uint32_t offset;
    uint32_t remainder;
  };

  __device__ hp5_result traverseDown(const uint4* __restrict__    pyramid,
                                     const uint32_t* __restrict__ level_offset,
                                     const uint32_t level_count,
                                     const uint32_t input_index)
  {
    uint32_t key = input_index;
    // Traverse apex
    uint32_t offset = 0;
    offset = processHP5Item(key, 0, pyramid[9]);
    offset = processHP5Item(key, 5 * offset, pyramid[10 + offset]);
    offset = processHP5Item(key, 5 * offset, pyramid[15 + offset]);
    for (unsigned l = level_count - 4; 0 < l; l--) {
      offset = processHP5Item(key, 5 * offset, pyramid[level_offset[l] + offset]);
    }
    uchar4 b = ((const uchar4*)(pyramid + level_offset[0]))[offset];
    offset = processHP5Item(key, 5 * offset, make_uint4(b.x, b.y, b.z, b.w));
    return { offset, key };
  }

  __device__ uint3 decodeCellPosition(const uint2 chunks, const uint32_t offset)
  {
    uint32_t q = offset / 800;

    uint3 chunk = make_uint3(q % chunks.x,
                             (q / chunks.x) % chunks.y,
                             (q / chunks.x) / chunks.y);

    uint32_t r = offset % 800;
    uint3 p0 = make_uint3(31 * chunk.x + ((r / 5) % 32),
                          5 * chunk.y + ((r % 5)),
                          5 * chunk.z + ((r / 5) / 32));
    return p0;
  }

  __device__ float4 sampleField(const float* __restrict__ field,
                                const uint3 field_offset,
                                const uint3 field_max_index,
                                const size_t field_row_stride,
                                const size_t field_slice_stride,
                                const uint3 i)
  {
    uint3 q = make_uint3(i.x + field_offset.x,
                         i.y + field_offset.y,
                         i.z + field_offset.z);
    bool less_x = q.x < field_max_index.x;
    bool less_y = q.y < field_max_index.y;
    bool less_z = q.z < field_max_index.y;

    q.x = less_x ? q.x : field_max_index.x;
    q.y = less_y ? q.y : field_max_index.y;
    q.z = less_z ? q.z : field_max_index.z;
    size_t o = q.x + q.y * field_row_stride + q.z * field_slice_stride;

    return make_float4(field[o + (less_x ? 1 : 0)],
                       field[o + (less_y ? field_row_stride : 0)],
                       field[o + (less_z ? field_slice_stride : 0)],
                       field[o]);
  }

  __device__ void emitVertexNormal(float* __restrict__ output,
                                   const float* __restrict__ field,
                                   const float3 scale,
                                   const uint3 field_offset,
                                   const uint3 field_max_index,
                                   const size_t field_row_stride,
                                   const size_t field_slice_stride,
                                   const float threshold,
                                   const uint3 i0,
                                   const uint3 i1)
  {
    float4 f0 = sampleField(field,
                            field_offset,
                            field_max_index,
                            field_row_stride,
                            field_slice_stride,
                            i0);

    float4 f1 = sampleField(field,
                            field_offset,
                            field_max_index,
                            field_row_stride,
                            field_slice_stride,
                            i1);

    float t = (threshold - f0.w) / (f1.w - f0.w);

    float nx = scale.x * ((1.f - t) * f0.x + t * f1.x - threshold);
    float ny = scale.y * ((1.f - t) * f0.y + t * f1.y - threshold);
    float nz = scale.z * ((1.f - t) * f0.z + t * f1.z - threshold);
    float nn = __frsqrt_rn(nx * nx + ny * ny + nz * nz);

    output[0] = scale.x * ((1.f - t) * i0.x + t * i1.x);
    output[1] = scale.y * ((1.f - t) * i0.y + t * i1.y);
    output[2] = scale.z * ((1.f - t) * i0.z + t * i1.z);
    output[3] = nn * nx;
    output[4] = nn * ny;
    output[5] = nn * nz;
  }


  __global__ void extractIndexedVertexPN(float* __restrict__          output,
                                         const uint4* __restrict__    pyramid,
                                         const float* __restrict__    field,
                                         const uint8_t* __restrict__  index_cases,
                                         const uint8_t* __restrict__  index_table,
                                         const size_t                 field_row_stride,
                                         const size_t                 field_slice_stride,
                                         const uint3                  field_offset,
                                         const uint3                  field_max_index,
                                         const uint32_t               output_count,
                                         const uint2                  chunks,
                                         const float3                 scale,
                                         const float                  threshold)
  {
    uint32_t ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix < output_count) {
      const uint32_t* level_offset = (const uint32_t*)(pyramid);

      hp5_result r = traverseDown(pyramid, level_offset, level_offset[15], ix);

      uint8_t index_case = index_cases[r.offset];
      uint3 p0 = decodeCellPosition(chunks, r.offset);

      uint32_t axes = piercingAxesFromCase(index_case);
      if (0 < r.remainder) axes = axes & (axes - 1);  // remove rightmost bit
      if (1 < r.remainder) axes = axes & (axes - 1);  // remove rightmost bit
      axes = axes & (-axes);                          // isolate rightmost bit

      uint3 p1 = p0;
      p1.x += (axes >> 1) & 1;
      p1.y += (axes >> 2) & 1;
      p1.z += (axes >> 4) & 1;

      emitVertexNormal(output + 6 * ix,
                       field,
                       scale,
                       field_offset,
                       field_max_index,
                       field_row_stride,
                       field_slice_stride,
                       threshold,
                       p0,
                       p1);
    }
  }

   __global__ void extractIndices(uint32_t* __restrict__       indices,
                                  const uint4* __restrict__    vertex_pyramid,
                                  const uint4* __restrict__    index_pyramid,
                                  const uint8_t* __restrict__  index_cases,
                                  const uint8_t* __restrict__  index_table,
                                  const uint3                  full_grid_size,  // Full size, including padding.
                                  const uint32_t               output_count,
                                  const uint2                  chunks)
  {
     // FIXME: no point in up-traversal if cell doesn't change.
     // FIXME: If extract 3 in a go, organize traversals to reduce up-traversal.
     // FIXME: Try to let a thread do all three indices of a triangle with shared down-traversal
     // FIXME: Try to recylce up-traversals, they might often be indentical (maybe check the table first to see if this is true)


    uint32_t ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix < output_count) {
      const uint32_t* level_offset = (const uint32_t*)(index_pyramid);
      const uint32_t level_count = level_offset[15];

      hp5_result r = traverseDown(index_pyramid, level_offset, level_count, ix);

      uint8_t index_case = index_cases[r.offset];
      uint8_t index_code = index_table[16u * index_case + r.remainder];

      // Find position within chunk
      uint32_t t1 = r.offset % 800;


      uint3 chunk_pos = make_uint3(((t1 / 5) % 32),
                                   ((t1 % 5)),
                                   ((t1 / 5) / 32));

      // Adjust linear offset
      //
      // Layout of each chunk:
      //
      // [z:5][x:32][y:5] (but x==31 is discarded, so x==30 jumps to next chunk).
      //
      // The chunks again is layed out as.
      //
      // FIXME: Check if matching chunk and in-chunk ordering simplifies stuff, x and 31 will be a mess, but maybe for the rest?
      //
      // [z][y][x].

      uint32_t vertex_offset = (r.offset +
                                ((index_code & (1 << 3)) ? (chunk_pos.x == 30 ? (800 - 30 * 5) : 5) : 0) +
                                ((index_code & (1 << 4)) ? (chunk_pos.y == 4 ? (800 * chunks.x - 4) : 1) : 0) +
                                ((index_code & (1 << 5)) ? (chunk_pos.z == 4 ? (800 * chunks.x * chunks.y - 4 * 5 * 32) : (5 * 32)): 0));

      uint32_t vertex_cell_case = index_cases[vertex_offset];
      uint32_t axes = piercingAxesFromCase(vertex_cell_case);

      int32_t vertex_index = 0;
      { // base level is uchar4, needs special treatment.
        uint32_t rem = vertex_offset % 5;
        vertex_offset = vertex_offset / 5;
        uchar4 item = ((const uchar4*)(vertex_pyramid + level_offset[0]))[vertex_offset];
        if (rem == 1) vertex_index += item.x;
        else if (rem == 2) vertex_index += item.y;
        else if (rem == 3) vertex_index += item.z;
        else if (rem == 4) vertex_index += item.w;
      }
      for (unsigned l = 1; l < level_count; l++) {
        uint32_t rem = vertex_offset % 5;
        vertex_offset = vertex_offset / 5;
        uint4 item = vertex_pyramid[level_offset[l] + vertex_offset];
        if (rem == 1) vertex_index += item.x;
        else if (rem == 2) vertex_index += item.y;
        else if (rem == 3) vertex_index += item.z;
        else if (rem == 4) vertex_index += item.w;
      }

      if (index_code & 0b001) {     // Vertex is on x-axis
        axes = 0b0000;              // That is the first vertex in that cell.
      }
      if (index_code & 0b010) {     // Vertex is on y-axis
        axes &= 0b0010;             // If cell has a vertex on the x-axis, that comes before the one on the y-axis.
      }
      if (index_code & 0b100) {     // Vertex is on z-axis
        axes &= 0b0110;             // If cell has vertices on either the x or y axes, those come before the one on the z-axis.
      }
      uint32_t n;
      asm("popc.b32 %0, %1;": "=r"(n) : "r"(axes));
      vertex_index += n;
      indices[ix] = vertex_index;
    }
  }


  __global__ void extractVertexPN(float* __restrict__          output,
                                  const uint4* __restrict__    pyramid,
                                  const float* __restrict__    field,
                                  const uint8_t* __restrict__  index_cases,
                                  const uint8_t* __restrict__  index_table,
                                  const size_t                 field_row_stride,
                                  const size_t                 field_slice_stride,
                                  const uint3                  field_offset,
                                  const uint3                  field_max_index,
                                  const uint32_t               output_count,
                                  const uint2                  chunks,
                                  const float3                 scale,
                                  const float                  threshold)
  {

    uint32_t ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix < output_count) {
      const uint32_t* level_offset = (const uint32_t*)(pyramid);

      hp5_result r = traverseDown(pyramid, level_offset, level_offset[15], ix);

      uint8_t index_case = index_cases[r.offset];
      uint8_t index_code = index_table[16u * index_case + r.remainder];
      uint3 p0 = decodeCellPosition(chunks, r.offset);
        
      p0.x += (index_code >> 3) & 1;
      p0.y += (index_code >> 4) & 1;
      p0.z += (index_code >> 5) & 1;
      uint3 p1 = p0;
      p1.x += (index_code >> 0) & 1;
      p1.y += (index_code >> 1) & 1;
      p1.z += (index_code >> 2) & 1;

      emitVertexNormal(output + 6 * ix,
                       field,
                       scale,
                       field_offset,
                       field_max_index,
                       field_row_stride,
                       field_slice_stride,
                       threshold,
                       p0,
                       p1);
    }
  }

  __global__ void runExtractionPN(float* __restrict__          vertex_output,
                                  uint32_t* __restrict__       index_output,
                                  const uint4* __restrict__    vertex_pyramid,
                                  const uint4* __restrict__    index_pyramid,
                                  const float* __restrict__    field,
                                  const uint8_t* __restrict__  index_cases,
                                  const uint8_t* __restrict__  index_table,
                                  const size_t                 field_row_stride,
                                  const size_t                 field_slice_stride,
                                  const uint3                  field_offset,
                                  const uint3                  field_size,
                                  const uint32_t               vertex_capacity,
                                  const uint32_t               index_capacity,
                                  const uint2                  chunks,
                                  const float3                 scale,
                                  const float                  threshold,
                                  bool                         alwaysExtract,
                                  bool                         indexed)
  {
    if (threadIdx.x == 0) {
      if (indexed) {
        uint32_t vertex_count_clamped = min(vertex_capacity, vertex_pyramid[8].x);
        // FIXME: Try to merge these two kernels as they are independent and can run on the
        //        same time.
        if (vertex_count_clamped) {
          extractIndexedVertexPN<<<(vertex_count_clamped + 255) / 256, 256>>>(vertex_output,
                                                                              vertex_pyramid,
                                                                              field,
                                                                              index_cases,
                                                                              index_table,
                                                                              field_row_stride,
                                                                              field_slice_stride,
                                                                              field_offset,
                                                                              make_uint3(field_size.x - 1,
                                                                                         field_size.y - 1,
                                                                                         field_size.z - 1),
                                                                              vertex_count_clamped,
                                                                              chunks,
                                                                              scale,
                                                                              threshold);
        }
        uint32_t index_count_clamped = min(index_capacity, index_pyramid[8].x);
        if(index_count_clamped) {
          extractIndices<<<(index_count_clamped + 255) / 256, 256>>>(index_output,
                                                                     vertex_pyramid,
                                                                     index_pyramid,
                                                                     index_cases,
                                                                     index_table,
                                                                     make_uint3(0, 0, 0),
                                                                     index_count_clamped,
                                                                     chunks);
        }
      }
      else {
        uint32_t vertex_count_clamped = min(vertex_capacity, index_pyramid[8].x);
        if (vertex_count_clamped) {
          extractVertexPN<<<(vertex_count_clamped + 255) / 256, 256>>>(vertex_output,
                                                                       index_pyramid,
                                                                       field,
                                                                       index_cases,
                                                                       index_table,
                                                                       field_row_stride,
                                                                       field_slice_stride,
                                                                       field_offset,
                                                                       make_uint3(field_size.x - 1,
                                                                                  field_size.y - 1,
                                                                                  field_size.z - 1),
                                                                       vertex_count_clamped,
                                                                       chunks,
                                                                       scale,
                                                                       threshold);
        }
      }
    }
  }


#define CHECKED_CUDA(a) do { cudaError_t error = (a); if(error != cudaSuccess) handleCudaError(error, __FILE__, __LINE__); } while(0)
  [[noreturn]]
  void handleCudaError(cudaError_t error, const char* file, int line)
  {
    fprintf(stderr, "%s@%d: CUDA: %s\n", file, line, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}


ComputeStuff::MC::Context* ComputeStuff::MC::createContext(const Tables* tables,
                                                           uint3 grid_size,
                                                           bool indexed,
                                                           cudaStream_t stream)
{
  // Minimal nonzero size, one chunk, 5 levels
  // [0]       40      160  (640)
  // [1]      200       32  (128)
  // [2]       15        7  (28)
  // [3]       10        2  (8)
  // [4]        9        1  (4)

  //assert(tables);
  auto* ctx = new Context();
  assert(0 < grid_size.x);
  assert(0 < grid_size.y);
  assert(0 < grid_size.z);

  ctx->tables = tables;
  ctx->grid_size = grid_size;
  ctx->indexed = indexed;

  // Each chunk handles a set of 32 x 5 x 5 cells.
  ctx->chunks = make_uint3(((grid_size.x - (indexed ? 0 : 1) + 30) / 31),
                           ((grid_size.y - (indexed ? 0 : 1) + 4) / 5),
                           ((grid_size.z - (indexed ? 0 : 1) + 4) / 5));
  assert(0 < ctx->chunks.x);
  assert(0 < ctx->chunks.y);
  assert(0 < ctx->chunks.z);
  assert(31 * (ctx->chunks.x - 1) < ctx->grid_size.x);  // cell (0,0,0) has  at least corner (0,0,0) inside field
  assert(5 * (ctx->chunks.y - 1) < ctx->grid_size.y);
  assert(5 * (ctx->chunks.z - 1) < ctx->grid_size.z);

  if (ctx->indexed) {
    assert(ctx->grid_size.x <= 31 * ctx->chunks.x);  // make sure grid is large enough
    assert(ctx->grid_size.y <= 5 * ctx->chunks.y);
    assert(ctx->grid_size.z <= 5 * ctx->chunks.z);
  }
  else {
    assert(ctx->grid_size.x <= 31 * ctx->chunks.x + 1);  // make sure grid is large enough
    assert(ctx->grid_size.y <= 5 * ctx->chunks.y+ 1);
    assert(ctx->grid_size.z <= 5 * ctx->chunks.z + 1);
  }

  ctx->chunk_total = ctx->chunks.x * ctx->chunks.y * ctx->chunks.z;
#if 0
  fprintf(stderr, "Grid size [%u x %u x %u]\n", ctx->grid_size.x, ctx->grid_size.y, ctx->grid_size.z);
  fprintf(stderr, "Chunks [%u x %u x %u] (= %u) cover=[%u x %u x %u]\n",
          ctx->chunks.x, ctx->chunks.y, ctx->chunks.z, ctx->chunk_total,
          31 * ctx->chunks.x, 5 * ctx->chunks.y, 5 * ctx->chunks.z);
#endif

  // Pyramid base level, as number of uvec4's:
  ctx->level_sizes[0] = (800 * ctx->chunk_total + 4) / 5;
  ctx->levels = 1 + static_cast<uint32_t>(std::ceil(std::log(ctx->level_sizes[0]) / std::log(5.0)));
  assert(4 <= ctx->levels);
  assert(ctx->levels < 16u); // 5^14 = 6103515625

  ctx->level_offsets[0] = 4 + 32; // First, level offsets (16 uint32's = 4 uvec4's), then pyramid apex (32 uvec4's).
  for (unsigned l = 1; l < ctx->levels - 3; l++) {
    ctx->level_sizes[l] = (ctx->level_sizes[l - 1] + 4) / 5;
    uint32_t prev_level_size = l == 1 ? (ctx->level_sizes[0] + 3) / 4 : ctx->level_sizes[0];  // Base-level is only uchar4.
    ctx->level_offsets[l] = ctx->level_offsets[l - 1] + prev_level_size;
  }
  ctx->total_size = ctx->level_offsets[ctx->levels - 4] + ctx->level_sizes[ctx->levels - 4];

  for (unsigned l = ctx->levels - 3; l < ctx->levels; l++) {
    ctx->level_sizes[l] = (ctx->level_sizes[l - 1] + 4) / 5;
  }
  assert(25 < ctx->level_offsets[ctx->levels - 4]);
  assert(ctx->level_offsets[ctx->levels - 3] <= 25);
  ctx->level_offsets[ctx->levels - 3] = 8 + 7; // up to 25 uvec4's
  ctx->level_offsets[ctx->levels - 2] = 8 + 2; // up to 5 uvec4's
  ctx->level_offsets[ctx->levels - 1] = 8 + 1; // one uvec4
  ctx->level_offsets[15] = ctx->levels;         // Store # levels in top entry.

  size_t sideband0_size = sizeof(uint32_t) * ctx->level_sizes[0];
  size_t sideband1_size = sizeof(uint32_t) * ctx->level_sizes[1];

  CHECKED_CUDA(cudaMalloc(&ctx->index_cases_d, sizeof(uint32_t) * 800 * ctx->chunk_total));
  CHECKED_CUDA(cudaMalloc(&ctx->index_pyramid, sizeof(uint4) * ctx->total_size));
  CHECKED_CUDA(cudaMalloc(&ctx->index_sidebands[0], sideband0_size));
  CHECKED_CUDA(cudaMalloc(&ctx->index_sidebands[1], sideband1_size));

  CHECKED_CUDA(cudaMemsetAsync(ctx->index_pyramid, 1, sizeof(uint4) * ctx->total_size, stream));
  CHECKED_CUDA(cudaMemsetAsync(ctx->index_sidebands[0], 1, sideband0_size, stream));
  CHECKED_CUDA(cudaMemsetAsync(ctx->index_sidebands[1], 1, sideband1_size, stream));


  CHECKED_CUDA(cudaMemcpyAsync(ctx->index_pyramid, ctx->level_offsets, sizeof(Context::level_offsets), cudaMemcpyHostToDevice, stream));

  if (indexed) {
    CHECKED_CUDA(cudaStreamCreateWithFlags(&ctx->indexStream, cudaStreamNonBlocking));
    CHECKED_CUDA(cudaEventCreateWithFlags(&ctx->baseEvent, cudaEventDisableTiming));
    CHECKED_CUDA(cudaEventCreateWithFlags(&ctx->indexDoneEvent, cudaEventDisableTiming));

    CHECKED_CUDA(cudaMalloc(&ctx->vertex_cases_d, sizeof(uint32_t) * 800 * ctx->chunk_total));
    CHECKED_CUDA(cudaMalloc(&ctx->vertex_pyramid, sizeof(uint4) * ctx->total_size));
    CHECKED_CUDA(cudaMalloc(&ctx->vertex_sidebands[0], sideband0_size));
    CHECKED_CUDA(cudaMalloc(&ctx->vertex_sidebands[1], sideband1_size));

    CHECKED_CUDA(cudaMemsetAsync(ctx->vertex_pyramid, 1, sizeof(uint4) * ctx->total_size, stream));
    CHECKED_CUDA(cudaMemsetAsync(ctx->vertex_sidebands[0], 1, sideband0_size, stream));
    CHECKED_CUDA(cudaMemsetAsync(ctx->vertex_sidebands[1], 1, sideband1_size, stream));

    CHECKED_CUDA(cudaMemcpyAsync(ctx->vertex_pyramid, ctx->level_offsets, sizeof(Context::level_offsets), cudaMemcpyHostToDevice, stream));
  }

#if 0
  for (unsigned l = 0; l < ctx->levels; l++) {
    fprintf(stderr, "[%d] %8d %8d  (%8d)\n", l, ctx->level_offsets[l], ctx->level_sizes[l], 4 * ctx->level_sizes[l]);
  }
  fprintf(stderr, "Total %d, levels %d \n", ctx->total_size, ctx->levels);
#endif

  CHECKED_CUDA(cudaHostAlloc(&ctx->sum_h, 2 * sizeof(uint32_t), cudaHostAllocMapped));
  CHECKED_CUDA(cudaHostGetDevicePointer(&ctx->sum_d, ctx->sum_h, 0));

  return ctx;
}

void ComputeStuff::MC::freeContext(Context* ctx, cudaStream_t stream)
{
  CHECKED_CUDA(cudaStreamSynchronize(stream));
  if (ctx == nullptr) return;

  if (ctx->index_cases_d) { CHECKED_CUDA(cudaFree(ctx->index_cases_d)); ctx->index_cases_d = nullptr; }
  if (ctx->index_pyramid) { CHECKED_CUDA(cudaFree(ctx->index_pyramid)); ctx->index_pyramid = nullptr; }
  if (ctx->index_sidebands[0]) { CHECKED_CUDA(cudaFree(ctx->index_sidebands[0])); ctx->index_sidebands[0] = nullptr; }
  if (ctx->index_sidebands[0]) { CHECKED_CUDA(cudaFree(ctx->index_sidebands[1])); ctx->index_sidebands[1] = nullptr; }

  if (ctx->index_cases_d) { CHECKED_CUDA(cudaFree(ctx->vertex_cases_d)); ctx->vertex_cases_d = nullptr; }
  if (ctx->index_pyramid) { CHECKED_CUDA(cudaFree(ctx->vertex_pyramid)); ctx->vertex_pyramid = nullptr; }
  if (ctx->index_sidebands[0]) { CHECKED_CUDA(cudaFree(ctx->vertex_sidebands[0])); ctx->vertex_sidebands[0] = nullptr; }
  if (ctx->index_sidebands[0]) { CHECKED_CUDA(cudaFree(ctx->vertex_sidebands[1])); ctx->vertex_sidebands[1] = nullptr; }

  if (ctx->sum_d) { CHECKED_CUDA(cudaFreeHost(ctx->sum_d)); ctx->sum_d = nullptr; }

  if (ctx->baseEvent)  CHECKED_CUDA(cudaEventDestroy(ctx->baseEvent));
  if (ctx->indexDoneEvent) CHECKED_CUDA(cudaEventDestroy(ctx->indexDoneEvent));
  if (ctx->indexStream) CHECKED_CUDA(cudaStreamDestroy(ctx->indexStream));

  delete ctx;
}

void ComputeStuff::MC::getCounts(Context* ctx, uint32_t* vertices, uint32_t* indices, cudaStream_t stream)
{
  // When reduce_apex kernel finishes, ctx->sum_h[0] contains the number of vertices. Just
  // sync'ing on the stream further down is slightly faster than adding an event
  // here and blocking on it in here. 
  CHECKED_CUDA(cudaStreamSynchronize(stream));
  *vertices = ctx->indexed ? ctx->sum_h[1] : ctx->sum_h[0];
  *indices = ctx->indexed ? ctx->sum_h[0] : 0;
}


void ComputeStuff::MC::buildPN(Context* ctx,
                               float* vertex_buffer,
                               uint32_t* index_buffer,
                               size_t vertex_buffer_bytesize,
                               size_t index_buffer_bytesize,
                               size_t field_row_stride,
                               size_t field_slice_stride,
                               uint3 field_offset,
                               uint3 field_size,
                               const float* field_d,
                               const float threshold,
                               cudaStream_t stream,
                               bool buildPyramid = true,
                               bool alwaysExtract = true)
{
  if (buildPyramid) {
    // Indexed pyramid buildup
    if (ctx->indexed) {
      // FIXME: Try to add extra reduction passes on the tail of this kernel (2 or 3 levels in one go
      //        have worked well before) and see if that reduces memory pressure.
      ::reduceBaseIndexed<float> << <ctx->chunk_total, 32, 0, stream >> > (ctx->index_cases_d,                   // block writes 5*5*32=800 uint8's
                                                                           (uchar4*)(ctx->vertex_pyramid + ctx->level_offsets[0]), // block writes 5*32 uvec4's
                                                                           ctx->vertex_sidebands[0],                    // block writes 5*32 uint32's
                                                                           (uchar4*)(ctx->index_pyramid + ctx->level_offsets[0]), // block writes 5*32 uvec4's
                                                                           ctx->index_sidebands[0],                    // block writes 5*32 uint32's
                                                                           ctx->tables->index_count,
                                                                           field_d,
                                                                           size_t(field_size.x),
                                                                           size_t(field_size.x) * field_size.y,
                                                                           threshold,
                                                                           ctx->chunks,
                                                                           make_uint3(ctx->grid_size.x - 1,
                                                                                      ctx->grid_size.y - 1,
                                                                                      ctx->grid_size.z - 1));

      CHECKED_CUDA(cudaEventRecord(ctx->baseEvent, stream));
      CHECKED_CUDA(cudaStreamWaitEvent(ctx->indexStream, ctx->baseEvent, 0));

      // FIXME: Try to run the vertex pyramid in a separate stream. Sync may be too costly though, and in that
      //        case it might be better to just merge the two launches and use e.g. block.y to tell which
      //        pyramid to reduce.
      bool sb = true;
      for (unsigned i = 1; i < ctx->levels - 3; i++) {
        const auto blocks = (ctx->level_sizes[i] + 31) / 32;
        reduce1 << <blocks, 5 * 32, 0, stream >> > (ctx->index_pyramid + ctx->level_offsets[i],    // Each block will write 32 uvec4's into this
                                                    ctx->index_sidebands[sb ? 1 : 0],              // Each block will write 32 uint32's into this
                                                    ctx->level_sizes[i],                           // Number of uvec4's in level i
                                                    ctx->index_sidebands[sb ? 0 : 1],              // Input, each block will read 5*32=160 values from here
                                                    ctx->level_sizes[i - 1]);                      // Number of sideband elements from level i-1
        // FIXME: Try to combine these two kernels into one, as they are independent.
        reduce1 << <blocks, 5 * 32, 0, ctx->indexStream >> > (ctx->vertex_pyramid + ctx->level_offsets[i], // Each block will write 32 uvec4's into this
                                                    ctx->vertex_sidebands[sb ? 1 : 0],           // Each block will write 32 uint32's into this
                                                    ctx->level_sizes[i],                         // Number of uvec4's in level i
                                                    ctx->vertex_sidebands[sb ? 0 : 1],           // Input, each block will read 5*32=160 values from here
                                                    ctx->level_sizes[i - 1]);                    // Number of sideband elements from level i-1
        sb = !sb;
      }
      reduceApex << <1, 4 * 32, 0, stream >> > (ctx->index_pyramid + 8,
                                                ctx->sum_d,
                                                ctx->index_sidebands[sb ? 0 : 1],
                                                ctx->level_sizes[ctx->levels - 4]);
      // FIXME: Try to combine these two kernels into one, as they are independent.
      reduceApex << <1, 4 * 32, 0, ctx->indexStream >> > (ctx->vertex_pyramid + 8,
                                                ctx->sum_d + 1,
                                                ctx->vertex_sidebands[sb ? 0 : 1],
                                                ctx->level_sizes[ctx->levels - 4]);


      CHECKED_CUDA(cudaEventRecord(ctx->indexDoneEvent, ctx->indexStream));
      CHECKED_CUDA(cudaStreamWaitEvent(stream, ctx->indexDoneEvent, 0));
    }

    // Non-indexed pyramid buildup
    else {
      ::reduceBase<float> << <ctx->chunk_total, 32, 0, stream >> > (ctx->index_cases_d,                   // block writes 5*5*32=800 uint8's
                                                                    (uchar4*)(ctx->index_pyramid + ctx->level_offsets[0]), // block writes 5*32 uvec4's
                                                                    ctx->index_sidebands[0],                    // block writes 5*32 uint32's
                                                                    ctx->tables->index_count,
                                                                    field_d,
                                                                    size_t(field_size.x),
                                                                    size_t(field_size.x) * field_size.y,
                                                                    threshold,
                                                                    ctx->chunks,
                                                                    make_uint3(ctx->grid_size.x - 1,
                                                                               ctx->grid_size.y - 1,
                                                                               ctx->grid_size.z - 1));

      bool sb = true;
      for (unsigned i = 1; i < ctx->levels - 3; i++) {
        const auto blocks = (ctx->level_sizes[i] + 31) / 32;
        reduce1 << <blocks, 5 * 32, 0, stream >> > (ctx->index_pyramid + ctx->level_offsets[i],    // Each block will write 32 uvec4's into this
                                                    ctx->index_sidebands[sb ? 1 : 0],              // Each block will write 32 uint32's into this
                                                    ctx->level_sizes[i],                           // Number of uvec4's in level i
                                                    ctx->index_sidebands[sb ? 0 : 1],              // Input, each block will read 5*32=160 values from here
                                                    ctx->level_sizes[i - 1]);                      // Number of sideband elements from level i-1
        sb = !sb;
      }
      reduceApex << <1, 4 * 32, 0, stream >> > (ctx->index_pyramid + 8,
                                                ctx->sum_d,
                                                ctx->index_sidebands[sb ? 0 : 1],
                                                ctx->level_sizes[ctx->levels - 4]);
    }

  }
  if (vertex_buffer) {
    ::runExtractionPN<<<1, 32, 0, stream>>>(vertex_buffer,
                                            index_buffer,
                                            ctx->vertex_pyramid,
                                            ctx->index_pyramid,
                                            field_d,
                                            ctx->index_cases_d,
                                            ctx->tables->index_table,
                                            field_row_stride,
                                            field_slice_stride,
                                            field_offset,
                                            field_size,
                                            static_cast<uint32_t>(vertex_buffer_bytesize / (6 * sizeof(float))),
                                            static_cast<uint32_t>(index_buffer_bytesize / sizeof(uint32_t)),
                                            make_uint2(ctx->chunks.x, ctx->chunks.y),
                                            make_float3(1.f / field_size.x,
                                                        1.f / field_size.y,
                                                        1.f / field_size.z),
                                            threshold,
                                            alwaysExtract,
                                            ctx->indexed);
  }
}

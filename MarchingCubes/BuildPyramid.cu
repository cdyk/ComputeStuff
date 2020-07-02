// This file is part of ComputeStuff copyright (C) 2020 Christopher Dyken.
// Released under the MIT license, please see LICENSE file for details.

#include "MC.h"
#include <cassert>
#include <cmath>

namespace {

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

  template<class FieldType, bool indexed, bool do_sync>
  __device__ void reduceBaseKernelMultiWarp(uint8_t* __restrict__ index_case_ptr,
                                            uchar4* __restrict__ out_index_level0_ptr,
                                            uchar4* __restrict__ out_vertex_level0_ptr,
                                            uint32_t* __restrict__ out_index_sideband_ptr,
                                            uint32_t* __restrict__ out_vertex_sideband_ptr,
                                            float* __restrict__ shmem,   // 32 * 6 * 2
                                            const uint8_t* __restrict__   index_count,
                                            const FieldType* __restrict__ ptr,
                                            size_t field_row_stride,
                                            size_t field_slice_stride,
                                            uint3 cell,
                                            uint3 grid_max_index,
                                            uint32_t warp,      // 0..5
                                            uint32_t thread,    // 0..31
                                            float threshold)
  {
    float2 mask000 = make_float2(0.f, 0.f);
    cell.z += warp;
    if (cell.z <= grid_max_index.z) {
      mask000 = fetch2(ptr + warp * field_slice_stride, field_row_stride, cell.y, grid_max_index.y, threshold);
    }
    shmem[0 * 32 * 6 + 32 * warp + thread] = mask000.x;
    shmem[1 * 32 * 6 + 32 * warp + thread] = mask000.y;
    __syncthreads();

    unsigned isum = 0;
    unsigned vsum = 0;
    float2 t0;
    if (warp < 5) {
      if (cell.z <= grid_max_index.z) {
        t0 = make_float2(__fmaf_rn(float(1 << 4), shmem[0 * 32 * 6 + 32 * (warp + 1) + thread], mask000.x),
                         __fmaf_rn(float(1 << 4), shmem[1 * 32 * 6 + 32 * (warp + 1) + thread], mask000.y));
      }
    }
    if (do_sync) {
      __syncthreads();
    }
    if (warp < 5) {
      if (cell.z <= grid_max_index.z) {

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
            index_case_ptr[5 * (32 * warp) + 0] = case_y0;
            index_case_ptr[5 * (32 * warp) + 1] = case_y1;
            index_case_ptr[5 * (32 * warp) + 2] = case_y2;
            index_case_ptr[5 * (32 * warp) + 3] = case_y3;
            index_case_ptr[5 * (32 * warp) + 4] = case_y4;
            out_index_level0_ptr[32 * warp] = make_uchar4(ic_y0,
                                                          ic_y0 + ic_y1,
                                                          ic_y0 + ic_y1 + ic_y2,
                                                          ic_y0 + ic_y1 + ic_y2 + ic_y3);
          }
          if (indexed && vsum) {
            out_vertex_level0_ptr[32 * warp] = make_uchar4(vc_y0,
                                                           vc_y0 + vc_y1,
                                                           vc_y0 + vc_y1 + vc_y2,
                                                           vc_y0 + vc_y1 + vc_y2 + vc_y3);

          }
        }

      }

      out_index_sideband_ptr[32 * warp] = isum;
      if (indexed) {
        out_vertex_sideband_ptr[32 * warp] = vsum;
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

    reduceBaseKernel<FieldType, false>(index_cases_d + static_cast<size_t>(5 * 32 * 5 * blockIdx.x + 5 * thread), // Index doesn't need more than 32 bits
                                       out_index_level0_d + static_cast<size_t>(32 * 5 * blockIdx.x + thread),
                                       nullptr,
                                       out_index_sideband_d + static_cast<size_t>(32 * 5 * blockIdx.x + thread),
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

    reduceBaseKernel<FieldType, true>(index_cases_d + static_cast<size_t>(5 * 32 * 5 * blockIdx.x + 5 * thread), // Index doesn't need more than 32 bits
                                      out_index_level0_d + static_cast<size_t>(32 * 5 * blockIdx.x + thread),
                                      out_vertex_level0_d + static_cast<size_t>(32 * 5 * blockIdx.x + thread),
                                      out_index_sideband_d + static_cast<size_t>(32 * 5 * blockIdx.x + thread),
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

  template<class FieldType>
  __global__ __launch_bounds__(32 * 6) void reduceBaseIndexedMultiWarp(uint8_t* __restrict__           index_cases_d,
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
    const uint32_t chunk_ix = blockIdx.x;

    // Figure out which chunk we are in.
    // FIXME: Try to use grid extents to mach chunk grid to avoid all these modulo/divisions.
    uint3 chunk = make_uint3(chunk_ix % chunks.x,
                             (chunk_ix / chunks.x) % chunks.y,
                             (chunk_ix / chunks.x) / chunks.y);
    uint3 cell = make_uint3(31 * chunk.x + thread,
                            5 * chunk.y,
                            5 * chunk.z);

    __shared__ float shmem[32 * 6 * 2];

    reduceBaseKernelMultiWarp<FieldType, true, false>(index_cases_d + static_cast<size_t>(5 * 32 * 5 * blockIdx.x + 5 * thread), // Index doesn't need more than 32 bits
                                                      out_index_level0_d + static_cast<size_t>(32 * 5 * blockIdx.x + thread),
                                                      out_vertex_level0_d + static_cast<size_t>(32 * 5 * blockIdx.x + thread),
                                                      out_index_sideband_d + static_cast<size_t>(32 * 5 * blockIdx.x + thread),
                                                      out_vertex_sideband_d + static_cast<size_t>(32 * 5 * blockIdx.x + thread),
                                                      shmem,
                                                      index_count,
                                                      field_d + cell.z * field_slice_stride
                                                      + cell.y * field_row_stride
                                                      + min(grid_max_index.x, cell.x),
                                                      field_row_stride,
                                                      field_slice_stride,
                                                      cell,
                                                      grid_max_index,
                                                      warp,
                                                      thread,
                                                      threshold);
  }


  template<class FieldType>
  __global__ __launch_bounds__(32 * 6) void reduceBaseIndexedMultiWarp2(uint8_t* __restrict__           index_cases_d,
                                                                        uchar4* __restrict__            out_vertex_level0_d,
                                                                        uint4* __restrict__            out_vertex_level1_d,
                                                                        uint32_t* __restrict__          out_vertex_sideband_d,
                                                                        uchar4* __restrict__            out_index_level0_d,
                                                                        uint4* __restrict__            out_index_level1_d,
                                                                        uint32_t* __restrict__          out_index_sideband_d,
                                                                        const uint8_t* __restrict__     index_count,
                                                                        const FieldType* __restrict__   field_d,
                                                                        const size_t                    field_row_stride,
                                                                        const size_t                    field_slice_stride,
                                                                        const float                     threshold,
                                                                        const uint3                     chunks,
                                                                        const uint32_t                  n0,
                                                                        const uint32_t                  n1,
                                                                        const uint3                     grid_max_index)
  {
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t thread = threadIdx.x % 32;
    const uint32_t chunk_ix = blockIdx.x;

    uint3 chunk = make_uint3(chunk_ix % chunks.x,
                             (chunk_ix / chunks.x) % chunks.y,
                             (chunk_ix / chunks.x) / chunks.y);
    uint3 cell = make_uint3(31 * chunk.x + thread,
                            5 * chunk.y,
                            5 * chunk.z);

    __shared__ float shmem[32 * 6 * 2];
    __shared__ uint32_t vtx_sideband[6 * 32]; // If we sync in reduceBaseKernel after fetching adjacent Z, we can recycle shmem
    __shared__ uint32_t idx_sideband[6 * 32];
    reduceBaseKernelMultiWarp<FieldType, true, false>(index_cases_d + static_cast<size_t>(5 * 32 * 5 * blockIdx.x + 5 * thread), // Index doesn't need more than 32 bits
                                                      out_index_level0_d + static_cast<size_t>(32 * 5 * blockIdx.x + thread),
                                                      out_vertex_level0_d + static_cast<size_t>(32 * 5 * blockIdx.x + thread),
                                                      idx_sideband + thread,
                                                      vtx_sideband + thread,
                                                      shmem,
                                                      index_count,
                                                      field_d + cell.z * field_slice_stride
                                                      + cell.y * field_row_stride
                                                      + min(grid_max_index.x, cell.x),
                                                      field_row_stride,
                                                      field_slice_stride,
                                                      cell,
                                                      grid_max_index,
                                                      warp,
                                                      thread,
                                                      threshold);
    __syncthreads();

    // Run two warps with remaining reduction
    if (warp < 2) {
      auto* sbi = warp == 0 ? idx_sideband : vtx_sideband;
      auto* hpo = warp == 0 ? out_index_level1_d : out_vertex_level1_d;
      auto* sbo = warp == 0 ? out_index_sideband_d : out_vertex_sideband_d;
      const uint32_t offset1 = 32 * blockIdx.x + thread;
      if (offset1 < n1) {
        uint4 hp = make_uint4(sbi[5 * thread + 0],
                              sbi[5 * thread + 1],
                              sbi[5 * thread + 2],
                              sbi[5 * thread + 3]);
        uint32_t sum = hp.x + hp.y + hp.z + hp.w + sbi[5 * thread + 4];
        if (sum) {
          hpo[offset1] = make_uint4(hp.x,
                                    hp.x + hp.y,
                                    hp.x + hp.y + hp.z,
                                    hp.x + hp.y + hp.z + hp.w);
        }
        sbo[offset1] = sum;
      }
    }

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

  [[noreturn]]
  void handleCudaError(cudaError_t error, const char* file, int line)
  {
    fprintf(stderr, "%s@%d: CUDA: %s\n", file, line, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
#define CHECKED_CUDA(a) do { cudaError_t error = (a); if(error != cudaSuccess) handleCudaError(error, __FILE__, __LINE__); } while(0)
}


void ComputeStuff::MC::Internal::buildPyramid(Context* ctx,
                                              size_t field_row_stride,
                                              size_t field_slice_stride,
                                              uint3 field_offset,
                                              uint3 field_size,
                                              const float* field_d,
                                              const float threshold,
                                              cudaStream_t stream)
{
  uint3 grid_max_index = make_uint3(ctx->grid_size.x - 1,
                                    ctx->grid_size.y - 1,
                                    ctx->grid_size.z - 1);
  // Indexed pyramid buildup
  if (ctx->indexed) {

    unsigned next_level = ~0u;
    switch (ctx->build_mode) {
    case BaseLevelBuildMode::SingleLevelSingleWarp: {
      uint32_t bn = ctx->chunk_total;
      uint32_t tn = 32;
      next_level = 1;
      reduceBaseIndexed<float><<<bn, tn, 0, stream>>>(ctx->index_cases_d,                   // block writes 5*5*32=800 uint8's
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
                                                      grid_max_index);
      break;
    }
    case BaseLevelBuildMode::SingleLevelMultiWarp: {
      uint32_t bn = ctx->chunk_total;
      uint32_t tn = 32 * 6;
      next_level = 1;
      reduceBaseIndexedMultiWarp<float><<<bn, tn, 0, stream>>>(ctx->index_cases_d,                   // block writes 5*5*32=800 uint8's
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
                                                               grid_max_index);
      break;
    }
    case BaseLevelBuildMode::DoubleLevelMultiWarp: {
      uint32_t bn = ctx->chunk_total;
      uint32_t tn = 32 * 6;
      next_level = 2;
      reduceBaseIndexedMultiWarp2<<<bn, tn, 0, stream>>>(ctx->index_cases_d,
                                                         (uchar4*)(ctx->vertex_pyramid + ctx->level_offsets[0]),
                                                         ctx->vertex_pyramid + ctx->level_offsets[1],
                                                         ctx->vertex_sidebands[0],
                                                         (uchar4*)(ctx->index_pyramid + ctx->level_offsets[0]),
                                                         ctx->index_pyramid + ctx->level_offsets[1],
                                                         ctx->index_sidebands[0],
                                                         ctx->tables->index_count,
                                                         field_d,
                                                         size_t(field_size.x),
                                                         size_t(field_size.x) * field_size.y,
                                                         threshold,
                                                         ctx->chunks,
                                                         ctx->level_sizes[0],
                                                         ctx->level_sizes[1],
                                                         grid_max_index);
      break;
    }
    default:
      assert(false && "Invalid build mode");
      abort();
    }
    CHECKED_CUDA(cudaEventRecord(ctx->baseEvent, stream));
    CHECKED_CUDA(cudaStreamWaitEvent(ctx->indexStream, ctx->baseEvent, 0));

    // FIXME: Try to run the vertex pyramid in a separate stream. Sync may be too costly though, and in that
    //        case it might be better to just merge the two launches and use e.g. block.y to tell which
    //        pyramid to reduce.
    bool sb = true;
    for (unsigned i = next_level; i < ctx->levels - 3; i++) {
      const auto blocks = (ctx->level_sizes[i] + 31) / 32;
      reduce1<<<blocks, 5 * 32, 0, stream>>>(ctx->index_pyramid + ctx->level_offsets[i],    // Each block will write 32 uvec4's into this
                                             ctx->index_sidebands[sb ? 1 : 0],              // Each block will write 32 uint32's into this
                                             ctx->level_sizes[i],                           // Number of uvec4's in level i
                                             ctx->index_sidebands[sb ? 0 : 1],              // Input, each block will read 5*32=160 values from here
                                             ctx->level_sizes[i - 1]);                      // Number of sideband elements from level i-1
      // FIXME: Try to combine these two kernels into one, as they are independent.
      reduce1<<<blocks, 5 * 32, 0, ctx->indexStream>>>(ctx->vertex_pyramid + ctx->level_offsets[i], // Each block will write 32 uvec4's into this
                                                       ctx->vertex_sidebands[sb ? 1 : 0],           // Each block will write 32 uint32's into this
                                                       ctx->level_sizes[i],                         // Number of uvec4's in level i
                                                       ctx->vertex_sidebands[sb ? 0 : 1],           // Input, each block will read 5*32=160 values from here
                                                       ctx->level_sizes[i - 1]);                    // Number of sideband elements from level i-1
      sb = !sb;
    }
    reduceApex<<<1, 4 * 32, 0, stream>>>(ctx->index_pyramid + 4,
                                         ctx->sum_d,
                                         ctx->index_sidebands[sb ? 0 : 1],
                                         ctx->level_sizes[ctx->levels - 4]);
    // FIXME: Try to combine these two kernels into one, as they are independent.
    reduceApex<<<1, 4 * 32, 0, ctx->indexStream>>>(ctx->vertex_pyramid + 4,
                                                   ctx->sum_d + 1,
                                                   ctx->vertex_sidebands[sb ? 0 : 1],
                                                   ctx->level_sizes[ctx->levels - 4]);

    CHECKED_CUDA(cudaEventRecord(ctx->indexDoneEvent, ctx->indexStream));
    CHECKED_CUDA(cudaStreamWaitEvent(stream, ctx->indexDoneEvent, 0));
  }

  // Non-indexed pyramid buildup
  else {
    reduceBase<float><<<ctx->chunk_total, 32, 0, stream>>>(ctx->index_cases_d,                   // block writes 5*5*32=800 uint8's
                                                           (uchar4*)(ctx->index_pyramid + ctx->level_offsets[0]), // block writes 5*32 uvec4's
                                                           ctx->index_sidebands[0],                    // block writes 5*32 uint32's
                                                           ctx->tables->index_count,
                                                           field_d,
                                                           size_t(field_size.x),
                                                           size_t(field_size.x)* field_size.y,
                                                           threshold,
                                                           ctx->chunks,
                                                           grid_max_index);

    bool sb = true;
    for (unsigned i = 1; i < ctx->levels - 3; i++) {
      const auto blocks = (ctx->level_sizes[i] + 31) / 32;
      reduce1<<<blocks, 5 * 32, 0, stream>>>(ctx->index_pyramid + ctx->level_offsets[i],    // Each block will write 32 uvec4's into this
                                             ctx->index_sidebands[sb ? 1 : 0],              // Each block will write 32 uint32's into this
                                             ctx->level_sizes[i],                           // Number of uvec4's in level i
                                             ctx->index_sidebands[sb ? 0 : 1],              // Input, each block will read 5*32=160 values from here
                                             ctx->level_sizes[i - 1]);                      // Number of sideband elements from level i-1
      sb = !sb;
    }
    reduceApex<<<1, 4 * 32, 0, stream>>>(ctx->index_pyramid + 4,
                                         ctx->sum_d,
                                         ctx->index_sidebands[sb ? 0 : 1],
                                         ctx->level_sizes[ctx->levels - 4]);
  }

}

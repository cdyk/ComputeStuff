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

  __device__ __forceinline__ uint3 cellFromChunkIx(const uint3 chunks, const uint32_t chunk_ix, const uint32_t thread)
  {
    uint3 chunk = make_uint3(chunk_ix % chunks.x,
                             (chunk_ix / chunks.x) % chunks.y,
                             (chunk_ix / chunks.x) / chunks.y);
    uint3 cell = make_uint3(31 * chunk.x + thread,
                            5 * chunk.y,
                            5 * chunk.z);
    return cell;
  }

  template<typename FieldType>
  __device__ __forceinline__ const FieldType* __restrict__ adjustFieldPtrToCell(const FieldType* __restrict__ field_d,
                                                                                const size_t field_row_stride,
                                                                                const size_t field_slice_stride,
                                                                                const uint3 grid_max_index,
                                                                                const uint3 cell)
  {
    return field_d + (min(grid_max_index.z, cell.z) * field_slice_stride +
                      min(grid_max_index.y, cell.y) * field_row_stride +
                      min(grid_max_index.x, cell.x));
  }

  __device__ __forceinline__ void reductionStep(uint4* __restrict__ hpo,
                                                uint32_t* __restrict__ sbo,
                                                const uint32_t* __restrict__ sbi,
                                                const bool mask_sideband,
                                                const bool not_masked)
  {
    uint32_t sum = 0;
    if (not_masked) {
      uint4 hp = make_uint4(sbi[0], sbi[1], sbi[2], sbi[3]);
      sum = hp.x + hp.y + hp.z + hp.w + sbi[4];
      if (sum) {
        *hpo = make_uint4(hp.x,
                          hp.x + hp.y,
                          hp.x + hp.y + hp.z,
                          hp.x + hp.y + hp.z + hp.w);
      }
    }
    if (!mask_sideband || not_masked) {
      *sbo = sum;
    }
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
  //
  // This version lets a single warp process a full chunk, keeping y-values in
  // registers and looping over z.
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

  // This version uses 6 warps to process a full chunk, keeping y-values in
  // registers and using warp number as z-index.
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

  // Build just the base-level, uses 6 warps to process a chunk
  template<class FieldType, bool indexed>
  __global__ __launch_bounds__(32 * 6) void reduceBaseMultiWarp(uint8_t* __restrict__           index_cases_d,
                                                                uchar4* __restrict__            vtx_outlevel0_d,
                                                                uint32_t* __restrict__          vtx_sideband0_d,
                                                                uchar4* __restrict__            idx_outlevel0_d,
                                                                uint32_t* __restrict__          idx_sideband0_d,
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
    const uint3 cell = cellFromChunkIx(chunks, chunk_ix, thread);
    const FieldType* __restrict__ field_cell_d = adjustFieldPtrToCell(field_d, field_row_stride, field_slice_stride, grid_max_index, cell);

    __shared__ float shmem[32 * 6 * 2];

    const uint32_t off_d = 32 * 5 * chunk_ix + thread;
    reduceBaseKernelMultiWarp<FieldType, indexed, false>(index_cases_d + 5 * off_d,
                                                        idx_outlevel0_d + off_d, indexed ? vtx_outlevel0_d + off_d : nullptr,
                                                        idx_sideband0_d + off_d, indexed ? vtx_sideband0_d + off_d : nullptr,
                                                        shmem,
                                                        index_count,
                                                        field_cell_d,
                                                        field_row_stride,
                                                        field_slice_stride,
                                                        cell,
                                                        grid_max_index,
                                                        warp,
                                                        thread,
                                                        threshold);
  }


  // Build the base-level, and run a reducton pass afterwards, building two levels in total.
  template<class FieldType, bool indexed>
  __global__ __launch_bounds__(32 * 6) void reduceBaseDoubleMultiWarp(uint8_t* __restrict__           index_cases_d,
                                                                      uchar4* __restrict__            vtx_outlevel0_d,
                                                                      uint4* __restrict__             vtx_outlevel1_d,
                                                                      uint32_t* __restrict__          vtx_sideband1_d,
                                                                      uchar4* __restrict__            idx_outlevel0_d,
                                                                      uint4* __restrict__             idx_outlevel1_d,
                                                                      uint32_t* __restrict__          idx_sideband1_d,
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
    const uint3 cell = cellFromChunkIx(chunks, chunk_ix, thread);
    const FieldType* __restrict__ field_cell_d = adjustFieldPtrToCell(field_d, field_row_stride, field_slice_stride, grid_max_index, cell);

    __shared__ float shmem[32 * 6 * 2];
    __shared__ uint32_t vtx_sideband0_s[6 * 32]; // If we sync in reduceBaseKernel after fetching adjacent Z, we can recycle shmem
    __shared__ uint32_t idx_sideband0_s[6 * 32];

    const uint32_t off_s = thread;
    const uint32_t off_d = 32 * 5 * chunk_ix + thread;
    reduceBaseKernelMultiWarp<FieldType, indexed, false>(index_cases_d + 5 * off_d,
                                                         idx_outlevel0_d + off_d, indexed ? vtx_outlevel0_d + off_d : nullptr,
                                                         idx_sideband0_s + off_s, indexed ? vtx_sideband0_s + off_s : nullptr,
                                                         shmem,
                                                         index_count,
                                                         field_cell_d,
                                                         field_row_stride,
                                                         field_slice_stride,
                                                         cell,
                                                         grid_max_index,
                                                         warp,
                                                         thread,
                                                         threshold);
    __syncthreads();

    // Run two warps with remaining reduction
    if (warp < (indexed ? 2 : 1)) {
      uint32_t off_s = thread;
      uint32_t off_d = 32 * blockIdx.x + thread;
      reductionStep((!indexed || warp == 0 ? idx_outlevel1_d : vtx_outlevel1_d) + off_d,
                    (!indexed || warp == 0 ? idx_sideband1_d : vtx_sideband1_d) + off_d,
                    (!indexed || warp == 0 ? idx_sideband0_s : vtx_sideband0_s) + 5 * off_s,
                    true, off_d < n1);
    }
  }

  // Build the base-level, and run two reduction passes afterwards, building three levels in total.
  template<class FieldType, bool indexed>
  __global__ __launch_bounds__(32 * 5) void reduceBaseTriple(uint8_t* __restrict__           index_cases_d,
                                                             uchar4* __restrict__            vtx_outlevel0_d,
                                                             uint4* __restrict__             vtx_outlevel1_d,
                                                             uint4* __restrict__             vtx_outlevel2_d,
                                                             uint32_t* __restrict__          vtx_sideband2_d,
                                                             uchar4* __restrict__            idx_outlevel0_d,
                                                             uint4* __restrict__             idx_outlevel1_d,
                                                             uint4* __restrict__             idx_outlevel2_d,
                                                             uint32_t* __restrict__          idx_sideband2_d,
                                                             const uint8_t* __restrict__     index_count_table_d,
                                                             const FieldType* __restrict__   field_d,
                                                             const size_t                    field_row_stride,
                                                             const size_t                    field_slice_stride,
                                                             const float                     threshold,
                                                             const uint3                     chunks,
                                                             const uint32_t                  chunk_count,
                                                             const uint32_t                  n1,
                                                             const uint32_t                  n2,
                                                             const uint3                     grid_max_index)
  {
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t thread = threadIdx.x % 32;
    const uint32_t chunk_ix = 5 * blockIdx.x + warp;
    uint3 chunk = make_uint3(chunk_ix % chunks.x,
                             (chunk_ix / chunks.x) % chunks.y,
                             (chunk_ix / chunks.x) / chunks.y);
    uint3 cell = make_uint3(31 * chunk.x + thread,
                            5 * chunk.y,
                            5 * chunk.z);
    const FieldType* __restrict__ field_cell_d = field_d + (min(grid_max_index.z, cell.z) * field_slice_stride +
                                                            min(grid_max_index.y, cell.y) * field_row_stride +
                                                            min(grid_max_index.x, cell.x));

    // Base level, process 5[warp] * 800 cells.
    // Write 5*5*32 uvec4 and sideband values
    __shared__ uint32_t vtx_sideband0_s[5 * 5 * 32];
    __shared__ uint32_t idx_sideband0_s[5 * 5 * 32];
    if (chunk_ix < chunk_count) {
      const uint32_t off_s = 32 * 5 * warp + thread;
      const uint32_t off_d = 32 * 5 * chunk_ix + thread;
      reduceBaseKernel<FieldType, indexed>(index_cases_d + 5 * off_d,
                                           idx_outlevel0_d + off_d, indexed ? vtx_outlevel0_d + off_d : nullptr,
                                           idx_sideband0_s + off_s, indexed ? vtx_sideband0_s + off_s : nullptr,
                                           index_count_table_d,
                                           field_cell_d,
                                           field_row_stride,
                                           field_slice_stride,
                                           cell,
                                           grid_max_index,
                                           thread,
                                           threshold);
    }
    else {
      for (unsigned k = 0; k < 5; k++) {
        idx_sideband0_s[32 * (5 * warp + k) + thread] = 0;
        if (indexed) {
          vtx_sideband0_s[32 * (5 * warp + k) + thread] = 0;
        }
      }
    }

    // Reduction into level 1, read 5*5*32 sideband values, write 5*32 uvec4 and sideband values
    __shared__ uint32_t vtx_sideband1_s[5 * 32];
    __shared__ uint32_t idx_sideband1_s[5 * 32];
    {
      const uint32_t off_s = 32 * warp + thread;
      const uint32_t off_d = 32 * chunk_ix + thread;
      reductionStep(idx_outlevel1_d +     off_d,
                    idx_sideband1_s +     off_s,
                    idx_sideband0_s + 5 * off_s,
                    false,                off_d < n1);
      if (indexed) {
        reductionStep(vtx_outlevel1_d + off_d,
                      vtx_sideband1_s + off_s,
                      vtx_sideband0_s + 5 * off_s,
                      false, off_d < n1);
      }
    }
    __syncthreads();

    // Reduction into level 2, read 5*32 values, write 32 uvec4 and sideband values
    if (warp < (indexed ? 2 : 1)) {
      uint32_t off_s = thread;
      uint32_t off_d = 32 * blockIdx.x + thread;
      reductionStep((!indexed || warp == 0 ? idx_outlevel2_d : vtx_outlevel2_d) + off_d,
                    (!indexed || warp == 0 ? idx_sideband2_d : vtx_sideband2_d) + off_d,
                    (!indexed || warp == 0 ? idx_sideband1_s : vtx_sideband1_s) + 5 * off_s,
                    true, off_d < n2);
    }

  }

  __global__  __launch_bounds__(5 * 32) void reduceTriple(uint4* __restrict__          hp1_d,
                                                          uint4* __restrict__          hp2_d,
                                                          uint4* __restrict__          hp3_d,
                                                          uint32_t* __restrict__       sb3_d,
                                                          const uint32_t               n1,
                                                          const uint32_t               n2,
                                                          const uint32_t               n3,
                                                          const uint32_t* __restrict__ sb0_d,
                                                          const uint32_t               n0)
  {
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t thread = threadIdx.x % 32;

    __shared__ uint32_t sb0_s[5 * 5 * 32];
    __shared__ uint32_t sb1_s[5 * 5 * 32];
    for (unsigned k = 0; k < 5; k++) {
      for (unsigned l = 0; l < 5; l++) {
        const uint32_t off_d = 32 * (5 * (5 * (5 * blockIdx.x + warp) + k) + l) + thread;
        sb0_s[32 * (5 * warp + l) + thread] = off_d < n0 ? sb0_d[off_d] : 0;
      }
      const uint32_t off_s = 32 * (5 * warp + k) + thread;
      const uint32_t off_d = 32 * (5 * (5 * blockIdx.x + warp) + k) + thread;
      reductionStep(hp1_d + off_d,
                    sb1_s + off_s,
                    sb0_s + 5 * (32 * warp + thread),
                    false, off_d < n1);
    }

    __shared__ uint32_t sb2_s[5 * 32];
    {
      const uint32_t off_s = 32 * warp + thread;
      const uint32_t off_d = 5 * 32 * blockIdx.x + off_s;
      reductionStep(hp2_d + off_d,
                    sb2_s + off_s,
                    sb1_s + 5 * off_s,
                    false, off_d < n2);
    }

    __syncthreads();

    if (warp == 0) { // First warp
      const uint32_t off_s = thread;
      const uint32_t off_d = 32 * blockIdx.x + off_s;
      reductionStep(hp3_d + off_d,
                    sb3_d + off_d,
                    sb2_s + 5 * off_s,
                    true, off_d < n3);
    }
  }


  __global__  __launch_bounds__(5 * 32) void reduceDouble(uint4* __restrict__          hp1_d,
                                                          uint4* __restrict__          hp2_d,
                                                          uint32_t* __restrict__       sb2_d,
                                                          const uint32_t               n1,
                                                          const uint32_t               n2,
                                                          const uint32_t* __restrict__ sb0_d,
                                                          const uint32_t               n0)
  {
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t thread = threadIdx.x % 32;

    __shared__ uint32_t sb0_s[5 * 5 * 32];
    for (unsigned k = 0; k < 5; k++) {
      const uint32_t off_s = 5 * 32 * warp + 32 * k + thread;
      const uint32_t off_d = 5 * 5 * 32 * blockIdx.x + off_s;
      sb0_s[off_s] = off_d < n0 ? sb0_d[off_d] : 0;
    }

    __shared__ uint32_t sb1_s[5 * 32];
    {
      const uint32_t off_s = 32 * warp + thread;
      const uint32_t off_d = 5 * 32 * blockIdx.x + off_s;
      reductionStep(hp1_d + off_d,
                    sb1_s + off_s,
                    sb0_s + 5 * off_s,
                    false, off_d < n1);
    }

    __syncthreads();

    if (warp == 0) { // First warp
      const uint32_t off_s = thread;
      const uint32_t off_d = 32 * blockIdx.x + off_s;
      reductionStep(hp2_d + off_d,
                    sb2_d + off_d,
                    sb1_s + 5 * off_s,
                    true, off_d < n2);
    }
  }

  __global__  __launch_bounds__(5 * 32) void reduceSingle(uint4* __restrict__          hp1_d,  //< Each block will write 32 uvec4's into this
                                                          uint32_t* __restrict__       sb1_d,  //< Each block will write 32 values into this.
                                                          const uint32_t               n1,     //< Number of uvec4's in hp1_d
                                                          const uint32_t* __restrict__ sb0_d,  //< Each block will read 5*32=160 values from here
                                                          const uint32_t               n0)     //< Number of elements in sb0_d
  {
    const uint32_t offset0 = 5 * 32 * blockIdx.x + threadIdx.x;

    // FIXME: Test idea, each warp reads 32 values. read instead 32/4 uint4's.
    __shared__ uint32_t sb0_s[5 * 32];
    sb0_s[threadIdx.x] = offset0 < n0 ? sb0_d[offset0] : 0;

    __syncthreads();

    if (threadIdx.x < 32) { // First warp
      const uint32_t off_s = threadIdx.x;
      const uint32_t off_d = 32 * blockIdx.x + threadIdx.x;

      reductionStep(hp1_d + off_d,
                    sb1_d + off_d,
                    sb0_s + 5 * off_s,
                    true, off_d < n1);
    }
  }


  // Build 3 top levels (=apex), which are tiny.
  __global__ __launch_bounds__(128) void reduceApex(uint4* __restrict__ apex_d,
                                                    uint32_t* sum_d,
                                                    const uint32_t* in_d,
                                                    uint32_t N)
  {
    // 0 : sum + 3 padding
    // 1 : 1 uvec4 of top level.
    // 2 : 5 values of top level - 1
    // 7 : 25 values of top level - 2
    // 32: total number of uvec4's in apex.

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
  unsigned nl = ~0u;
  auto build_mode = ctx->build_mode;
  if (build_mode == BaseLevelBuildMode::TripleLevelSingleWarpChunk && ctx->levels == 5) {
    // There are only two levels below the apex
    ctx->build_mode = BaseLevelBuildMode::DoubleLevelMultiWarpChunk;
  }

  switch (ctx->build_mode) {
  case BaseLevelBuildMode::SingleLevelMultiWarpChunk: {
    uint32_t bn = ctx->chunk_total;
    uint32_t tn = 32 * 6;
    nl = 1;
    if (ctx->indexed) {
      reduceBaseMultiWarp<float, true><<<bn, tn, 0, stream>>>(ctx->index_cases_d,                   // block writes 5*5*32=800 uint8's
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
    }
    else {
      reduceBaseMultiWarp<float, false><<<bn, tn, 0, stream>>>(ctx->index_cases_d,
                                                               nullptr, nullptr,
                                                               (uchar4*)(ctx->index_pyramid + ctx->level_offsets[0]), ctx->index_sidebands[0],                    // block writes 5*32 uint32's
                                                               ctx->tables->index_count,
                                                               field_d,
                                                               size_t(field_size.x),
                                                               size_t(field_size.x) * field_size.y,
                                                               threshold,
                                                               ctx->chunks,
                                                               grid_max_index);
    }
    break;
  }
  case BaseLevelBuildMode::DoubleLevelMultiWarpChunk: {
    uint32_t bn = ctx->chunk_total;
    uint32_t tn = 32 * 6;
    nl = 2;
    if (ctx->indexed) {
      reduceBaseDoubleMultiWarp<float, true><<<bn, tn, 0, stream>>>(ctx->index_cases_d,
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
    }
    else {
      reduceBaseDoubleMultiWarp<float, false><<<bn, tn, 0, stream>>>(ctx->index_cases_d,
                                                                     nullptr,
                                                                     nullptr,
                                                                     nullptr,
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
    }
    break;
  }
  case BaseLevelBuildMode::TripleLevelSingleWarpChunk: {
    uint32_t bn = (ctx->level_sizes[2] + 31) / 32;
    uint32_t tn = 32 * 5;
    nl = 3;
    if (ctx->indexed) {
      reduceBaseTriple<float, true><<<bn, tn, 0, stream>>>(ctx->index_cases_d,
                                                            (uchar4*)(ctx->vertex_pyramid + ctx->level_offsets[0]),
                                                            ctx->vertex_pyramid + ctx->level_offsets[1],
                                                            ctx->vertex_pyramid + ctx->level_offsets[2],
                                                            ctx->vertex_sidebands[0],
                                                            (uchar4*)(ctx->index_pyramid + ctx->level_offsets[0]),
                                                            ctx->index_pyramid + ctx->level_offsets[1],
                                                            ctx->index_pyramid + ctx->level_offsets[2],
                                                            ctx->index_sidebands[0],
                                                            ctx->tables->index_count,
                                                            field_d,
                                                            size_t(field_size.x),
                                                            size_t(field_size.x) * field_size.y,
                                                            threshold,
                                                            ctx->chunks,
                                                            ctx->chunk_total,
                                                            ctx->level_sizes[1],
                                                            ctx->level_sizes[2],
                                                            grid_max_index);
    }
    else {
      reduceBaseTriple<float, false><<<bn, tn, 0, stream>>>(ctx->index_cases_d,
                                                            nullptr,
                                                            nullptr,
                                                            nullptr,
                                                            nullptr,
                                                            (uchar4*)(ctx->index_pyramid + ctx->level_offsets[0]),
                                                            ctx->index_pyramid + ctx->level_offsets[1],
                                                            ctx->index_pyramid + ctx->level_offsets[2],
                                                            ctx->index_sidebands[0],
                                                            ctx->tables->index_count,
                                                            field_d,
                                                            size_t(field_size.x),
                                                            size_t(field_size.x)* field_size.y,
                                                            threshold,
                                                            ctx->chunks,
                                                            ctx->chunk_total,
                                                            ctx->level_sizes[1],
                                                            ctx->level_sizes[2],
                                                            grid_max_index);
    }
    break;
  }
  default:
    assert(false && "Invalid build mode");
    abort();
  }
  if (ctx->indexed) {
    CHECKED_CUDA(cudaEventRecord(ctx->baseEvent, stream));
    CHECKED_CUDA(cudaStreamWaitEvent(ctx->indexStream, ctx->baseEvent, 0));
  }

  bool sb = true;

  // Do triple reductions as long as it goes...
  for (; nl + 2 < ctx->levels - 3; nl += 3) {
    const auto blocks = (ctx->level_sizes[nl + 1] + 31) / 32;
    reduceTriple << <blocks, 5 * 32, 0, stream >> > (ctx->index_pyramid + ctx->level_offsets[nl],
                                                     ctx->index_pyramid + ctx->level_offsets[nl + 1],
                                                     ctx->index_pyramid + ctx->level_offsets[nl + 2],
                                                     ctx->index_sidebands[sb ? 1 : 0],
                                                     ctx->level_sizes[nl],
                                                     ctx->level_sizes[nl + 1],
                                                     ctx->level_sizes[nl + 2],
                                                     ctx->index_sidebands[sb ? 0 : 1],
                                                     ctx->level_sizes[nl - 1]);
    if (ctx->indexed) {
      reduceTriple << <blocks, 5 * 32, 0, ctx->indexStream >> > (ctx->vertex_pyramid + ctx->level_offsets[nl],
                                                                 ctx->vertex_pyramid + ctx->level_offsets[nl + 1],
                                                                 ctx->vertex_pyramid + ctx->level_offsets[nl + 2],
                                                                 ctx->vertex_sidebands[sb ? 1 : 0],
                                                                 ctx->level_sizes[nl],
                                                                 ctx->level_sizes[nl + 1],
                                                                 ctx->level_sizes[nl + 2],
                                                                 ctx->vertex_sidebands[sb ? 0 : 1],
                                                                 ctx->level_sizes[nl - 1]);
    }
    sb = !sb;
  }

  // Do a double reduction if two levels remain
  for (; nl + 1 < ctx->levels - 3; nl += 2) {
    const auto blocks = (ctx->level_sizes[nl + 1] + 31) / 32;
    reduceDouble<<<blocks, 5 * 32, 0, stream>>>(ctx->index_pyramid + ctx->level_offsets[nl],
                                                ctx->index_pyramid + ctx->level_offsets[nl + 1],
                                                ctx->index_sidebands[sb ? 1 : 0],
                                                ctx->level_sizes[nl],
                                                ctx->level_sizes[nl + 1],
                                                ctx->index_sidebands[sb ? 0 : 1],
                                                ctx->level_sizes[nl - 1]);
    if (ctx->indexed) {
      reduceDouble<<<blocks, 5 * 32, 0, ctx->indexStream>>>(ctx->vertex_pyramid + ctx->level_offsets[nl],
                                                            ctx->vertex_pyramid + ctx->level_offsets[nl + 1],
                                                            ctx->vertex_sidebands[sb ? 1 : 0],
                                                            ctx->level_sizes[nl],
                                                            ctx->level_sizes[nl + 1],
                                                            ctx->vertex_sidebands[sb ? 0 : 1],
                                                            ctx->level_sizes[nl - 1]);
    }
    sb = !sb;
  }

  // Do a single reduction if one remaining.
  for (; nl < ctx->levels - 3; nl++) {
    const auto blocks = (ctx->level_sizes[nl] + 31) / 32;
    reduceSingle<<<blocks, 5 * 32, 0, stream>>>(ctx->index_pyramid + ctx->level_offsets[nl],
                                                ctx->index_sidebands[sb ? 1 : 0],
                                                ctx->level_sizes[nl],
                                                ctx->index_sidebands[sb ? 0 : 1],
                                                ctx->level_sizes[nl - 1]);
    if(ctx->indexed) {
      reduceSingle<<<blocks, 5 * 32, 0, ctx->indexStream>>>(ctx->vertex_pyramid + ctx->level_offsets[nl],
                                                            ctx->vertex_sidebands[sb ? 1 : 0],
                                                            ctx->level_sizes[nl],
                                                            ctx->vertex_sidebands[sb ? 0 : 1],
                                                            ctx->level_sizes[nl - 1]);
    }
    sb = !sb;
  }
  reduceApex<<<1, 4 * 32, 0, stream>>>(ctx->index_pyramid + 4,
                                        ctx->sum_d,
                                        ctx->index_sidebands[sb ? 0 : 1],
                                        ctx->level_sizes[ctx->levels - 4]);
  if (ctx->indexed) {
    reduceApex << <1, 4 * 32, 0, ctx->indexStream >> > (ctx->vertex_pyramid + 4,
                                                        ctx->sum_d + 1,
                                                        ctx->vertex_sidebands[sb ? 0 : 1],
                                                        ctx->level_sizes[ctx->levels - 4]);
    CHECKED_CUDA(cudaEventRecord(ctx->indexDoneEvent, ctx->indexStream));
    CHECKED_CUDA(cudaStreamWaitEvent(stream, ctx->indexDoneEvent, 0));
  }
}

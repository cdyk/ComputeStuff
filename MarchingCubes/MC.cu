// This file is part of ComputeStuff copyright (C) 2020 Christopher Dyken.
// Released under the MIT license, please see LICENSE file for details.

#include "MC.h"
#include <cassert>
#include <cmath>

namespace {

  struct uint5 {
    uint32_t e0;
    uint32_t e1;
    uint32_t e2;
    uint32_t e3;
    uint32_t e4;
  };


  template<typename T>
  __device__ float2 fetch(const T* ptr, const T* end, const size_t stride, const float t, unsigned thread)
  {
    float t0 = ((ptr < end) && (ptr[0] < t) ? 1.f : 0.f); ptr += stride;
    float t1 = ((ptr < end) && (ptr[0] < t) ? 1.f : 0.f); ptr += stride;
    float t2 = ((ptr < end) && (ptr[0] < t) ? 1.f : 0.f); ptr += stride;
    float t3 = ((ptr < end) && (ptr[0] < t) ? 1.f : 0.f); ptr += stride;
    float t4 = ((ptr < end) && (ptr[0] < t) ? 1.f : 0.f); ptr += stride;
    float t5 = ((ptr < end) && (ptr[0] < t) ? 1.f : 0.f); ptr += stride;

    float r0 = __fmaf_rn(float(1 << 16), t2, __fmaf_rn(float(1 << 8), t1, t0));
    float r1 = __fmaf_rn(float(1 << 16), t5, __fmaf_rn(float(1 << 8), t4, t3));
    return make_float2(r0, r1);
  }

  __device__ float2 mergeZ(const float2 z0, const float2 z1)
  {
    return make_float2(__fmaf_rn(float(1 << 4), z1.x, z0.x),
                       __fmaf_rn(float(1 << 4), z1.y, z0.y));
  }
  __device__ __forceinline__ uint32_t piercingAxesFromCase(uint32_t c)
  {
    return ((((c & 1) << 1) |
             ((c & 1) << 2) |
             ((c & 1) << 4) ) ^ c ) & 0b10110;
  }

  __device__ __forceinline__ uint32_t axisCountFromCase(const uint32_t c)
  {
    uint32_t n;
    asm("{\n"
        "  .reg .u32 t1, t2, t3;\n"
        "  bfe.s32 t1, %1, 0, 1;\n"     // Sign-extend bit 0 (0,0,0) over whole word
        "  xor.b32 t2, %1, t1;\n"       // XOR with (0,0,0) to see which corners change sign wrt (0,0,0).
        "  and.b32 t3, t2, 0b10110;\n"  // Mask out (1,0,0), (0,1,0) and (0,0,1).
        "  popc.b32 %0, t3;\n"          // Count number of 1's (should be 0..3).
        "}"
        : "=r"(n) :"r"(c));

    unsigned q = piercingAxesFromCase(c);
    assert(((q >> 1) & 1) + ((q >> 2) & 1) + ((q >> 4) & 1) == n);

    unsigned m =
      ((c & 1) ^ ((c >> 1) & 1)) +
      ((c & 1) ^ ((c >> 2) & 1)) +
      ((c & 1) ^ ((c >> 4) & 1));
    assert(n == m);
    assert(n <= 3);
    return n;
  }

  template<class FieldType, bool indexed>
  __global__ __launch_bounds__(32) void reduce_base(uint8_t* __restrict__           index_cases_d,
                                                    uint4* __restrict__             out_vertex_level0_d,
                                                    uint32_t* __restrict__          out_vertex_sideband_d,
                                                    uint4* __restrict__             out_index_level0_d,
                                                    uint32_t* __restrict__          out_index_sideband_d,
                                                    const uint8_t* __restrict__     index_count,
                                                    const FieldType* __restrict__   field_d,
                                                    const FieldType* __restrict__   field_end_d,
                                                    const size_t                    field_row_stride,
                                                    const size_t                    field_slice_stride,
                                                    const float                     threshold,
                                                    const uint3                     chunks,
                                                    const uint3                     cells)
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

    // TODO check if it is faster to only process 31 cells and don't rely on cache for fetching field.

    // One warp processes a 32 * 5 * 5 chunk, outputs 800 mc cases to base level and 800/5=160 sums to sideband.
    const FieldType* ptr = field_d
      + cell.z * field_slice_stride
      + cell.y * field_row_stride
      + cell.x;
    float2 prev = fetch(ptr, field_end_d, field_row_stride, threshold, thread);
    ptr += field_slice_stride;

    bool xmask = cell.x < cells.x;
    for (unsigned y = 0; y < 5; y++) {

      unsigned isum = 0;
      unsigned vsum = 0;
      bool zmask = cell.z < cells.z;
      if (zmask) {
        float2 next = fetch(ptr, field_end_d, field_row_stride, threshold, thread);
        ptr += field_slice_stride;

      float2 t0 = make_float2(__fmaf_rn(float(1 << 4), next.x, prev.x),
                              __fmaf_rn(float(1 << 4), next.y, prev.y));

        prev = next;

        float2 tt = make_float2(__shfl_down_sync(0xffffffff, t0.x, 1),
                                __shfl_down_sync(0xffffffff, t0.y, 1));

        if (xmask && thread < 31) {
          uint32_t g0 = static_cast<uint32_t>(__fmaf_rn(2.f, tt.x, t0.x));
          uint32_t g1 = static_cast<uint32_t>(__fmaf_rn(2.f, tt.y, t0.y));
          uint32_t s0 = __byte_perm(g0, g1, (4 << 12) | (2 << 8) | (1 << 4) | (0 << 0));
          g0 = g0 | (s0 >> 6);
          g1 = g1 | (g1 >> 6);

          uint5 cases = {
            __byte_perm(g0, 0, (4 << 12) | (4 << 8) | (4 << 4) | (0 << 0)),
            __byte_perm(g0, 0, (4 << 12) | (4 << 8) | (4 << 4) | (1 << 0)),
            __byte_perm(g0, 0, (4 << 12) | (4 << 8) | (4 << 4) | (2 << 0)),
            __byte_perm(g1, 0, (4 << 12) | (4 << 8) | (4 << 4) | (0 << 0)),
            __byte_perm(g1, 0, (4 << 12) | (4 << 8) | (4 << 4) | (1 << 0))
          };

          uint32_t vc0 = axisCountFromCase(cases.e0);
          uint32_t vc1 = axisCountFromCase(cases.e1);
          uint32_t vc2 = axisCountFromCase(cases.e2);
          uint32_t vc3 = axisCountFromCase(cases.e3);
          uint32_t vc4 = axisCountFromCase(cases.e4);

          vsum = vc0 + vc1 + vc2 + vc3 + vc4;
          if (vsum) {
            out_index_level0_d[32 * 5 * blockIdx.x + 32 * y + thread] = make_uint4(vc0,
                                                                                   vc0 + vc1,
                                                                                   vc0 + vc1 + vc2,
                                                                                   vc0 + vc1 + vc2 + vc3);
          }

          uint5 counts{
            cell.y + 0u < cells.y ? index_count[cases.e0] : 0u,
            cell.y + 1u < cells.y ? index_count[cases.e1] : 0u,
            cell.y + 2u < cells.y ? index_count[cases.e2] : 0u,
            cell.y + 3u < cells.y ? index_count[cases.e3] : 0u,
            cell.y + 4u < cells.y ? index_count[cases.e4] : 0u,
          };

          isum = counts.e0 + counts.e1 + counts.e2 + counts.e3 + counts.e4;
          if (isum) {
            // MC cases and HP base level is only visited if sum is nonzero
            index_cases_d[5 * (32 * 5 * blockIdx.x + 32 * y + thread) + 0] = cases.e0;
            index_cases_d[5 * (32 * 5 * blockIdx.x + 32 * y + thread) + 1] = cases.e1;
            index_cases_d[5 * (32 * 5 * blockIdx.x + 32 * y + thread) + 2] = cases.e2;
            index_cases_d[5 * (32 * 5 * blockIdx.x + 32 * y + thread) + 3] = cases.e3;
            index_cases_d[5 * (32 * 5 * blockIdx.x + 32 * y + thread) + 4] = cases.e4;
            out_index_level0_d[32 * 5 * blockIdx.x + 32 * y + thread] = make_uint4(counts.e0,
                                                                             counts.e0 + counts.e1,
                                                                             counts.e0 + counts.e1 + counts.e2,
                                                                             counts.e0 + counts.e1 + counts.e2 + counts.e3);
          }
          if (indexed && vsum) {
            out_vertex_level0_d[32 * 5 * blockIdx.x + 32 * y + thread] = make_uint4(vc0,
                                                                                    vc0 + vc1,
                                                                                    vc0 + vc1 + vc2,
                                                                                    vc0 + vc1 + vc2 + vc3);

          }
          
        }
      }
      cell.z++;
      if (indexed) {
        out_vertex_sideband_d[32 * (5 * blockIdx.x + y) + threadIdx.x] = vsum;
      }
      out_index_sideband_d[32 * (5 * blockIdx.x + y) + threadIdx.x] = isum;
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

    // Idea, each warp reads 32 values. read instead 32/4 uint4's.

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
    for (unsigned l = level_count - 4; l < level_count; l--) {
      offset = processHP5Item(key, 5 * offset, pyramid[level_offset[l] + offset]);
    }
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

    float t = 0.5;// (threshold - f0.w) / (f1.w - f0.w);

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
                                                           uint3 cells,
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

  ctx->tables = tables;
  ctx->cells = cells;
  ctx->indexed = indexed;
  fprintf(stderr, "Cells [%u x %u x %u]\n", ctx->cells.x, ctx->cells.y, ctx->cells.z);

  // Each chunk handles a set of 32 x 5 x 5 cells.
  ctx->chunks = make_uint3(((cells.x + (indexed ? 1 : 0) + 30) / 31),
                           ((cells.y + (indexed ? 1 : 0) + 4) / 5),
                           ((cells.z + (indexed ? 1 : 0) + 4) / 5));
  ctx->chunk_total = ctx->chunks.x * ctx->chunks.y * ctx->chunks.z;
  fprintf(stderr, "Chunks [%u x %u x %u] = %u\n",
          ctx->chunks.x, ctx->chunks.y, ctx->chunks.z, ctx->chunk_total);


  // Pyramid base level, as number of uvec4's:
  ctx->level_sizes[0] = (800 * ctx->chunk_total + 4) / 5;
  ctx->levels = 1 + static_cast<uint32_t>(std::ceil(std::log(ctx->level_sizes[0]) / std::log(5.0)));
  assert(4 <= ctx->levels);
  assert(ctx->levels < 16u); // 5^14 = 6103515625

  ctx->level_offsets[0] = 4 + 32; // First, level offsets (16 uint32's = 4 uvec4's), then pyramid apex (32 uvec4's).
  for (unsigned l = 1; l < ctx->levels - 3; l++) {
    ctx->level_sizes[l] = (ctx->level_sizes[l - 1] + 4) / 5;
    ctx->level_offsets[l] = ctx->level_offsets[l - 1] + ctx->level_sizes[l - 1];
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

  CHECKED_CUDA(cudaMemcpy(ctx->index_pyramid, ctx->level_offsets, sizeof(Context::level_offsets), cudaMemcpyHostToDevice));

  if (indexed) {
    CHECKED_CUDA(cudaMalloc(&ctx->vertex_cases_d, sizeof(uint32_t) * 800 * ctx->chunk_total));
    CHECKED_CUDA(cudaMalloc(&ctx->vertex_pyramid, sizeof(uint4) * ctx->total_size));
    CHECKED_CUDA(cudaMalloc(&ctx->vertex_sidebands[0], sideband0_size));
    CHECKED_CUDA(cudaMalloc(&ctx->vertex_sidebands[1], sideband1_size));

    CHECKED_CUDA(cudaMemsetAsync(ctx->vertex_pyramid, 1, sizeof(uint4) * ctx->total_size, stream));
    CHECKED_CUDA(cudaMemsetAsync(ctx->vertex_sidebands[0], 1, sideband0_size, stream));
    CHECKED_CUDA(cudaMemsetAsync(ctx->vertex_sidebands[1], 1, sideband1_size, stream));

    CHECKED_CUDA(cudaMemcpy(ctx->vertex_pyramid, ctx->level_offsets, sizeof(Context::level_offsets), cudaMemcpyHostToDevice));
  }


  for (unsigned l = 0; l < ctx->levels; l++) {
    fprintf(stderr, "[%d] %8d %8d  (%8d)\n", l, ctx->level_offsets[l], ctx->level_sizes[l], 4 * ctx->level_sizes[l]);
  }
  fprintf(stderr, "Total %d, levels %d \n", ctx->total_size, ctx->levels);

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

  delete ctx;
}


void ComputeStuff::MC::destroyContext(Context* ctx)
{
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
                               void* output_buffer,
                               size_t output_buffer_size,
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
    if (ctx->indexed) {
      ::reduce_base<float, true><<<ctx->chunk_total, 32, 0, stream>>>(ctx->index_cases_d,                   // block writes 5*5*32=800 uint8's
                                                                      ctx->vertex_pyramid + ctx->level_offsets[0], // block writes 5*32 uvec4's
                                                                      ctx->vertex_sidebands[0],                    // block writes 5*32 uint32's
                                                                      ctx->index_pyramid + ctx->level_offsets[0], // block writes 5*32 uvec4's
                                                                      ctx->index_sidebands[0],                    // block writes 5*32 uint32's
                                                                      ctx->tables->index_count,
                                                                      field_d,
                                                                      field_d + size_t(field_size.x) * field_size.y * field_size.z,
                                                                      size_t(field_size.x),
                                                                      size_t(field_size.x) * field_size.y,
                                                                      threshold,
                                                                      ctx->chunks,
                                                                      ctx->cells);
    }
    else {
      ::reduce_base<float, false><<<ctx->chunk_total, 32, 0, stream>>>(ctx->index_cases_d,                   // block writes 5*5*32=800 uint8's
                                                                       nullptr,
                                                                       nullptr,
                                                                       ctx->index_pyramid + ctx->level_offsets[0], // block writes 5*32 uvec4's
                                                                       ctx->index_sidebands[0],                    // block writes 5*32 uint32's
                                                                       ctx->tables->index_count,
                                                                       field_d,
                                                                       field_d + size_t(field_size.x) * field_size.y * field_size.z,
                                                                       size_t(field_size.x),
                                                                       size_t(field_size.x) * field_size.y,
                                                                       threshold,
                                                                       ctx->chunks,
                                                                       ctx->cells);
    }

    bool sb = true;
    for (unsigned i = 1; i < ctx->levels - 3; i++) {
      const auto blocks = (ctx->level_sizes[i] + 31) / 32;
      reduce1<<<blocks, 5 * 32, 0, stream>>>(ctx->index_pyramid + ctx->level_offsets[i],    // Each block will write 32 uvec4's into this
                                             ctx->index_sidebands[sb ? 1 : 0],              // Each block will write 32 uint32's into this
                                             ctx->level_sizes[i],                           // Number of uvec4's in level i
                                             ctx->index_sidebands[sb ? 0 : 1],              // Input, each block will read 5*32=160 values from here
                                             ctx->level_sizes[i - 1]);                      // Number of sideband elements from level i-1
      if (ctx->indexed) {
        reduce1<<<blocks, 5 * 32, 0, stream>>>(ctx->vertex_pyramid + ctx->level_offsets[i], // Each block will write 32 uvec4's into this
                                               ctx->vertex_sidebands[sb ? 1 : 0],           // Each block will write 32 uint32's into this
                                               ctx->level_sizes[i],                         // Number of uvec4's in level i
                                               ctx->vertex_sidebands[sb ? 0 : 1],           // Input, each block will read 5*32=160 values from here
                                               ctx->level_sizes[i - 1]);                    // Number of sideband elements from level i-1
      }
      sb = !sb;
    }
    reduceApex<<<1, 4 * 32, 0, stream>>>(ctx->index_pyramid + 8,
                                         ctx->sum_d,
                                         ctx->index_sidebands[sb ? 0 : 1],
                                         ctx->level_sizes[ctx->levels - 4]);
    if (ctx->indexed) {
      reduceApex<<<1, 4 * 32, 0, stream>>>(ctx->vertex_pyramid + 8,
                                           ctx->sum_d + 1,
                                           ctx->vertex_sidebands[sb ? 0 : 1],
                                           ctx->level_sizes[ctx->levels - 4]);
    }
  }

  if (output_buffer) {
    ::runExtractionPN<<<1, 32, 0, stream>>>(reinterpret_cast<float*>(output_buffer),
                                            nullptr,
                                            ctx->vertex_pyramid,
                                            ctx->index_pyramid,
                                            field_d,
                                            ctx->index_cases_d,
                                            ctx->tables->index_table,
                                            field_row_stride,
                                            field_slice_stride,
                                            field_offset,
                                            field_size,
                                            static_cast<uint32_t>(output_buffer_size / (6 * sizeof(float))),
                                            0,
                                            make_uint2(ctx->chunks.x, ctx->chunks.y),
                                            make_float3(1.f / ctx->cells.x,
                                                        1.f / ctx->cells.y,
                                                        1.f / ctx->cells.z),
                                            threshold,
                                            alwaysExtract,
                                            ctx->indexed);
  }
}

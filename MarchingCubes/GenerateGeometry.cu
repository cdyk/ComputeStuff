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

  __device__ __forceinline__ uint32_t processHP5Item(uint32_t& key, uint32_t offset, const uint4 item)
  {
    if (key < item.x) { }
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
    // Traverse apex, offset 
    // 4 + 0 : sum + 3 padding
    // 4 + 1 : 1 uvec4 of level 0.
    // 4 + 2 : 5 values of level 0 (top)
    // 4 + 7 : 25 values of level 1
    // 4 + 32: total sum.

    uint32_t offset = 0;
    offset = processHP5Item(key, 0, pyramid[4 + 1]);
    offset = processHP5Item(key, 5 * offset, pyramid[4 + 2 + offset]);
    offset = processHP5Item(key, 5 * offset, pyramid[4 + 7 + offset]);
    for (unsigned l = level_count - 4; 0 < l; l--) {
      offset = processHP5Item(key, 5 * offset, pyramid[level_offset[l] + offset]);
    }
    uchar4 b = ((const uchar4*)(pyramid + 4 + 32))[offset];
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

  __global__ void launchExtractIndexedVertexPN(float* __restrict__          vertex_output,
                                               const uint4* __restrict__    vertex_pyramid,
                                               const float* __restrict__    field,
                                               const uint8_t* __restrict__  index_cases,
                                               const uint8_t* __restrict__  index_table,
                                               const size_t                 field_row_stride,
                                               const size_t                 field_slice_stride,
                                               const uint3                  field_offset,
                                               const uint3                  field_max_index,
                                               const uint32_t               vertex_capacity,
                                               const uint2                  chunks,
                                               const float3                 scale,
                                               const float                  threshold,
                                               const bool                   always_extract)
  {
    if (threadIdx.x == 0) {
      uint32_t vertex_count_clamped = min(vertex_capacity, vertex_pyramid[4].x);

      if (vertex_count_clamped) {
        extractIndexedVertexPN<<<(vertex_count_clamped + 255) / 256, 256>>>(vertex_output,
                                                                            vertex_pyramid,
                                                                            field,
                                                                            index_cases,
                                                                            index_table,
                                                                            field_row_stride,
                                                                            field_slice_stride,
                                                                            field_offset,
                                                                            field_max_index,
                                                                            vertex_count_clamped,
                                                                            chunks,
                                                                            scale,
                                                                            threshold);
      }
    }
  }

  __global__ void extractIndices(uint32_t* __restrict__       indices,
                                 const uint4* __restrict__    vertex_pyramid,
                                 const uint4* __restrict__    index_pyramid,
                                 const uint8_t* __restrict__  index_cases,
                                 const uint8_t* __restrict__  index_table,
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
                                ((index_code & (1 << 5)) ? (chunk_pos.z == 4 ? (800 * chunks.x * chunks.y - 4 * 5 * 32) : (5 * 32)) : 0));

      uint32_t vertex_cell_case = index_cases[vertex_offset];
      uint32_t axes = piercingAxesFromCase(vertex_cell_case);

      // Traverse apex, offset 
      // 4 + 0 : sum + 3 padding
      // 4 + 1 : 1 uvec4 of level 0.
      // 4 + 2 : 5 values of level 0 (top)
      // 4 + 7 : 25 values of level 1
      // 4 + 32: total sum.
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
      for (unsigned l = 1; l + 1 < level_count; l++) {
        uint32_t rem = vertex_offset % 5;
        vertex_offset = vertex_offset / 5;
        uint4 item = vertex_pyramid[level_offset[l] + vertex_offset];
        if (rem == 1) vertex_index += item.x;
        else if (rem == 2) vertex_index += item.y;
        else if (rem == 3) vertex_index += item.z;
        else if (rem == 4) vertex_index += item.w;
      }
      { // Top level, vertex offset is now in range 0..4
        uint4 item = vertex_pyramid[4 + 1];
        if (vertex_offset == 1) vertex_index += item.x;
        else if (vertex_offset == 2) vertex_index += item.y;
        else if (vertex_offset == 3) vertex_index += item.z;
        else if (vertex_offset == 4) vertex_index += item.w;
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

  __global__ void launchExtractIndices(uint32_t* __restrict__       index_output,
                                       const uint4* __restrict__    vertex_pyramid,
                                       const uint4* __restrict__    index_pyramid,
                                       const uint8_t* __restrict__  index_cases,
                                       const uint8_t* __restrict__  index_table,
                                       const uint32_t               index_capacity,
                                       const uint2                  chunks,
                                       const bool                   always_extract)
  {
    if (threadIdx.x == 0) {
      uint32_t index_count_clamped = min(index_capacity, index_pyramid[4].x);
      if (index_count_clamped) {
        extractIndices<<<(index_count_clamped + 255) / 256, 256>>>(index_output,
                                                                   vertex_pyramid,
                                                                   index_pyramid,
                                                                   index_cases,
                                                                   index_table,
                                                                   index_count_clamped,
                                                                   chunks);
      }
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

      // Down traversal is identical for three and three vertices (they belong
      // to the same triangle). I tried (RTX2080 + CUDA11) to just do a single
      // downtraversal and a loop over the three vertices, reducing the number
      // of blocks by 3. But that was considerably slower, so it appears that
      // the cache handles the downtraversal pretty well.
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

  __global__ void launchExtractVertexPN(float* __restrict__          vertex_output,
                                        const uint4* __restrict__    index_pyramid,
                                        const float* __restrict__    field,
                                        const uint8_t* __restrict__  index_cases,
                                        const uint8_t* __restrict__  index_table,
                                        const size_t                 field_row_stride,
                                        const size_t                 field_slice_stride,
                                        const uint3                  field_offset,
                                        const uint3                  field_max_index,
                                        const uint32_t               vertex_capacity,
                                        const uint2                  chunks,
                                        const float3                 scale,
                                        const float                  threshold,
                                        const bool                   always_extract)
  {
    if (threadIdx.x == 0) {
      uint32_t vertex_count_clamped = min(vertex_capacity, index_pyramid[4].x);
      if (vertex_count_clamped) {
        extractVertexPN<<<(vertex_count_clamped + 255) / 256, 256>>>(vertex_output,
                                                                     index_pyramid,
                                                                     field,
                                                                     index_cases,
                                                                     index_table,
                                                                     field_row_stride,
                                                                     field_slice_stride,
                                                                     field_offset,
                                                                     field_max_index,
                                                                     vertex_count_clamped,
                                                                     chunks,
                                                                     scale,
                                                                     threshold);
      }
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


void ComputeStuff::MC::Internal::GenerateGeometryPN(Context* ctx,
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
                                                    bool alwaysExtract)
{
  const auto vertex_capacity = static_cast<uint32_t>(vertex_buffer_bytesize / (6 * sizeof(float)));
  const auto index_capacity = static_cast<uint32_t>(index_buffer_bytesize / sizeof(uint32_t));
  const auto max_field_index = make_uint3(field_size.x - 1,
                                          field_size.y - 1,
                                          field_size.z - 1);
  const auto scale = make_float3(1.f / (field_size.x - 1.f),
                                 1.f / (field_size.y - 1.f),
                                 1.f / (field_size.z - 1.f));
  if (ctx->indexed) {
    if (vertex_buffer && index_buffer) {
      switch (ctx->extraction_mode) {
      case ExtractionMode::Blocking: {
        CHECKED_CUDA(cudaStreamSynchronize(stream));
        uint32_t vertex_count = min(vertex_capacity, ctx->sum_h[1]);
        if (vertex_count) {
          extractIndexedVertexPN<<<(vertex_count + 255) / 256, 256, 0, stream>>>(vertex_buffer,
                                                                                 ctx->vertex_pyramid,
                                                                                 field_d,
                                                                                 ctx->index_cases_d,
                                                                                 ctx->tables->index_table,
                                                                                 field_row_stride,
                                                                                 field_slice_stride,
                                                                                 field_offset,
                                                                                 max_field_index,
                                                                                 vertex_capacity,
                                                                                 make_uint2(ctx->chunks.x, ctx->chunks.y),
                                                                                 scale,
                                                                                 threshold);

        }
        uint32_t index_count = min(index_capacity, ctx->sum_h[0]);
        if (index_count) {
          extractIndices<<<(index_count + 255) / 256, 256, 0, ctx->indexStream>>>(index_buffer,
                                                                                  ctx->vertex_pyramid,
                                                                                  ctx->index_pyramid,
                                                                                  ctx->index_cases_d,
                                                                                  ctx->tables->index_table,
                                                                                  index_capacity,
                                                                                  make_uint2(ctx->chunks.x, ctx->chunks.y));
        }
        break;
      }
      case ExtractionMode::DynamicParallelism:
        launchExtractIndexedVertexPN<<<1, 32, 0, stream>>>(vertex_buffer,
                                                           ctx->vertex_pyramid,
                                                           field_d,
                                                           ctx->index_cases_d,
                                                           ctx->tables->index_table,
                                                           field_row_stride,
                                                           field_slice_stride,
                                                           field_offset,
                                                           max_field_index,
                                                           vertex_capacity,
                                                           make_uint2(ctx->chunks.x, ctx->chunks.y),
                                                           scale,
                                                           threshold,
                                                           alwaysExtract);
        launchExtractIndices<<<1, 32, 0, ctx->indexStream>>>(index_buffer,
                                                             ctx->vertex_pyramid,
                                                             ctx->index_pyramid,
                                                             ctx->index_cases_d,
                                                             ctx->tables->index_table,
                                                             index_capacity,
                                                             make_uint2(ctx->chunks.x, ctx->chunks.y),
                                                             alwaysExtract);
        break;
      default:
        assert(false && "Unhandled extraction mode");
        break;
      }
      CHECKED_CUDA(cudaEventRecord(ctx->indexExtractDoneEvent, ctx->indexStream));
      CHECKED_CUDA(cudaStreamWaitEvent(stream, ctx->indexExtractDoneEvent, 0));
    }

  }
  else {
    if (vertex_buffer) {
      switch (ctx->extraction_mode) {
      case ExtractionMode::Blocking: {
        CHECKED_CUDA(cudaStreamSynchronize(stream));
        uint32_t vertex_count = min(vertex_capacity, ctx->sum_h[0]);
        if (vertex_count) {
          extractVertexPN<<<(vertex_count + 255) / 256, 256, 0, stream>>>(vertex_buffer,
                                                                          ctx->index_pyramid,
                                                                          field_d,
                                                                          ctx->index_cases_d,
                                                                          ctx->tables->index_table,
                                                                          field_row_stride,
                                                                          field_slice_stride,
                                                                          field_offset,
                                                                          max_field_index,
                                                                          vertex_capacity,
                                                                          make_uint2(ctx->chunks.x, ctx->chunks.y),
                                                                          scale,
                                                                          threshold);
        }
        break;
      }
      case ExtractionMode::DynamicParallelism:
        launchExtractVertexPN<<<1, 32, 0, stream>>>(vertex_buffer,
                                                    ctx->index_pyramid,
                                                    field_d,
                                                    ctx->index_cases_d,
                                                    ctx->tables->index_table,
                                                    field_row_stride,
                                                    field_slice_stride,
                                                    field_offset,
                                                    max_field_index,
                                                    vertex_capacity,
                                                    make_uint2(ctx->chunks.x, ctx->chunks.y),
                                                    scale,
                                                    threshold,
                                                    alwaysExtract);
        break;
      default:
        assert(false && "Unhandled extraction mode");
        break;
      }
    }
  }

}
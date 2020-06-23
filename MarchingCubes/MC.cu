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

  struct uint6 {
    uint32_t e0;
    uint32_t e1;
    uint32_t e2;
    uint32_t e3;
    uint32_t e4;
    uint32_t e5;
  };

  template<typename T>
  __device__ uint6 fetch(const T* ptr, const T* end, const size_t stride, const float t)
  {
    unsigned t0 = ((ptr < end) && (ptr[0] < t) ? 1 : 0) | ((ptr <= end) && (ptr[1] < t) ? 2 : 0); ptr += stride;
    unsigned t1 = ((ptr < end) && (ptr[0] < t) ? 1 : 0) | ((ptr <= end) && (ptr[1] < t) ? 2 : 0); ptr += stride;
    unsigned t2 = ((ptr < end) && (ptr[0] < t) ? 1 : 0) | ((ptr <= end) && (ptr[1] < t) ? 2 : 0); ptr += stride;
    unsigned t3 = ((ptr < end) && (ptr[0] < t) ? 1 : 0) | ((ptr <= end) && (ptr[1] < t) ? 2 : 0); ptr += stride;
    unsigned t4 = ((ptr < end) && (ptr[0] < t) ? 1 : 0) | ((ptr <= end) && (ptr[1] < t) ? 2 : 0); ptr += stride;
    unsigned t5 = ((ptr < end) && (ptr[0] < t) ? 1 : 0) | ((ptr <= end) && (ptr[1] < t) ? 2 : 0); ptr += stride;
    return { t0, t1, t2, t3, t4, t5 };
  }

  __device__ uint6 mergeZ(const uint6 z0, const uint6 z1)
  {
    return {
      (z1.e0 << 4) | (z0.e0),
      (z1.e1 << 4) | (z0.e1),
      (z1.e2 << 4) | (z0.e2),
      (z1.e3 << 4) | (z0.e3),
      (z1.e4 << 4) | (z0.e4),
      (z1.e5 << 4) | (z0.e5),
    };
  }

  __device__ uint5 mergeY(const uint6 y) {
    return {
      (y.e1 << 2) | (y.e0),
      (y.e2 << 2) | (y.e1),
      (y.e3 << 2) | (y.e2),
      (y.e4 << 2) | (y.e3),
      (y.e5 << 2) | (y.e4),
    };
  }

  template<class FieldType>
  __global__ __launch_bounds__(32) void reduce_base(uint8_t* __restrict__           cases_d,
                                                    uint4*   __restrict__           out_level0_d,
                                                    uint32_t* __restrict__          out_sideband_d,
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
    uint3 cell = make_uint3(32 * chunk.x + thread,
                            5 * chunk.y,
                            5 * chunk.z);

    // One warp processes a 32 * 5 * 5 chunk, outputs 800 mc cases to base level and 800/5=160 sums to sideband.
    const FieldType* ptr = field_d
                         + cell.z * field_slice_stride
                         + cell.y * field_row_stride
                         + cell.x;
    uint6 prev = fetch(ptr, field_end_d, field_row_stride, threshold);
    ptr += field_slice_stride;

    bool xmask = cell.x < cells.x;
    for (unsigned y = 0; y < 5; y++) {

      unsigned sum = 0;
      bool zmask = cell.z < cells.z;
      if (xmask && zmask) {

        uint6 next = fetch(ptr, field_end_d, field_row_stride, threshold);
        ptr += field_slice_stride;

        uint5 cases = mergeY(mergeZ(prev, next));
        prev = next;

        uint5 counts{
          cell.y + 0u < cells.y ? index_count[cases.e0] : 0u,
          cell.y + 1u < cells.y ? index_count[cases.e1] : 0u,
          cell.y + 2u < cells.y ? index_count[cases.e2] : 0u,
          cell.y + 3u < cells.y ? index_count[cases.e3] : 0u,
          cell.y + 4u < cells.y ? index_count[cases.e4] : 0u,
        };

        sum = counts.e0 + counts.e1 + counts.e2 + counts.e3 + counts.e4;

        // todo: check if sum is nonzero.
        cases_d[5 * (32 * 5 * blockIdx.x + 32 * y + thread) + 0] = cases.e0;
        cases_d[5 * (32 * 5 * blockIdx.x + 32 * y + thread) + 1] = cases.e1;
        cases_d[5 * (32 * 5 * blockIdx.x + 32 * y + thread) + 2] = cases.e2;
        cases_d[5 * (32 * 5 * blockIdx.x + 32 * y + thread) + 3] = cases.e3;
        cases_d[5 * (32 * 5 * blockIdx.x + 32 * y + thread) + 4] = cases.e4;
        out_level0_d[32 * 5 * blockIdx.x + 32 * y + thread] = make_uint4(cases.e0, cases.e1, cases.e2, cases.e3);
      }
      cell.z++;
      out_sideband_d[32 * (5 * blockIdx.x + y) + threadIdx.x] = sum;// t.x + t.y + t.z + t.w + sideband[5 * thread + 4];
    }
  }


  // Reads 160 values, outputs HP level of 128 values, and 32 sideband values.
  __global__  __launch_bounds__( 5* 32) void reduce1(uint4* __restrict__          hp1_d,  //< Each block will write 32 uvec4's into this
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
        hp1_d[offset1] = hp;
        sb1_d[offset1] = hp.x + hp.y + hp.z + hp.w + sb[5 * threadIdx.x + 4];
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
                                                           cudaStream_t stream)
{
  // Minimal nonzero size:
  //
  // L0    800
  // L1    160
  // L2     32
  // L3      7  In apex
  // L4      2  In apex


  //assert(tables);
  auto * ctx = new Context();

  ctx->tables = tables;
  ctx->cells = cells;
  fprintf(stderr, "Cells [%u x %u x %u]\n", ctx->cells.x, ctx->cells.y, ctx->cells.z);

  // Each chunk handles a set of 32 x 5 x 5 cells.
  ctx->chunks = make_uint3(((cells.x + 31) / 32),
                           ((cells.y + 4) / 5),
                           ((cells.z + 4) / 5));
  ctx->chunk_total = ctx->chunks.x * ctx->chunks.y * ctx->chunks.z;
  fprintf(stderr, "Chunks [%u x %u x %u] = %u\n",
          ctx->chunks.x, ctx->chunks.y, ctx->chunks.z, ctx->chunk_total);

  // cases array is of size 800 * chunk count:
  CHECKED_CUDA(cudaMalloc(&ctx->cases, sizeof(uint32_t) * 800 * ctx->chunk_total ));

  // Pyramid base level, as number of uvec4's:
  ctx->level_sizes[0] = (800 * ctx->chunk_total + 4) / 5;
  ctx->levels = 1 + static_cast<uint32_t>(std::ceil(std::log(ctx->level_sizes[0]) / std::log(5.0)));
  assert(4 <= ctx->levels);
  assert(ctx->levels < 32u);

  ctx->level_offsets[0] = 32; // Apex in front (32 uvec4's).
  for (unsigned l = 1; l < ctx->levels - 3; l++) {
    ctx->level_sizes[l]   = (ctx->level_sizes[l - 1] + 4) / 5;
    ctx->level_offsets[l] = ctx->level_offsets[l - 1] + ctx->level_sizes[l - 1];
  }
  ctx->total_size = ctx->level_offsets[ctx->levels - 4] + ctx->level_sizes[ctx->levels - 4];

  for (unsigned l = ctx->levels - 3; l < ctx->levels; l++) {
    ctx->level_sizes[l] = (ctx->level_sizes[l - 1] + 4) / 5;
  }
  assert(25 < ctx->level_offsets[ctx->levels - 4]);
  assert(ctx->level_offsets[ctx->levels - 3] <= 25);
  ctx->level_offsets[ctx->levels - 3] = 7; // up to 25 uvec4's
  ctx->level_offsets[ctx->levels - 2] = 2; // up to 5 uvec4's
  ctx->level_offsets[ctx->levels - 1] = 1; // one uvec4

  // Alloc pyramid
  CHECKED_CUDA(cudaMalloc(&ctx->pyramid, sizeof(uint4) * ctx->total_size));
  CHECKED_CUDA(cudaMemsetAsync(ctx->pyramid, 1, sizeof(uint4) * ctx->total_size, stream));

  // Alloc sideband 0
  size_t sideband0_size = ctx->level_sizes[0];
  fprintf(stderr, "sideband 0: %zu\n", sideband0_size);
  CHECKED_CUDA(cudaMalloc(&ctx->sidebands[0], sizeof(uint32_t) * sideband0_size));
  CHECKED_CUDA(cudaMemsetAsync(ctx->sidebands[0], 1, sizeof(uint32_t) * sideband0_size, stream));

  // Alloc sideband 1
  size_t sideband1_size = ctx->level_sizes[1];
  fprintf(stderr, "sideband 1: %zu\n", sideband1_size);
  CHECKED_CUDA(cudaMalloc(&ctx->sidebands[1], sizeof(uint32_t) * sideband1_size));
  CHECKED_CUDA(cudaMemsetAsync(ctx->sidebands[1], 1, sizeof(uint32_t) * sideband1_size, stream));

  for (unsigned l = 0; l < ctx->levels; l++) {
    fprintf(stderr, "[%d] %8d %8d  (%8d)\n", l, ctx->level_offsets[l], ctx->level_sizes[l], 4*ctx->level_sizes[l]);
  }
  fprintf(stderr, "Total %d\n", ctx->total_size);

  CHECKED_CUDA(cudaHostAlloc(&ctx->sum_h, sizeof(uint32_t), cudaHostAllocMapped));
  CHECKED_CUDA(cudaHostGetDevicePointer(&ctx->sum_d, ctx->sum_h, 0));


  return ctx;
}

void ComputeStuff::MC::destroyContext(Context* ctx)
{
  delete ctx;
}

uint32_t ComputeStuff::MC::buildP3(Context* ctx,
                                   void* output_buffer,
                                   size_t output_buffer_size,
                                   uint3 offset,
                                   uint3 field_size,
                                   const float* field_d,
                                   const float threshold,
                                   cudaStream_t stream)
{


  const unsigned chunks = ctx->chunk_total;
  fprintf(stderr, "chunks=%d, threads=%d\n", chunks, chunks * 32);
  fprintf(stderr, "moo=%u\n", (32 * 5 * 5 * chunks) / 4);
  fprintf(stderr, "moo=%u\n", 32 * 5 * chunks);
  ::reduce_base<float><<<chunks, 32, 0, stream>>>(ctx->cases,                           // block writes 5*5*32=800 uint8's
                                                  ctx->pyramid + ctx->level_offsets[0], // block writes 5*32 uvec4's
                                                  ctx->sidebands[0],                    // block writes 5*32 uint32's
                                                  ctx->tables->index_count,
                                                  field_d,
                                                  field_d + size_t(field_size.x) * field_size.y * field_size.z,
                                                  size_t(field_size.x),
                                                  size_t(field_size.x) * field_size.y,
                                                  threshold,
                                                  ctx->chunks,
                                                  ctx->cells);
  CHECKED_CUDA(cudaStreamSynchronize(stream));
  std::vector<uint32_t> tmp(ctx->level_sizes[0]);
  CHECKED_CUDA(cudaMemcpyAsync(tmp.data(), ctx->sidebands[0], sizeof(uint32_t) * tmp.size(), cudaMemcpyDeviceToHost, stream));
  CHECKED_CUDA(cudaStreamSynchronize(stream));

  unsigned sum = 0;
  for (auto& item : tmp) {
    sum += item;
  }
  fprintf(stderr, "sum = %d\n", sum);

  // Reduction: 
  // Input: level[i-1] elements
  // Output: 4*(N/5) into pyramid
  //            N/5  into 


  bool sb = true;
  for (unsigned i = 1; i < ctx->levels - 3; i++) {


    // Reads 160 values, outputs HP level of 128 values, and 32 sideband values.
    const auto blocks = (ctx->level_sizes[i] + 31) / 32;
    reduce1<<<blocks, 5*32, 0, stream>>>(ctx->pyramid + ctx->level_offsets[i], // Each block will write 32 uvec4's into this
                                         ctx->sidebands[sb ? 1 : 0],           // Each block will write 32 uint32's into this
                                         ctx->level_sizes[i],                  // Number of uvec4's in level i
                                         ctx->sidebands[sb ? 0 : 1],           // Input, each block will read 5*32=160 values from here
                                         ctx->level_sizes[i-1]);               // Number of sideband elements from level i-1
    CHECKED_CUDA(cudaStreamSynchronize(stream));

    std::vector<uint32_t> tmp(ctx->level_sizes[i]);
    CHECKED_CUDA(cudaMemcpyAsync(tmp.data(), ctx->sidebands[sb ? 1 : 0], sizeof(uint32_t) * tmp.size(), cudaMemcpyDeviceToHost, stream));
    CHECKED_CUDA(cudaStreamSynchronize(stream));

    unsigned sum = 0;
    for (auto& item : tmp) {
      sum += item;
    }
    fprintf(stderr, "%u sum = %d\n", i, sum);
    sb = !sb;
  }
  reduceApex<<<1, 4*32, 0, stream>>>(ctx->pyramid,
                                     ctx->sum_d,
                                     ctx->sidebands[sb ? 0 : 1],
                                     ctx->level_sizes[ctx->levels - 4]);

  CHECKED_CUDA(cudaStreamSynchronize(stream));
  fprintf(stderr, "final sum: %u\n", *ctx->sum_h);

  return 0;
}


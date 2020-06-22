#include "MC.h"
#include <cassert>
#include <cmath>

namespace {

  template<typename FieldType>
  struct reduce_base_args
  {
    uint4* __restrict__ out_level0_d;
    uint4* __restrict__ out_level1_d;
    uint32_t* __restrict__ out_sideband_d;
    const uint8_t* index_count;
    const FieldType* field_d;
    const FieldType* field_end_d;
    size_t field_row_stride;
    size_t field_slice_stride;
    float threshold;
    uint3 chunks;
    uint3 cells;
  };

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
  __global__ void __launch_bounds__(32)
    reduce_base(reduce_base_args<FieldType> arg)
  {
    const uint32_t warp = threadIdx.x / 32;
    const uint32_t thread = threadIdx.x % 32;

    const uint32_t chunk_ix = blockIdx.x + warp;

    uint3 chunk = make_uint3(chunk_ix % arg.chunks.x,
                            (chunk_ix / arg.chunks.x) % arg.chunks.y,
                            (chunk_ix / arg.chunks.x) / arg.chunks.y);
    uint3 cell = make_uint3(32 * chunk.x + thread,
                            5 * chunk.y,
                            5 * chunk.z);

    // One warp processes a 32 * 5 * 5 chunk, outputs 800 mc cases to base level and 800/5=160 sums to sideband.
    const FieldType* ptr = arg.field_d
                         + cell.z * arg.field_slice_stride
                         + cell.y * arg.field_row_stride
                         + cell.x;
    uint6 prev = fetch(ptr, arg.field_end_d, arg.field_row_stride, arg.threshold);
    ptr += arg.field_slice_stride;

    bool xmask = cell.x < arg.cells.x;
    for (unsigned y = 0; y < 5; y++) {

      unsigned sum = 0;
      bool zmask = cell.z < arg.cells.z;
      if (xmask && zmask) {

        uint6 next = fetch(ptr, arg.field_end_d, arg.field_row_stride, arg.threshold);
        ptr += arg.field_slice_stride;

        uint5 cases = mergeY(mergeZ(prev, next));
        prev = next;

        uint5 counts;

#if 1
        counts.e0 = cell.y + 0 < arg.cells.y ? arg.index_count[cases.e0 & 0xffu] : 0;
        counts.e1 = cell.y + 1 < arg.cells.y ? arg.index_count[cases.e1 & 0xffu] : 0;
        counts.e2 = cell.y + 2 < arg.cells.y ? arg.index_count[cases.e2 & 0xffu] : 0;
        counts.e3 = cell.y + 3 < arg.cells.y ? arg.index_count[cases.e3 & 0xffu] : 0;
        counts.e4 = cell.y + 4 < arg.cells.y ? arg.index_count[cases.e4 & 0xffu] : 0;
#else

        cases.e0 &= 0xffu;
        cases.e1 &= 0xffu;
        cases.e2 &= 0xffu;
        cases.e3 &= 0xffu;
        cases.e4 &= 0xffu;

        counts.e0 = (cell.y + 0 < arg.cells.y) ? ((cases.e0 != 0) && (cases.e0 != 255) ? 1 : 0) : 0;
        counts.e1 = (cell.y + 1 < arg.cells.y) ? ((cases.e1 != 0) && (cases.e1 != 255) ? 1 : 0) : 0;
        counts.e2 = (cell.y + 2 < arg.cells.y) ? ((cases.e2 != 0) && (cases.e2 != 255) ? 1 : 0) : 0;
        counts.e3 = (cell.y + 3 < arg.cells.y) ? ((cases.e3 != 0) && (cases.e3 != 255) ? 1 : 0) : 0;
        counts.e4 = (cell.y + 4 < arg.cells.y) ? ((cases.e4 != 0) && (cases.e4 != 255) ? 1 : 0) : 0;
#endif
        //out_level0_d[32 * 5 * blockIdx.x + 32 * y + thread] = make_uint4(count0, count1, count2, count3);
        sum = counts.e0 + counts.e1 + counts.e2 + counts.e3 + counts.e4;
      }
      cell.z++;
      arg.out_sideband_d[32 * (5 * blockIdx.x + y) + threadIdx.x] = sum;// t.x + t.y + t.z + t.w + sideband[5 * thread + 4];
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
  //assert(tables);
  auto * ctx = new Context();

  ctx->tables = tables;
  ctx->cells = cells;

  // Each chunk handles a set of 32 x 5 x 5 cells.
  ctx->chunks = make_uint3(((cells.x + 31) / 32),
                           ((cells.y + 4) / 5),
                           ((cells.z + 4) / 5));

  fprintf(stderr, "Cells [%u x %u x %u]\n", ctx->cells.x, ctx->cells.y, ctx->cells.z);
  fprintf(stderr, "Chunks [%u x %u x %u] -> [%u x %u x %u] cell extension\n",
          ctx->chunks.x, ctx->chunks.y, ctx->chunks.z,
          32*ctx->chunks.x, 5* ctx->chunks.y, 5* ctx->chunks.z);

  const uint32_t base_N = 32 * 5 * 5 * ctx->chunks.x * ctx->chunks.y * ctx->chunks.z;
  ctx->levels = 1 + static_cast<uint32_t>(std::ceil(std::log(base_N) / std::log(5.0)));
  assert(4 <= ctx->levels);
  assert(ctx->levels < 32u);

  auto n = base_N;
  for (unsigned l = 0; l < ctx->levels; l++) {
    ctx->level_sizes[ctx->levels - 1 - l] = n;
    n = (n + 4) / 5;
  }
  assert(1 == ctx->level_sizes[0]);
  assert(1 < ctx->level_sizes[1]);

  ctx->level_offsets[0] = 1;  // 1 element
  ctx->level_offsets[1] = 2;  // 5 elements
  ctx->level_offsets[2] = 7;  // 5*5 elements, 7+5*5 = 32;
  size_t off = 32;
  for (unsigned l = 3; l < ctx->levels; l++) {
    ctx->level_offsets[l] = off;
    off = 4 * ((off + ctx->level_sizes[l] + 3) / 4);
  }
  assert(off < std::numeric_limits<uint32_t>::max());
  ctx->total_size = off;


  CHECKED_CUDA(cudaMalloc(&ctx->buffer, sizeof(uint32_t)*ctx->total_size));

  CHECKED_CUDA(cudaMalloc(&ctx->sidebands[0], sizeof(uint32_t) * base_N / 5));
  CHECKED_CUDA(cudaMalloc(&ctx->sidebands[1], sizeof(uint32_t) * base_N / 25));
  CHECKED_CUDA(cudaMemsetAsync(ctx->sidebands[0], 1, sizeof(uint32_t) * base_N / 25, stream));


  for (unsigned l = 0; l < ctx->levels; l++) {
    fprintf(stderr, "[%d] %d %d\n", l, ctx->level_offsets[l], ctx->level_sizes[l]);
  }
  fprintf(stderr, "[%d] %d\n", ctx->levels, ctx->total_size);

  return ctx;
}

void ComputeStuff::MC::destroyContext(Context* ctx)
{
  delete ctx;
}

uint32_t ComputeStuff::MC::buildP3(Context* ctx,
                                   uint3 offset,
                                   uint3 field_size,
                                   const float* field_d,
                                   const float threshold,
                                   cudaStream_t stream)
{
  const auto blocks = ctx->chunks.x * ctx->chunks.y * ctx->chunks.z;

  fprintf(stderr, "blocks=%d, threads=%d\n", blocks, blocks * 160);

  struct reduce_base_args<float> reduce_base_args{};
  reduce_base_args.out_level0_d = reinterpret_cast<uint4*>(ctx->buffer + ctx->level_offsets[ctx->levels - 1]);
  reduce_base_args.out_level1_d = reinterpret_cast<uint4*>(ctx->buffer + ctx->level_offsets[ctx->levels - 2]);
  reduce_base_args.out_sideband_d = ctx->sidebands[0];
  reduce_base_args.index_count = ctx->tables->index_count;
  reduce_base_args.field_d = field_d;
  reduce_base_args.field_end_d = field_d + 10000000000000 + size_t(field_size.x) * field_size.y * field_size.z;
  reduce_base_args.field_row_stride = size_t(field_size.x);
  reduce_base_args.field_slice_stride = size_t(field_size.x) * field_size.y;
  reduce_base_args.threshold = threshold;
  reduce_base_args.chunks = ctx->chunks;
  reduce_base_args.cells = ctx->cells;
  void* reduce_base_args_ptr[1] = { &reduce_base_args };
  CHECKED_CUDA(cudaLaunchKernel(::reduce_base<float>,
                                make_uint3(ctx->chunks.x* ctx->chunks.y* ctx->chunks.z, 1, 1),
                                make_uint3(32, 1, 1),
                                (void**)&reduce_base_args_ptr,
                                0,
                                stream));

  std::vector<uint32_t> tmp(blocks * 160);
  CHECKED_CUDA(cudaMemcpyAsync(tmp.data(), ctx->sidebands[0], sizeof(uint32_t)*tmp.size(), cudaMemcpyDeviceToHost, stream));
  CHECKED_CUDA(cudaStreamSynchronize(stream));

  unsigned sum = 0;
  for (auto& item : tmp) {
    sum += item;
  }
  fprintf(stderr, "sum = %d\n", sum);


  return 0;
}


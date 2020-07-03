// This file is part of ComputeStuff copyright (C) 2020 Christopher Dyken.
// Released under the MIT license, please see LICENSE file for details.

#include "MC.h"
#include <cassert>
#include <cmath>

namespace {




  

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

  // Pyramid elements are uvec4's where the 5th element are stored in a sideband
  // that is used to build the next level.
  //
  // MC cells are processed in chunks of 32 x 5 x 5 elements, and since the minimum
  // extent is one chunk, minimum level is 5.
  //
  // The top 3 levels are called the apex, and are built in one go. Thus, there is
  // minimum 2 levels below the apex.
  //
  // level 0: ceil(800 / 5) = 160 uvec4's + sideband
  // level 1: ceil(160 / 5) =  32 uvec4's + sideband
  // level 2:  ceil(32 / 5) =   7 uvec4's + sideband
  // level 3:   ceil(7 / 5) =   2 uvec4's + sideband
  // level 4:   ceil(2 / 5) =   1 uvec4 + total sum
  ctx->level_sizes[0] = (800 * ctx->chunk_total + 4) / 5;
  ctx->levels = 1 + static_cast<uint32_t>(std::ceil(std::log(ctx->level_sizes[0]) / std::log(5.0)));
  assert(5 <= ctx->levels);
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
  ctx->level_offsets[ctx->levels - 3] = 4 + 7; // up to 25 uvec4's
  ctx->level_offsets[ctx->levels - 2] = 4 + 2; // up to 5 uvec4's
  ctx->level_offsets[ctx->levels - 1] = 4 + 1; // one uvec4
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
    CHECKED_CUDA(cudaEventCreateWithFlags(&ctx->indexExtractDoneEvent, cudaEventDisableTiming));

    CHECKED_CUDA(cudaMalloc(&ctx->vertex_pyramid, sizeof(uint4) * ctx->total_size));
    CHECKED_CUDA(cudaMalloc(&ctx->vertex_sidebands[0], sideband0_size));
    CHECKED_CUDA(cudaMalloc(&ctx->vertex_sidebands[1], sideband1_size));

    CHECKED_CUDA(cudaMemsetAsync(ctx->vertex_pyramid, 1, sizeof(uint4) * ctx->total_size, stream));
    CHECKED_CUDA(cudaMemsetAsync(ctx->vertex_sidebands[0], 1, sideband0_size, stream));
    CHECKED_CUDA(cudaMemsetAsync(ctx->vertex_sidebands[1], 1, sideband1_size, stream));

    CHECKED_CUDA(cudaMemcpyAsync(ctx->vertex_pyramid, ctx->level_offsets, sizeof(Context::level_offsets), cudaMemcpyHostToDevice, stream));
  }


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
  if (ctx->index_sidebands[1]) { CHECKED_CUDA(cudaFree(ctx->index_sidebands[1])); ctx->index_sidebands[1] = nullptr; }

  if (ctx->vertex_pyramid) { CHECKED_CUDA(cudaFree(ctx->vertex_pyramid)); ctx->vertex_pyramid = nullptr; }
  if (ctx->vertex_sidebands[0]) { CHECKED_CUDA(cudaFree(ctx->vertex_sidebands[0])); ctx->vertex_sidebands[0] = nullptr; }
  if (ctx->vertex_sidebands[1]) { CHECKED_CUDA(cudaFree(ctx->vertex_sidebands[1])); ctx->vertex_sidebands[1] = nullptr; }

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
    Internal::buildPyramid(ctx,
                           field_row_stride,
                           field_slice_stride,
                           field_offset,
                           field_size,
                           field_d,
                           threshold,
                           stream);
  }
  Internal::GenerateGeometryPN(ctx,
                               vertex_buffer,
                               index_buffer,
                               vertex_buffer_bytesize,
                               index_buffer_bytesize,
                               field_row_stride,
                               field_slice_stride,
                               field_offset,
                               field_size,
                               field_d,
                               threshold,
                               stream,
                               alwaysExtract);
}

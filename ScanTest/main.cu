#include <iostream>
#include <vector>
#include <cassert>

#include <Scan.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

namespace {

  void logFailure(cudaError_t error, const char *file, int line)
  {
    std::cerr << file << '@' << line << ": CUDA error: " << cudaGetErrorName(error) << std::endl;
    abort();
  }
}
#define assertSuccess(a) do { cudaError_t rv = (a); if(rv != cudaSuccess) logFailure(rv, __FILE__, __LINE__); } while(0)


int main()
{
  assertSuccess(cudaSetDevice(0));

  // zero-copy mem for total sum readback.

  uint32_t* sum_h;
  assertSuccess(cudaHostAlloc(&sum_h, sizeof(uint32_t), cudaHostAllocMapped));

  uint32_t* sum_d;
  assertSuccess(cudaHostGetDevicePointer(&sum_d, sum_h, 0));

  std::vector<uint32_t> offsets;
  std::vector<uint32_t> counts;
  std::vector<uint32_t> offsetsGold;

  std::vector<uint32_t> scratch;

  size_t N = 2 * 4 * 4 * 32;
  {
    uint32_t* offsets_d;
    uint32_t* scratch_d;
    uint32_t* counts_d;

    assertSuccess(cudaMalloc(&offsets_d, sizeof(uint32_t)*(N + 1)));
    assertSuccess(cudaMalloc(&scratch_d, ComputeStuff::Scan::scratchByteSize<uint32_t>(N)));
    assertSuccess(cudaMalloc(&counts_d, sizeof(uint32_t)*N));

    counts.resize(N);
    offsetsGold.resize(N + 1);
    offsetsGold[0] = 0;
    offsets.resize(N + 1);

    for (uint32_t modulo = 1; modulo < 10; modulo++) {
      std::cerr << "N=" << N << ", modulo=" << modulo << " scratch=" << ComputeStuff::Scan::scratchByteSize<uint32_t>(N) / sizeof(uint32_t) <<  std::endl;
      for (size_t i = 0; i < N; i++) {
        counts[i] = modulo==1 ? 1 : (i % modulo);
        offsetsGold[i + 1] = offsetsGold[i] + counts[i];
      }
      assertSuccess(cudaMemcpy(counts_d, counts.data(), sizeof(uint32_t)*N, cudaMemcpyHostToDevice));

      ComputeStuff::Scan::calcOffsets(offsets_d, scratch_d, counts_d, N);
      assertSuccess(cudaGetLastError());

#if 1
      scratch.resize(ComputeStuff::Scan::scratchByteSize<uint32_t>(N)/sizeof(uint32_t));
      assertSuccess(cudaMemcpy(scratch.data(), scratch_d, sizeof(uint32_t)*scratch.size(), cudaMemcpyDeviceToHost));
#endif

      assertSuccess(cudaMemcpy(offsets.data(), offsets_d, sizeof(uint32_t)*(N + 1), cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < N + 1; i++) {
        assert(offsets[i] == offsetsGold[i]);
      }

      ComputeStuff::Scan::calcOffsets(offsets_d, sum_d, scratch_d, counts_d, N);
      assertSuccess(cudaGetLastError());

      // Huh, cudaStreamSynchronize for stream 0 is needed for sum_h to be in sync.
      // I thought stream 0 was in sync...
      assertSuccess(cudaStreamSynchronize(0));
      assert(*((volatile uint32_t*)sum_h) == offsetsGold.back());

      assertSuccess(cudaMemcpy(offsets.data(), offsets_d, sizeof(uint32_t)*(N + 1), cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < N + 1; i++) {
        assert(offsets[i] == offsetsGold[i]);
      }

    }

    assertSuccess(cudaFree(counts_d));
    assertSuccess(cudaFree(scratch_d));
    assertSuccess(cudaFree(offsets_d));
  }

  assertSuccess(cudaFreeHost(sum_h));

  return 0;
}

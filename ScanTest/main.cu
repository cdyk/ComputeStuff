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


void runSize(uint32_t N)
{
  std::vector<uint32_t> offsets(N + 1);
  std::vector<uint32_t> counts(N);
  std::vector<uint32_t> offsetsGold(N + 1);
  std::vector<uint32_t> scratch;

  uint32_t* sum_h, * sum_d;
  assertSuccess(cudaHostAlloc(&sum_h, sizeof(uint32_t), cudaHostAllocMapped));
  assertSuccess(cudaHostGetDevicePointer(&sum_d, sum_h, 0));


  uint32_t* offsets_d;
  uint32_t* scratch_d;
  uint32_t* counts_d;
  assertSuccess(cudaMalloc(&offsets_d, sizeof(uint32_t)*(N + 1)));
  assertSuccess(cudaMalloc(&scratch_d, ComputeStuff::Scan::scratchByteSize(N)));
  assertSuccess(cudaMalloc(&counts_d, sizeof(uint32_t)*N));

  for (uint32_t modulo = 1; modulo < 10; modulo++) {
    std::cerr << "N=" << N << ", modulo=" << modulo << ", levels=" << ComputeStuff::Scan::levels(N) << ", scratch=" << ComputeStuff::Scan::scratchByteSize(N) / sizeof(uint32_t) << std::endl;

    offsetsGold[0] = 0;
    for (size_t i = 0; i < N; i++) {
      counts[i] = modulo == 1 ? 1 : (i % modulo);
      offsetsGold[i + 1] = offsetsGold[i] + counts[i];
    }
    assertSuccess(cudaMemcpy(counts_d, counts.data(), sizeof(uint32_t)*N, cudaMemcpyHostToDevice));

    ComputeStuff::Scan::calcOffsets(offsets_d, scratch_d, counts_d, N);
    assertSuccess(cudaStreamSynchronize(0));
    assertSuccess(cudaGetLastError());

#if 0
    scratch.resize(ComputeStuff::Scan::scratchByteSize(N) / sizeof(uint32_t));
    assertSuccess(cudaMemcpy(scratch.data(), scratch_d, sizeof(uint32_t)*scratch.size(), cudaMemcpyDeviceToHost));
#endif

    assertSuccess(cudaMemcpy(offsets.data(), offsets_d, sizeof(uint32_t)*(N + 1), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < N + 1; i++) {
      auto a = offsets[i];
      auto b = offsetsGold[i];
      assert(a == b);
    }

    ComputeStuff::Scan::calcOffsets(offsets_d, sum_d, scratch_d, counts_d, N);
    assertSuccess(cudaStreamSynchronize(0));
    assertSuccess(cudaGetLastError());

    assert(*((volatile uint32_t*)sum_h) == offsetsGold.back());

    assertSuccess(cudaMemcpy(offsets.data(), offsets_d, sizeof(uint32_t)*(N + 1), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N + 1; i++) {
      assert(offsets[i] == offsetsGold[i]);
    }
  }

  assertSuccess(cudaFree(counts_d));
  assertSuccess(cudaFree(scratch_d));
  assertSuccess(cudaFree(offsets_d));
  assertSuccess(cudaFreeHost(sum_h));
}


int main()
{
  assertSuccess(cudaSetDevice(0));

  runSize(static_cast<uint32_t>(0u));
  for (uint64_t N = 0; N < (uint64_t)(1 << 31 - 1); N = (N == 0 ? 1 : 7 * N + N / 3))
  {
    runSize(static_cast<uint32_t>(N));
  }
  //runSize(1 << 31 - 1);


  return 0;
}

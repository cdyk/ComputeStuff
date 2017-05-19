#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <vector>
#include <cassert>

#include <Scan.h>


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


void runTest(uint32_t N)
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

void runPerf(uint32_t N)
{
  cudaStream_t stream;
  assertSuccess(cudaStreamCreate(&stream));

  cudaEvent_t startA, stopA, startB, stopB;
  assertSuccess(cudaEventCreate(&startA));
  assertSuccess(cudaEventCreate(&startB));
  assertSuccess(cudaEventCreate(&stopA));
  assertSuccess(cudaEventCreate(&stopB));

  thrust::host_vector<uint32_t> in_h(N);
  std::vector<uint32_t> in_s(N);
  for (size_t i = 0; i < N; i++) {
    in_h[i] = in_s[i] = i % 3;
  }
  uint32_t* offsets_d;
  uint32_t* scratch_d;
  uint32_t* counts_d;
  assertSuccess(cudaMalloc(&offsets_d, sizeof(uint32_t)*(N + 1)));
  assertSuccess(cudaMalloc(&scratch_d, ComputeStuff::Scan::scratchByteSize(N)));
  assertSuccess(cudaMalloc(&counts_d, sizeof(uint32_t)*N));
  assertSuccess(cudaMemcpy(counts_d, in_s.data(), sizeof(uint32_t)*N, cudaMemcpyHostToDevice));

  thrust::device_vector<uint32_t> in_d = in_h;
  thrust::device_vector<uint32_t> out_d(N);

  // Run thrust::exclusive_scan
  for (uint32_t i = 0; i < 10; i++) {  // warm-up
    thrust::exclusive_scan(thrust::cuda::par.on(stream), in_d.begin(), in_d.end(), out_d.begin());
  }
  cudaEventRecord(startA, stream);
  for (uint32_t i = 0; i < 50; i++) {  // perf-run
    thrust::exclusive_scan(thrust::cuda::par.on(stream), in_d.begin(), in_d.end(), out_d.begin());
  }
  cudaEventRecord(stopA, stream);

  // Run ComputeStuff scan
  for (uint32_t i = 0; i < 10; i++) {  // warm-up
    ComputeStuff::Scan::calcOffsets(offsets_d, scratch_d, counts_d, N);
  }
  cudaEventRecord(startB, stream);
  for (uint32_t i = 0; i < 50; i++) {  // perf-run
    ComputeStuff::Scan::calcOffsets(offsets_d, scratch_d, counts_d, N);
  }
  cudaEventRecord(stopB, stream);

  cudaEventSynchronize(stopB);
  float elapsedA, elapsedB;
  assertSuccess(cudaEventElapsedTime(&elapsedA, startA, stopA));
  assertSuccess(cudaEventElapsedTime(&elapsedB, startB, stopB));

  std::cerr << "N=" << N << ",\tthrust=" << (elapsedA / 50.0) << "ms,\tComputeStuff=" << (elapsedB / 50.0) << "ms,\tratio CS/thrust=" << (elapsedB/elapsedA) << std::endl;
 

  assertSuccess(cudaFree(counts_d));
  assertSuccess(cudaFree(scratch_d));
  assertSuccess(cudaFree(offsets_d));



  assertSuccess(cudaStreamDestroy(stream));
  assertSuccess(cudaEventDestroy(startA));
  assertSuccess(cudaEventDestroy(startB));
  assertSuccess(cudaEventDestroy(stopA));
  assertSuccess(cudaEventDestroy(stopB));
}

int main(int argc, char** argv)
{
  bool perf = true;
  bool test = false;
  for (int i = 1; i < argc; i++) {
    if (strcmp("--perf", argv[i])) {
      perf = true;
    }
    else if (strcmp("--no-perf", argv[i])) {
      perf = false;
    }
    else if (strcmp("--test", argv[i])) {
      test = true;
    }
    else if (strcmp("--no-test", argv[i])) {
      test = false;
    }
  }

  assertSuccess(cudaSetDevice(0));

  if (test) {
    runTest(static_cast<uint32_t>(0u));
    for (uint64_t N = 1; N < (uint64_t)(1 << 31 - 1); N = (N == 0 ? 1 : 7 * N + N / 3))
    {
      runTest(static_cast<uint32_t>(N));
    }
    //runSize(1 << 31 - 1);
  }

  if (perf) {
    for (uint64_t N = 1; N < (uint64_t)(1 << 29 - 1); N = 3 * N + N / 3) {
      runPerf(static_cast<uint32_t>(N));
    }
  }


  return 0;
}

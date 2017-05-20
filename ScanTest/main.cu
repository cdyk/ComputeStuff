// This file is part of ComputeStuff copyright (C) 2017 Christopher Dyken.
// Released under the MIT license, please see LICENSE file for details.

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <vector>

#include <Scan.h>

namespace {

  bool test = true;
  bool perf = true;

  bool inclusiveScan = true;
  bool exclusiveScan = true;
  bool offsetTable = true;
  bool doCompact = true;


  void logFailure(cudaError_t error, const char *file, int line)
  {
    std::cerr << file << '@' << line << ": CUDA error: " << cudaGetErrorName(error) << std::endl;
    abort();
  }
}
#define assertSuccess(a) do { cudaError_t rv = (a); if(rv != cudaSuccess) logFailure(rv, __FILE__, __LINE__); } while(0)

void assertMatching(const uint32_t* result, const uint32_t* gold, uint32_t N)
{
  for (size_t i = 0; i < N; i++) {
    auto a = result[i];
    auto b = gold[i];
    if (a != b) {
      std::cerr << "a=" << a << " !=  b=" << b << std::endl;
      abort();
    }
  }
}

void runTest(uint32_t N)
{
  std::vector<uint32_t> offsets(N + 1);
  std::vector<uint32_t> counts(N);
  std::vector<uint32_t> offsetsGold(N + 1);

  std::vector<uint32_t> compact(N);
  std::vector<uint32_t> compactGold(N + 1);

  uint32_t* sum_h, *sum_d;
  assertSuccess(cudaHostAlloc(&sum_h, sizeof(uint32_t), cudaHostAllocMapped));
  assertSuccess(cudaHostGetDevicePointer(&sum_d, sum_h, 0));


  uint32_t* output_d;
  uint32_t* scratch_d;
  uint32_t* input_d;
  assertSuccess(cudaMalloc(&output_d, sizeof(uint32_t)*(N + 1)));
  assertSuccess(cudaMalloc(&scratch_d, ComputeStuff::Scan::scratchByteSize(N)));
  assertSuccess(cudaMalloc(&input_d, sizeof(uint32_t)*N));

  for (uint32_t modulo = 1; modulo < 10; modulo++) {
    std::cerr << "N=" << N << ", modulo=" << modulo << ", scratch=" << ComputeStuff::Scan::scratchByteSize(N) / sizeof(uint32_t) << std::endl;

    // Set up problem
    offsetsGold[0] = 0;
    uint32_t compactGold_sum = 0;
    for (size_t i = 0; i < N; i++) {
      counts[i] = modulo == 1 ? 1 : (i % modulo);
      offsetsGold[i + 1] = offsetsGold[i] + counts[i];

      compact[i] = modulo == 1 ? 1 : (i % modulo == 0 ? i + 1 : 0); // Any nonzero number flags surviving element.
      if (compact[i] != 0) {
        compactGold[compactGold_sum++] = i;
      }
    }
    assertSuccess(cudaMemcpy(input_d, counts.data(), sizeof(uint32_t)*N, cudaMemcpyHostToDevice));


    // Inclusive scan
    // --------------

    if (inclusiveScan) {

      // Disjoint input and output
      assertSuccess(cudaMemset(output_d, ~0, N * sizeof(uint32_t)));
      ComputeStuff::Scan::inclusiveScan(output_d, scratch_d, input_d, N);
      assertSuccess(cudaStreamSynchronize(0));
      assertSuccess(cudaMemcpy(offsets.data(), output_d, sizeof(uint32_t)*N, cudaMemcpyDeviceToHost));
      assertMatching(offsets.data(), offsetsGold.data() + 1, N);
    }

    // Exclusive scan
    // --------------

    if (exclusiveScan) {

      // Disjoint input and output
      assertSuccess(cudaMemset(output_d, ~0, N * sizeof(uint32_t)));
      ComputeStuff::Scan::exclusiveScan(output_d, scratch_d, input_d, N);
      assertSuccess(cudaStreamSynchronize(0));
      assertSuccess(cudaMemcpy(offsets.data(), output_d, sizeof(uint32_t)*N, cudaMemcpyDeviceToHost));
      assertMatching(offsets.data(), offsetsGold.data(), N);

      // In-place
      assertSuccess(cudaMemcpy(output_d, input_d, sizeof(uint32_t)*N, cudaMemcpyDeviceToDevice));
      ComputeStuff::Scan::exclusiveScan(output_d, scratch_d, output_d, N);
      assertSuccess(cudaStreamSynchronize(0));
      assertSuccess(cudaMemcpy(offsets.data(), output_d, sizeof(uint32_t)*N, cudaMemcpyDeviceToHost));
      assertMatching(offsets.data(), offsetsGold.data(), N);

    }

    // Offset table
    // ------------

    if (offsetTable) {

      // Offset without sum, disjoint input and output
      assertSuccess(cudaMemset(output_d, ~0, (N + 1) * sizeof(uint32_t)));
      ComputeStuff::Scan::calcOffsets(output_d, scratch_d, input_d, N);
      assertSuccess(cudaStreamSynchronize(0));
      assertSuccess(cudaMemcpy(offsets.data(), output_d, sizeof(uint32_t)*(N + 1), cudaMemcpyDeviceToHost));
      assertMatching(offsets.data(), offsetsGold.data(), N + 1);

      // Offset without sum, in-place
      assertSuccess(cudaMemcpy(output_d, input_d, sizeof(uint32_t)*N, cudaMemcpyDeviceToDevice));
      assertSuccess(cudaMemset(output_d + N, ~0, sizeof(uint32_t)));
      ComputeStuff::Scan::calcOffsets(output_d, scratch_d, output_d, N);
      assertSuccess(cudaStreamSynchronize(0));
      assertSuccess(cudaMemcpy(offsets.data(), output_d, sizeof(uint32_t)*(N + 1), cudaMemcpyDeviceToHost));
      assertMatching(offsets.data(), offsetsGold.data(), N + 1);

      // Offset with sum, disjoint input and output
      assertSuccess(cudaMemset(output_d, ~0, (N + 1) * sizeof(uint32_t)));
      *sum_h = ~0;
      ComputeStuff::Scan::calcOffsets(output_d, sum_d, scratch_d, input_d, N);
      assertSuccess(cudaStreamSynchronize(0));
      assertSuccess(cudaMemcpy(offsets.data(), output_d, sizeof(uint32_t)*(N + 1), cudaMemcpyDeviceToHost));
      assertMatching(offsets.data(), offsetsGold.data(), N + 1);
      if (*((volatile uint32_t*)sum_h) != offsetsGold.back()) {
        std::cerr << "Wrong sum." << std::endl;
        abort();
      }

      // Offset with sum, in-place
      assertSuccess(cudaMemcpy(output_d, input_d, sizeof(uint32_t)*N, cudaMemcpyDeviceToDevice));
      assertSuccess(cudaMemset(output_d + N, ~0, sizeof(uint32_t)));
      *sum_h = ~0;
      ComputeStuff::Scan::calcOffsets(output_d, sum_d, scratch_d, output_d, N);
      assertSuccess(cudaStreamSynchronize(0));
      assertSuccess(cudaMemcpy(offsets.data(), output_d, sizeof(uint32_t)*(N + 1), cudaMemcpyDeviceToHost));
      assertMatching(offsets.data(), offsetsGold.data(), N + 1);
      if (*((volatile uint32_t*)sum_h) != offsetsGold.back()) {
        std::cerr << "Wrong sum." << std::endl;
        abort();
      }
    }

    if (doCompact) {
      *sum_h = 0;
      assertSuccess(cudaMemcpy(input_d, compact.data(), sizeof(uint32_t)*N, cudaMemcpyHostToDevice));
      ComputeStuff::Scan::compact(output_d, sum_d, scratch_d, input_d, N);
      assertSuccess(cudaStreamSynchronize(0));
      assertSuccess(cudaMemcpy(offsets.data(), output_d, sizeof(uint32_t)*N, cudaMemcpyDeviceToHost));
#if 0
      assertMatching(offsets.data(), compactGold.data(), compactGold_sum);
#endif
      if (*((volatile uint32_t*)sum_h) != compactGold_sum) {
        std::cerr << "Wrong sum." << std::endl;
        abort();
      }
    }

  }

  assertSuccess(cudaFree(input_d));
  assertSuccess(cudaFree(scratch_d));
  assertSuccess(cudaFree(output_d));
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
  uint32_t* output_d;
  uint32_t* scratch_d;
  uint32_t* input_d;
  assertSuccess(cudaMalloc(&output_d, sizeof(uint32_t)*(N + 1)));
  assertSuccess(cudaMalloc(&scratch_d, ComputeStuff::Scan::scratchByteSize(N)));
  assertSuccess(cudaMalloc(&input_d, sizeof(uint32_t)*N));
  assertSuccess(cudaMemcpy(input_d, in_s.data(), sizeof(uint32_t)*N, cudaMemcpyHostToDevice));

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
    ComputeStuff::Scan::exclusiveScan(output_d, scratch_d, input_d, N, stream);
  }
  cudaEventRecord(startB, stream);
  for (uint32_t i = 0; i < 50; i++) {  // perf-run
    ComputeStuff::Scan::exclusiveScan(output_d, scratch_d, input_d, N, stream);
  }
  cudaEventRecord(stopB, stream);

  cudaEventSynchronize(stopB);
  float elapsedA, elapsedB;
  assertSuccess(cudaEventElapsedTime(&elapsedA, startA, stopA));
  assertSuccess(cudaEventElapsedTime(&elapsedB, startB, stopB));

  std::cerr << "|" << N << "|" << (elapsedA / 50.0) << "ms|" << (elapsedB / 50.0) << "ms|" << (elapsedB / elapsedA) << "|" << std::endl;


  assertSuccess(cudaFree(input_d));
  assertSuccess(cudaFree(scratch_d));
  assertSuccess(cudaFree(output_d));



  assertSuccess(cudaStreamDestroy(stream));
  assertSuccess(cudaEventDestroy(startA));
  assertSuccess(cudaEventDestroy(startB));
  assertSuccess(cudaEventDestroy(stopA));
  assertSuccess(cudaEventDestroy(stopB));
}

int main(int argc, char** argv)
{
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
    else if (strcmp("--inclusive-scan", argv[i])) {
      inclusiveScan = true;
    }
    else if (strcmp("--no-inclusive-scan", argv[i])) {
      inclusiveScan = false;
    }
    else if (strcmp("--exclusive-scan", argv[i])) {
      exclusiveScan = true;
    }
    else if (strcmp("--no-exclusive-scan", argv[i])) {
      exclusiveScan = false;
    }
    else if (strcmp("--offset-table", argv[i])) {
      offsetTable = true;
    }
    else if (strcmp("--no-offset-table", argv[i])) {
      offsetTable = false;
    }
  }

  std::cerr << "test=" << test << ", perf=" << perf << std::endl;
  std::cerr << "inclusive-scan=" << inclusiveScan << ", exclusive-scan=" << exclusiveScan << ", offset-table=" << offsetTable << std::endl;

  assertSuccess(cudaSetDevice(0));

  cudaDeviceProp props;
  assertSuccess(cudaGetDeviceProperties(&props, 0));
  if (props.major < 3) {
    std::cerr << "Compute capability 3.0 is minimum." << std::endl;
    return -1;
  }


  if (test) {
    for (uint64_t N = 1; N < (uint64_t)(props.totalGlobalMem / 10); N = (N == 0 ? 1 : 7 * N + N / 3))
    {
      runTest(static_cast<uint32_t>(N));
    }
  }

  if (perf) {
    std::cerr << "| N | thrust | ComputeStuff | ratio |" << std::endl;
    std::cerr << "|---|--------|--------------|-------|" << std::endl;
    for (uint64_t N = 1; N < (uint64_t)(props.totalGlobalMem / 10); N = 3 * N + N / 3) {
      runPerf(static_cast<uint32_t>(N));
    }
  }


  return 0;
}

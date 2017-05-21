// This file is part of ComputeStuff copyright (C) 2017 Christopher Dyken.
// Released under the MIT license, please see LICENSE file for details.

#include <vector>
#include <iomanip>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <HP5.h>
#include <Scan.h>

namespace {

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

void assertMatching(volatile uint32_t* a, uint32_t b)
{
  if (*a != b) {
    std::cerr << "a=" << *a << ", b=" << b << std::endl;
    abort();
  }
}

void buildCompactProblemWorstCase(std::vector<uint32_t>& out, uint32_t& sum, std::vector<uint32_t>& in, uint32_t N, uint32_t m)
{
  sum = 0;
  in.resize(N);
  out.resize(N);
  for (uint32_t i = 0; i < N; i++) {
    in[i] = m == 1 ? 1 : (i % m);
    if (in[i]) {
      out[sum++] = i;
    }
  }
}

void buildCompactProblemBestCase(std::vector<uint32_t>& out, uint32_t& sum, std::vector<uint32_t>& in, uint32_t N, uint32_t m)
{
  sum = 0;
  in.resize(N);
  out.resize(N);

  auto s = (N + m - 1) / m;
  for (uint32_t i = 0; i < s; i++) {
    in[i] = 1;
    out[sum++] = i;
  }
  for (uint32_t i = s; i < N; i++) {
    in[i] = 0;
  }
}

void runCompactTest(uint32_t N, uint32_t m)
{
  cudaStream_t stream;
  assertSuccess(cudaStreamCreate(&stream));

  cudaEvent_t startA, stopA, startB, stopB, startC, stopC, startD, stopD;
  assertSuccess(cudaEventCreate(&startA));
  assertSuccess(cudaEventCreate(&startB));
  assertSuccess(cudaEventCreate(&startC));
  assertSuccess(cudaEventCreate(&startD));
  assertSuccess(cudaEventCreate(&stopA));
  assertSuccess(cudaEventCreate(&stopB));
  assertSuccess(cudaEventCreate(&stopC));
  assertSuccess(cudaEventCreate(&stopD));

  uint32_t* sum_h, *sum_d;
  assertSuccess(cudaHostAlloc(&sum_h, sizeof(uint32_t), cudaHostAllocMapped));
  assertSuccess(cudaHostGetDevicePointer(&sum_d, sum_h, 0));

  uint32_t *out_d, *in_d, *hp5_scratch_d, *scan_scratch_d;
  assertSuccess(cudaMalloc(&out_d, sizeof(uint32_t)*N));
  assertSuccess(cudaMalloc(&in_d, sizeof(uint32_t)*N));
  assertSuccess(cudaMalloc(&hp5_scratch_d, ComputeStuff::HP5::scratchByteSize(N)));
  assertSuccess(cudaMalloc(&scan_scratch_d, ComputeStuff::Scan::scratchByteSize(N)));

  std::vector<uint32_t> out_h(N);

  uint32_t sum;
  std::vector<uint32_t> out, in;

  // Best case
  buildCompactProblemBestCase(out, sum, in, N, m);
  assertSuccess(cudaMemcpy(in_d, in.data(), sizeof(uint32_t)*N, cudaMemcpyHostToDevice));
  *sum_h = ~0u;
  for (uint32_t i = 0; i < 10; i++) {
    ComputeStuff::Scan::compact(out_d, sum_d, scan_scratch_d, in_d, N, stream);
  }
  cudaEventRecord(startA, stream);
  for (uint32_t i = 0; i < 50; i++) {
    ComputeStuff::Scan::compact(out_d, sum_d, scan_scratch_d, in_d, N, stream);
  }
  cudaEventRecord(stopA, stream);
  cudaEventSynchronize(stopA);
  cudaMemcpy(out_h.data(), out_d, sizeof(uint32_t)*N, cudaMemcpyDeviceToHost);

  assertMatching(sum_h, sum);
  assertMatching(out_h.data(), out.data(), sum);

  for (uint32_t i = 0; i < 10; i++) {
    ComputeStuff::HP5::compact(out_d, sum_d, hp5_scratch_d, in_d, N, stream);
  }
  cudaEventRecord(startB, stream);
  for (uint32_t i = 0; i < 50; i++) {
    ComputeStuff::HP5::compact(out_d, sum_d, hp5_scratch_d, in_d, N, stream);
  }
  cudaEventRecord(stopB, stream);
  cudaEventSynchronize(stopB);

  //assertMatching(sum_h, sum);
  //assertMatching(out_h.data(), out.data(), sum);

  // Worst case
  buildCompactProblemWorstCase(out, sum, in, N, m);
  assertSuccess(cudaMemcpy(in_d, in.data(), sizeof(uint32_t)*N, cudaMemcpyHostToDevice));
  *sum_h = ~0u;
  for (uint32_t i = 0; i < 10; i++) { // Warm-up
    ComputeStuff::Scan::compact(out_d, sum_d, scan_scratch_d, in_d, N, stream);
  }
  cudaEventRecord(startC, stream);
  for (uint32_t i = 0; i < 50; i++) { // Perf run
    ComputeStuff::Scan::compact(out_d, sum_d, scan_scratch_d, in_d, N, stream);
  }
  cudaEventRecord(stopC, stream);
  cudaEventSynchronize(stopC);
  cudaMemcpy(out_h.data(), out_d, sizeof(uint32_t)*N, cudaMemcpyDeviceToHost);

  assertMatching(sum_h, sum);
  assertMatching(out_h.data(), out.data(), sum);

  *sum_h = ~0u;
  for (uint32_t i = 0; i < 10; i++) { // Warm-up
    ComputeStuff::HP5::compact(out_d, sum_d, hp5_scratch_d, in_d, N, stream);
  }
  cudaEventRecord(startD, stream);
  for (uint32_t i = 0; i < 50; i++) { // Perf run
    ComputeStuff::HP5::compact(out_d, sum_d, hp5_scratch_d, in_d, N, stream);
  }
  cudaEventRecord(stopD, stream);
  cudaEventSynchronize(stopD);

  //assertMatching(sum_h, sum);
  //assertMatching(out_h.data(), out.data(), sum);


  float elapsedA, elapsedB, elapsedC, elapsedD;
  assertSuccess(cudaEventElapsedTime(&elapsedA, startA, stopA));
  assertSuccess(cudaEventElapsedTime(&elapsedB, startB, stopB));
  assertSuccess(cudaEventElapsedTime(&elapsedC, startC, stopC));
  assertSuccess(cudaEventElapsedTime(&elapsedD, startD, stopD));

  std::cerr << std::setprecision(3)
    << "| " << N << " | "
    << (int)(100/m) << "% | "
    << (elapsedA / 50.0) << "ms | "
    << (elapsedB / 50.0) << "ms | "
    << (elapsedB / elapsedA) << " | "
    << (elapsedC / 50.0) << "ms | "
    << (elapsedD / 50.0) << "ms | "
    << (elapsedD / elapsedC) << " | " << std::endl;

  assertSuccess(cudaEventDestroy(startA));
  assertSuccess(cudaEventDestroy(startB));
  assertSuccess(cudaEventDestroy(startC));
  assertSuccess(cudaEventDestroy(startD));
  assertSuccess(cudaEventDestroy(stopA));
  assertSuccess(cudaEventDestroy(stopB));
  assertSuccess(cudaEventDestroy(stopC));
  assertSuccess(cudaEventDestroy(stopD));
  assertSuccess(cudaFreeHost(sum_h));
  assertSuccess(cudaFree(out_d));
  assertSuccess(cudaFree(in_d));
  assertSuccess(cudaFree(scan_scratch_d));
  assertSuccess(cudaStreamDestroy(stream));
}

int main()
{
  assertSuccess(cudaSetDevice(0));

  cudaDeviceProp props;
  assertSuccess(cudaGetDeviceProperties(&props, 0));
  if (props.major < 3) {
    std::cerr << "Compute capability 3.0 is minimum." << std::endl;
    return -1;
  }


  for (uint64_t N = 1; N < (uint64_t)(props.totalGlobalMem / (sizeof(uint32_t) * 4)); N = 3 * N + N / 3) {
    for (uint32_t m = 1; m < 10; m++) {
      runCompactTest(static_cast<uint32_t>(N), m);
    }
  }

}
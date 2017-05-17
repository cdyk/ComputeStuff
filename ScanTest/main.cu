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

  size_t N = 4 * 4 * 32;
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
      std::cerr << "N=" << N << ", modulo=" << modulo << std::endl;
      for (size_t i = 0; i < N; i++) {
        counts[i] = modulo==1 ? 1 : (i % modulo);
        offsetsGold[i + 1] = offsetsGold[i] + counts[i];
      }
      assertSuccess(cudaMemcpy(counts_d, counts.data(), sizeof(uint32_t)*N, cudaMemcpyHostToDevice));

      ComputeStuff::Scan::calcOffsets(offsets_d, scratch_d, counts_d, N);
      assertSuccess(cudaGetLastError());
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

#if 0


  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    goto Error;
  }

  
  
  
  const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
#endif
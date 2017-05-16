#include <limits>
#include "Scan.h"

#define WARPSIZE 32
namespace {
  

  __global__
  __launch_bounds__(4 * WARPSIZE)
  void
  scan(uint32_t* output,
       const uint32_t* input,
       uint32_t N)
  {
    const uint32_t lane = threadIdx.x % WARPSIZE;
    const uint32_t blockOffset = 4 * 4 * WARPSIZE * blockIdx.x;
    const uint32_t threadOffset = blockOffset + 4 * threadIdx.x;

    // Fetch
    uint4 a;
    if (threadOffset + 3 < N) {
       a = *reinterpret_cast<const uint4*>(input + threadOffset);
    }
    else if (N <= threadOffset) {
      a = make_uint4(0, 0, 0, 0);
    }
    else {
      a.x = input[threadOffset];
      a.y = threadOffset + 1 < N ? input[threadOffset + 1] : 0;
      a.z = threadOffset + 2 < N ? input[threadOffset + 2] : 0;
      a.w = 0;
    }
    uint32_t s = a.x + a.y + a.z + a.w;

    // Per-warp reduce
    {
      uint32_t t;
      t = __shfl_up(s, 1); if (1 <= lane) s += t;
      t = __shfl_up(s, 2); if (2 <= lane) s += t;
      t = __shfl_up(s, 4); if (4 <= lane) s += t;
      t = __shfl_up(s, 8); if (8 <= lane) s += t;
      t = __shfl_up(s, 16); if (16 <= lane) s += t;
    }

    // Store
    if (threadOffset + 3 < N) {
      *reinterpret_cast<uint4*>(output + threadOffset) = make_uint4(s,
                                                                    s + a.x,
                                                                    s + a.x + a.y,
                                                                    s + a.x + a.y + a.z);
    }
    else if(threadOffset < N) {
      output[threadOffset + 0] = s;
      s += a.x;
      if (threadOffset + 1 < N) output[threadOffset + 1] = s;
      s += a.y;
      if (threadOffset + 2 < N) output[threadOffset + 2] = s;
    }

    
  }

}



template<>
size_t ComputeStuff::Scan::scratchByteSize<uint32_t>(size_t N)
{
  return sizeof(uint32_t)*42;
}

cudaError_t ComputeStuff::Scan::calcOffsets(uint32_t* offsets_d,
                                            uint32_t* sum_d,
                                            uint32_t* scratch_d,
                                            const uint32_t* counts_d,
                                            size_t N,
                                            cudaStream_t  stream)
{
  if (N <= std::numeric_limits<uint32_t>::max()) {
    uint32_t n = static_cast<uint32_t>(N);

    uint32_t blockSize = 4 * WARPSIZE;
    uint32_t blocks = (n + blockSize - 1) / blockSize;

    scan<<<blocks, blockSize, 0, stream >>>(offsets_d, counts_d, n);


    return cudaGetLastError();
  }
  else {
    return cudaErrorNotYetImplemented;
  }
}

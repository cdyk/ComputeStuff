#pragma once
// This file is part of ComputeStuff copyright (C) 2017 Christopher Dyken.
// Released under the MIT license, please see LICENSE file for details.

#define WARPSIZE 32
#define SCAN_WARPS 4

namespace ComputeStuff {
  namespace Scan {
    namespace Kernels {

      __global__
      __launch_bounds__(SCAN_WARPS * WARPSIZE)
      void
      reduce(uint32_t* output,
             const uint32_t* input,
             uint32_t N)
      {
        const uint32_t lane = threadIdx.x % WARPSIZE;
        const uint32_t warp = threadIdx.x / WARPSIZE;
        const uint32_t threadOffset = 4 * (SCAN_WARPS * WARPSIZE * blockIdx.x + threadIdx.x);

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
      #pragma unroll
        for (uint32_t i = 1; i < WARPSIZE; i *= 2) {
          uint32_t t = __shfl_up(s, i);
          if (i <= lane) {
            s += t;
          }
        }

        __shared__ uint32_t warpSum[SCAN_WARPS];
        if (lane == (WARPSIZE - 1)) {
          warpSum[warp] = s;
        }

        __syncthreads();

        // Aggregate warp sums and write total.
        if (threadIdx.x == 0) {
          auto a = warpSum[0];

#pragma unroll
          for (uint32_t i = 1; i < SCAN_WARPS; i++) {
            a += warpSum[i];
          }

          output[blockIdx.x] = a;
        }
      }

      template<bool inclusive, bool writeSum0, bool writeSum1, bool readOffset>
      __global__
      __launch_bounds__(SCAN_WARPS * WARPSIZE)
      void
      scan(uint32_t* output,
           uint32_t* sum0,
           uint32_t* sum1,
           const uint32_t* input,
           const uint32_t* offset,
           uint32_t N)
      {
        const uint32_t lane = threadIdx.x % WARPSIZE;
        const uint32_t warp = threadIdx.x / WARPSIZE;
        const uint32_t threadOffset = 4 * (SCAN_WARPS * WARPSIZE * blockIdx.x + threadIdx.x);

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
        uint32_t q = s;

        // Per-warp reduce
        #pragma unroll
        for (uint32_t i = 1; i < WARPSIZE; i *= 2) {
          uint32_t t = __shfl_up(s, i);
          if (i <= lane) {
            s += t;
          }
        }

        __shared__ uint32_t warpSum[SCAN_WARPS];
        if (lane == (WARPSIZE - 1)) {
          warpSum[warp] = s;
        }

        __syncthreads();

        #pragma unroll
        for (uint32_t w = 0; w < SCAN_WARPS - 1; w++) {
          if (w < warp) s += warpSum[w];
        }

        if (threadIdx.x == (SCAN_WARPS*WARPSIZE - 1)) {
          if (writeSum0) *sum0 = s;
          if (writeSum1) *sum1 = s;
        }

        if (inclusive == false) {
          s -= q; // exclusive scan
        }

        if (readOffset) {
          s += offset[blockIdx.x];
        }

        // Store
        if (threadOffset + 3 < N) {
          *reinterpret_cast<uint4*>(output + threadOffset) = make_uint4(s,
                                                                        s + a.x,
                                                                        s + a.x + a.y,
                                                                        s + a.x + a.y + a.z);
        }
        else if (threadOffset < N) {
          output[threadOffset + 0] = s;
          s += a.x;
          if (threadOffset + 1 < N) output[threadOffset + 1] = s;
          s += a.y;
          if (threadOffset + 2 < N) output[threadOffset + 2] = s;
        }

      }
    }
  }
}
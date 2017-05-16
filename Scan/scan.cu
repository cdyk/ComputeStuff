#include "Scan.h"



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
  return cudaSuccess;
}

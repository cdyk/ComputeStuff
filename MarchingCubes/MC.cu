#include "MC.h"

ComputeStuff::MC::Tables* ComputeStuff::MC::createTables(cudaStream_t streaam)
{
  return nullptr;
}


ComputeStuff::MC::HistoPyramid* ComputeStuff::MC::createHistoPyramid(cudaStream_t stream, Tables* tables, uint32_t nx, uint32_t ny, uint32_t nz)
{
  return nullptr;
}

void ComputeStuff::MC::buildHistoPyramid(cudaStream_t stream, HistoPyramid* hp, float iso)
{

}


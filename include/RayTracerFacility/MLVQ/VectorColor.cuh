#pragma once
#include <Optix7.hpp>
#include <SharedCoordinates.cuh>
namespace RayTracerFacility {
struct VectorColor {
  // the index from which we start the search
  int m_startIndex;
  // no. of channels describing one color (in our case usually 2 (CIE a-b))
  int m_numOfChannels;
  // the data array of a-b colors
  CudaBuffer m_vectorColorBasisBuffer;
  float *m_vectorColorBasis;
  // current number of stored a-b colors
  int m_numOfColors;

  void Init() {
    m_startIndex = 0;
    m_numOfColors = 0;
    m_numOfChannels = 2;
  }

  __device__ float Get(const int &colorIndex, const int &posAB,
                       SharedCoordinates &tc) const {
    assert(posAB >= 0 || posAB < m_numOfChannels);
    assert(colorIndex >= 0 && (colorIndex < m_numOfColors));
    return m_vectorColorBasis[colorIndex * m_numOfChannels + posAB];
  }
};
} // namespace RayTracerFacility
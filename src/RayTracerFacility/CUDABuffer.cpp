#include <CUDABuffer.hpp>
#include <Optix7.hpp>
#include <cuda_gl_interop.h>
#include <cuda.h>
#include <sstream>
#include <fstream>
#include <stdexcept>

#include <glm/glm.hpp>

using namespace RayTracerFacility;

CUdeviceptr CudaBuffer::DevicePointer() const {
    return reinterpret_cast<CUdeviceptr>(m_dPtr);
}

void CudaBuffer::Resize(const size_t &size) {
    if (this->m_sizeInBytes == size) return;
    Free();
    this->m_sizeInBytes = size;
    CUDA_CHECK(Malloc(&m_dPtr, m_sizeInBytes));
}

void CudaBuffer::Free() {
    if (m_dPtr == nullptr) return;
    CUDA_CHECK(Free(m_dPtr));
    m_dPtr = nullptr;
    m_sizeInBytes = 0;
}

void CudaBuffer::Upload(void *t, const size_t &size, const size_t &count) {
    Resize(count * size);
    CUDA_CHECK(Memcpy(m_dPtr, t, count * size, cudaMemcpyHostToDevice));
}

void CudaBuffer::Download(void *t, const size_t &size, const size_t &count) const {
    CUDA_CHECK(Memcpy(t, m_dPtr, count * size, cudaMemcpyDeviceToHost));
}

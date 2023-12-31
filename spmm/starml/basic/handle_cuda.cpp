#include "starml/basic/handle_cuda.h"

namespace starml {
CUDAHandle::CUDAHandle(DeviceIndex index) {
  index_ = index;
  STARML_CUDA_CHECK(cudaStreamCreate(&stream_));
}
CUDAHandle::~CUDAHandle() { STARML_CUDA_CHECK(cudaStreamDestroy(stream_)); }
void CUDAHandle::synchronized() const {
  STARML_CUDA_CHECK(cudaStreamSynchronize(stream_));
}
void* CUDAHandle::stream() const { return stream_; }
void CUDAHandle::switch_device() const {
  STARML_CUDA_CHECK(cudaSetDevice(index_));
}

Handle* CUDAHandleEntry::create_handle(DeviceIndex index) {
  CUDAHandle* handle = new CUDAHandle(index);
  return handle;
}

static CUDAHandleEntry g_cuda_handle_entry;
static HandleEntryRegister g_cuda_handle_entry_register(kCUDA, &g_cuda_handle_entry);

}  // namespace starml
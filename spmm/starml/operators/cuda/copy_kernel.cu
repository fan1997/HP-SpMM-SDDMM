#include "starml/operators/copy.h"
#include "starml/basic/common_cuda.h"
#include "starml/utils/loguru.h"

namespace starml {
namespace {
void copy_gpu_to_cpu_impl(const Matrix& src, Matrix& dst, Handle* handle) {
  auto dtype = src.data_type().type();
  STARML_DISPATCH_TYPES(dtype, "CUDA_COPY", [&]() {
    auto src_data = src.data<scalar_t>();
    auto dst_data = dst.mutable_data<scalar_t>();
    int nbytes = src.size() * sizeof(scalar_t);
    if (handle == NULL) {
      STARML_CUDA_CHECK(
          cudaMemcpy(dst_data, src_data, nbytes, cudaMemcpyDeviceToHost));
    }
    else {
      STARML_CUDA_CHECK(
          cudaMemcpyAsync(dst_data, src_data, nbytes, cudaMemcpyDeviceToHost,
                          reinterpret_cast<cudaStream_t>(handle->stream())));
    }
  });
}

void copy_cpu_to_gpu_impl(const Matrix& src, Matrix& dst, Handle* handle) {
  auto dtype = src.data_type().type();
  STARML_DISPATCH_TYPES(dtype, "CUDA_COPY", [&]() {
    auto src_data = src.data<scalar_t>();
    auto dst_data = dst.mutable_data<scalar_t>();
    int nbytes = src.size() * sizeof(scalar_t);
    if (handle == NULL) {
      STARML_CUDA_CHECK(
          cudaMemcpy(dst_data, src_data, nbytes, cudaMemcpyHostToDevice));
    }
    else {
      STARML_CUDA_CHECK(
          cudaMemcpyAsync(dst_data, src_data, nbytes, cudaMemcpyHostToDevice,
                          reinterpret_cast<cudaStream_t>(handle->stream())));
    }
  });
}

void copy_gpu_to_gpu_impl(const Matrix& src, Matrix& dst, Handle* handle) {
}
}  // namespace
STARML_REGISTER_KERNEL(copy_dispatcher, &copy_gpu_to_cpu_impl, kCUDA, kCPU);
STARML_REGISTER_KERNEL(copy_dispatcher, &copy_cpu_to_gpu_impl, kCPU, kCUDA);
STARML_REGISTER_KERNEL(copy_dispatcher, &copy_gpu_to_gpu_impl, kCUDA, kCUDA);
}  // namespace starml

#include "starml/basic/common_cuda.h"
#include "starml/operators/unary_ops.h"

namespace starml {
namespace {
template <typename TScalarType, typename TResultType, typename TOp>
__global__ void unary_kernel(const TScalarType* data, int start, int end,
                             TOp op, TResultType* result_data) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i + start < end) {
    *(result_data + i + start) = op(*(data + i + start));
  }
}

template <typename TScalarType, typename TResultType, typename TOp>
void eval_unary(const TScalarType* data, TResultType* result_data, int start,
                int end, Handle* handle, TOp op) {
  dim3 dimGrid(ceil((end - start) / 256.0), 1, 1);
  dim3 dimBlock(256, 1, 1);
  cudaStream_t stream = NULL;
  if (handle != NULL) {
    stream = reinterpret_cast<cudaStream_t>(handle->stream());
  }
  unary_kernel<<<dimGrid, dimBlock, 0, stream>>>(data, start, end, op,
                                                 result_data);
}

void exp_impl(const Matrix& matrix, Matrix& result, Handle* handle) {
  auto dtype = matrix.data_type().type();
  auto result_dtype = result.data_type().type();
  auto cast_dtype = (dtype < result_dtype) ? result_dtype : dtype;
  STARML_DISPATCH_TYPES(result_dtype, "EXP_CUDA", [&]() {
    auto result_data = result.mutable_data<scalar_t>();
    using result_scalar_type = scalar_t;
    STARML_DISPATCH_TYPES(dtype, "EXP_CUDA", [&]() {
      auto data = matrix.data<scalar_t>();
      using scalar_type = scalar_t;
      STARML_DISPATCH_FLOATING_TYPES(cast_dtype, "EXP_CUDA", [&]() {
        eval_unary(data, result_data, 0, result.size(), handle,
                   [=] __device__(scalar_type a) -> result_scalar_type {
                     return ::exp(scalar_t(a));
                   });
      });
    });
  });
}

void cast_impl(const Matrix& matrix, Matrix& result, Handle* handle) {
  auto dtype = matrix.data_type().type();
  auto result_dtype = result.data_type().type();
  STARML_DISPATCH_TYPES(result_dtype, "CUDA_CAST", [&]() {
    auto result_data = result.mutable_data<scalar_t>();
    using result_scalar_type = scalar_t;
    STARML_DISPATCH_TYPES(dtype, "CUDA_CAST", [&]() {
      auto data = matrix.data<scalar_t>();
      using scalar_type = scalar_t;
      eval_unary(
          data, result_data, 0, result.size(), handle,
          [=] __device__(scalar_type a) -> result_scalar_type { return a; });
    });
  });
}

}  // namespace
STARML_REGISTER_KERNEL(exp_dispatcher, &exp_impl, kCUDA, kCUDA);
STARML_REGISTER_KERNEL(cast_dispatcher, &cast_impl, kCUDA, kCUDA);
}  // namespace starml
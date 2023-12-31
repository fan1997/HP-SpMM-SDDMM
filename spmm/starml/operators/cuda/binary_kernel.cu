#include "index_helper.cuh"
#include "starml/basic/common_cuda.h"
#include "starml/operators/binary_ops.h"
#include "starml/operators/expression.h"

namespace starml {
namespace {
template <typename TScalarType1, typename TScalarType2, typename TResultType,
          typename TOp>
__global__ void binary_kernel(const TScalarType1* data1,
                              const TScalarType2* data2, int start, int end,
                              TOp op, IndexHelper data1_index_helper,
                              IndexHelper data2_index_helper,
                              IndexHelper result_index_helper,
                              TResultType* result_data) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i + start < end) {
    int data1_offset = data1_index_helper.index(i + start);
    int data2_offset = data2_index_helper.index(i + start);
    int result_offset = result_index_helper.index(i + start);
    *(result_data + result_offset) =
        op(*(data1 + data1_offset), *(data2 + data2_offset));
  }
}

template <typename TScalarType1, typename TScalarType2, typename TResultType,
          typename TOp>
void eval_binary(const TScalarType1* data1, const TScalarType2* data2,
                 TResultType* result_data, const Expression& expr, int start,
                 int end, Handle* handle, TOp op) {
  int ndims = expr.dims(0).size();
  IndexHelper data1_index_helper =
      IndexHelper(expr.dims(0).data(), expr.strides(0).data(), ndims);
  IndexHelper data2_index_helper =
      IndexHelper(expr.dims(1).data(), expr.strides(1).data(), ndims);
  IndexHelper result_index_helper =
      IndexHelper(expr.dims(2).data(), expr.strides(2).data(), ndims);
  dim3 dimGrid(ceil((end - start) / 256.0), 1, 1);
  dim3 dimBlock(256, 1, 1);
  cudaStream_t stream = NULL;
  if (handle != NULL) {
    stream = reinterpret_cast<cudaStream_t>(handle->stream());
  }
  binary_kernel<<<dimGrid, dimBlock, 0, stream>>>(
      data1, data2, start, end, op, data1_index_helper, data2_index_helper,
      result_index_helper, result_data);
}

void add_impl(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
              Handle* handle) {
  auto dtype1 = matrix1.data_type().type();
  auto dtype2 = matrix2.data_type().type();
  auto result_dtype = result.data_type().type();
  Expression expr = Expression(matrix1, matrix2, result);
  STARML_DISPATCH_TYPES(dtype1, "ADD_CUDA", [&]() {
    auto data1 = matrix1.data<scalar_t>();
    using scalar_type1 = scalar_t;
    STARML_DISPATCH_TYPES(dtype2, "ADD_CUDA", [&]() {
      auto data2 = matrix2.data<scalar_t>();
      using scalar_type2 = scalar_t;
      STARML_DISPATCH_TYPES(result_dtype, "ADD_CUDA", [&]() {
        auto result_data = result.mutable_data<scalar_t>();
        using result_scalar_type = scalar_t;
        eval_binary(data1, data2, result_data, expr, 0, result.size(), handle,
                    [=] __device__(scalar_type1 a, scalar_type2 b)
                        -> result_scalar_type { return a + b; });
      });
    });
  });
}

void sub_impl(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
              Handle* handle) {
  auto dtype1 = matrix1.data_type().type();
  auto dtype2 = matrix2.data_type().type();
  auto result_dtype = result.data_type().type();
  Expression expr = Expression(matrix1, matrix2, result);
  STARML_DISPATCH_TYPES(dtype1, "SUB_CUDA", [&]() {
    auto data1 = matrix1.data<scalar_t>();
    using scalar_type1 = scalar_t;
    STARML_DISPATCH_TYPES(dtype2, "SUB_CUDA", [&]() {
      auto data2 = matrix2.data<scalar_t>();
      using scalar_type2 = scalar_t;
      STARML_DISPATCH_TYPES(result_dtype, "SUB_CUDA", [&]() {
        auto result_data = result.mutable_data<scalar_t>();
        using result_scalar_type = scalar_t;
        eval_binary(data1, data2, result_data, expr, 0, result.size(), handle,
                    [=] __device__(scalar_type1 a, scalar_type2 b)
                        -> result_scalar_type { return a - b; });
      });
    });
  });
}

void mul_impl(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
              Handle* handle) {
  auto dtype1 = matrix1.data_type().type();
  auto dtype2 = matrix2.data_type().type();
  auto result_dtype = result.data_type().type();
  Expression expr = Expression(matrix1, matrix2, result);
  STARML_DISPATCH_TYPES(dtype1, "MUL_CUDA", [&]() {
    auto data1 = matrix1.data<scalar_t>();
    using scalar_type1 = scalar_t;
    STARML_DISPATCH_TYPES(dtype2, "MUL_CUDA", [&]() {
      auto data2 = matrix2.data<scalar_t>();
      using scalar_type2 = scalar_t;
      STARML_DISPATCH_TYPES(result_dtype, "MUL_CUDA", [&]() {
        auto result_data = result.mutable_data<scalar_t>();
        using result_scalar_type = scalar_t;
        eval_binary(data1, data2, result_data, expr, 0, result.size(), handle,
                    [=] __device__(scalar_type1 a, scalar_type2 b)
                        -> result_scalar_type { return a * b; });
      });
    });
  });
}

void div_impl(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
              Handle* handle) {
  auto dtype1 = matrix1.data_type().type();
  auto dtype2 = matrix2.data_type().type();
  auto result_dtype = result.data_type().type();
  Expression expr = Expression(matrix1, matrix2, result);
  STARML_DISPATCH_TYPES(dtype1, "DIV_CUDA", [&]() {
    auto data1 = matrix1.data<scalar_t>();
    using scalar_type1 = scalar_t;
    STARML_DISPATCH_TYPES(dtype2, "DIV_CUDA", [&]() {
      auto data2 = matrix2.data<scalar_t>();
      using scalar_type2 = scalar_t;
      STARML_DISPATCH_TYPES(result_dtype, "DIV_CUDA", [&]() {
        auto result_data = result.mutable_data<scalar_t>();
        using result_scalar_type = scalar_t;
        eval_binary(data1, data2, result_data, expr, 0, result.size(), handle,
                    [=] __device__(scalar_type1 a, scalar_type2 b)
                        -> result_scalar_type { return a / b; });
      });
    });
  });
}

void equal_impl(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
                Handle* handle) {
  auto dtype1 = matrix1.data_type().type();
  auto dtype2 = matrix2.data_type().type();
  auto result_dtype = result.data_type().type();
  Expression expr = Expression(matrix1, matrix2, result);
  STARML_DISPATCH_TYPES(dtype1, "EQUAL_CUDA", [&]() {
    auto data1 = matrix1.data<scalar_t>();
    using scalar_type1 = scalar_t;
    STARML_DISPATCH_TYPES(dtype2, "EQUAL_CUDA", [&]() {
      auto data2 = matrix2.data<scalar_t>();
      using scalar_type2 = scalar_t;
      STARML_DISPATCH_TYPES(result_dtype, "EQUAL_CUDA", [&]() {
        auto result_data = result.mutable_data<scalar_t>();
        using result_scalar_type = scalar_t;
        eval_binary(data1, data2, result_data, expr, 0, result.size(), handle,
                    [=] __device__(scalar_type1 a, scalar_type2 b)
                        -> result_scalar_type { return a == b; });
      });
    });
  });
}

void greater_impl(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
                  Handle* handle) {
  auto dtype1 = matrix1.data_type().type();
  auto dtype2 = matrix2.data_type().type();
  auto result_dtype = result.data_type().type();
  Expression expr = Expression(matrix1, matrix2, result);
  STARML_DISPATCH_TYPES(dtype1, "GREATER_CUDA", [&]() {
    auto data1 = matrix1.data<scalar_t>();
    using scalar_type1 = scalar_t;
    STARML_DISPATCH_TYPES(dtype2, "GREATER_CUDA", [&]() {
      auto data2 = matrix2.data<scalar_t>();
      using scalar_type2 = scalar_t;
      STARML_DISPATCH_TYPES(result_dtype, "GREATER_CUDA", [&]() {
        auto result_data = result.mutable_data<scalar_t>();
        using result_scalar_type = scalar_t;
        eval_binary(data1, data2, result_data, expr, 0, result.size(), handle,
                    [=] __device__(scalar_type1 a, scalar_type2 b)
                        -> result_scalar_type { return a > b; });
      });
    });
  });
}

void greater_equal_impl(const Matrix& matrix1, const Matrix& matrix2,
                        Matrix& result, Handle* handle) {
  auto dtype1 = matrix1.data_type().type();
  auto dtype2 = matrix2.data_type().type();
  auto result_dtype = result.data_type().type();
  Expression expr = Expression(matrix1, matrix2, result);
  STARML_DISPATCH_TYPES(dtype1, "GREATER_EQUAL_CUDA", [&]() {
    auto data1 = matrix1.data<scalar_t>();
    using scalar_type1 = scalar_t;
    STARML_DISPATCH_TYPES(dtype2, "GREATER_EQUAL_CUDA", [&]() {
      auto data2 = matrix2.data<scalar_t>();
      using scalar_type2 = scalar_t;
      STARML_DISPATCH_TYPES(result_dtype, "GREATER_EQUAL_CUDA", [&]() {
        auto result_data = result.mutable_data<scalar_t>();
        using result_scalar_type = scalar_t;
        eval_binary(data1, data2, result_data, expr, 0, result.size(), handle,
                    [=] __device__(scalar_type1 a, scalar_type2 b)
                        -> result_scalar_type { return a >= b; });
      });
    });
  });
}

void less_impl(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
               Handle* handle) {
  auto dtype1 = matrix1.data_type().type();
  auto dtype2 = matrix2.data_type().type();
  auto result_dtype = result.data_type().type();
  Expression expr = Expression(matrix1, matrix2, result);
  STARML_DISPATCH_TYPES(dtype1, "LESS_CUDA", [&]() {
    auto data1 = matrix1.data<scalar_t>();
    using scalar_type1 = scalar_t;
    STARML_DISPATCH_TYPES(dtype2, "LESS_CUDA", [&]() {
      auto data2 = matrix2.data<scalar_t>();
      using scalar_type2 = scalar_t;
      STARML_DISPATCH_TYPES(result_dtype, "LESS_CUDA", [&]() {
        auto result_data = result.mutable_data<scalar_t>();
        using result_scalar_type = scalar_t;
        eval_binary(data1, data2, result_data, expr, 0, result.size(), handle,
                    [=] __device__(scalar_type1 a, scalar_type2 b)
                        -> result_scalar_type { return a < b; });
      });
    });
  });
}

void less_equal_impl(const Matrix& matrix1, const Matrix& matrix2,
                     Matrix& result, Handle* handle) {
  auto dtype1 = matrix1.data_type().type();
  auto dtype2 = matrix2.data_type().type();
  auto result_dtype = result.data_type().type();
  Expression expr = Expression(matrix1, matrix2, result);
  STARML_DISPATCH_TYPES(dtype1, "LESS_EQUAL_CUDA", [&]() {
    auto data1 = matrix1.data<scalar_t>();
    using scalar_type1 = scalar_t;
    STARML_DISPATCH_TYPES(dtype2, "LESS_EQUAL_CUDA", [&]() {
      auto data2 = matrix2.data<scalar_t>();
      using scalar_type2 = scalar_t;
      STARML_DISPATCH_TYPES(result_dtype, "LESS_EQUAL_CUDA", [&]() {
        auto result_data = result.mutable_data<scalar_t>();
        using result_scalar_type = scalar_t;
        eval_binary(data1, data2, result_data, expr, 0, result.size(), handle,
                    [=] __device__(scalar_type1 a, scalar_type2 b)
                        -> result_scalar_type { return a <= b; });
      });
    });
  });
}

}  // namespace
STARML_REGISTER_KERNEL(add_dispatcher, &add_impl, kCUDA, kCUDA, kCUDA);
STARML_REGISTER_KERNEL(sub_dispatcher, &sub_impl, kCUDA, kCUDA, kCUDA);
STARML_REGISTER_KERNEL(mul_dispatcher, &mul_impl, kCUDA, kCUDA, kCUDA);
STARML_REGISTER_KERNEL(div_dispatcher, &div_impl, kCUDA, kCUDA, kCUDA);
STARML_REGISTER_KERNEL(equal_dispatcher, &equal_impl, kCUDA, kCUDA, kCUDA);
STARML_REGISTER_KERNEL(greater_dispatcher, &greater_impl, kCUDA, kCUDA, kCUDA);
STARML_REGISTER_KERNEL(greater_equal_dispatcher, &greater_equal_impl, kCUDA,
                       kCUDA, kCUDA);
STARML_REGISTER_KERNEL(less_dispatcher, &less_impl, kCUDA, kCUDA, kCUDA);
STARML_REGISTER_KERNEL(less_equal_dispatcher, &less_equal_impl, kCUDA, kCUDA,
                       kCUDA);
}  // namespace starml
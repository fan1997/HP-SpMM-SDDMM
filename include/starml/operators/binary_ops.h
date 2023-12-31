#pragma once
#include "starml/basic/dispatch.h"
#include "starml/basic/matrix.h"
#include "starml/basic/scalar.h"
namespace starml {
using binary_op_kernel_fn = void (*)(const Matrix& matrix1,
                                     const Matrix& matrix2, Matrix& result,
                                     Handle* handle);
STARML_DECLARE_DISPATCHER(add_dispatcher, binary_op_kernel_fn);
STARML_DECLARE_DISPATCHER(sub_dispatcher, binary_op_kernel_fn);
STARML_DECLARE_DISPATCHER(mul_dispatcher, binary_op_kernel_fn);
STARML_DECLARE_DISPATCHER(div_dispatcher, binary_op_kernel_fn);
STARML_DECLARE_DISPATCHER(equal_dispatcher, binary_op_kernel_fn);
STARML_DECLARE_DISPATCHER(greater_dispatcher, binary_op_kernel_fn);
STARML_DECLARE_DISPATCHER(greater_equal_dispatcher, binary_op_kernel_fn);
STARML_DECLARE_DISPATCHER(less_dispatcher, binary_op_kernel_fn);
STARML_DECLARE_DISPATCHER(less_equal_dispatcher, binary_op_kernel_fn);

Matrix add(const Matrix& matrix1, const Matrix& matrix2, Handle* handle = NULL);
Matrix sub(const Matrix& matrix1, const Matrix& matrix2, Handle* handle = NULL);
Matrix mul(const Matrix& matrix1, const Matrix& matrix2, Handle* handle = NULL);
Matrix div(const Matrix& matrix1, const Matrix& matrix2, Handle* handle = NULL);
Matrix equal(const Matrix& matrix1, const Matrix& matrix2,
             Handle* handle = NULL);
Matrix greater(const Matrix& matrix1, const Matrix& matrix2,
               Handle* handle = NULL);
Matrix greater_equal(const Matrix& matrix1, const Matrix& matrix2,
                     Handle* handle = NULL);
Matrix less(const Matrix& matrix1, const Matrix& matrix2,
            Handle* handle = NULL);
Matrix less_equal(const Matrix& matrix1, const Matrix& matrix2,
                  Handle* handle = NULL);

Matrix& add(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
            Handle* handle = NULL);
Matrix& sub(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
            Handle* handle = NULL);
Matrix& mul(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
            Handle* handle = NULL);
Matrix& div(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
            Handle* handle = NULL);
Matrix& equal(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
              Handle* handle = NULL);
Matrix& greater(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
                Handle* handle = NULL);
Matrix& greater_equal(const Matrix& matrix1, const Matrix& matrix2,
                      Matrix& result, Handle* handle = NULL);
Matrix& less(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
             Handle* handle = NULL);
Matrix& less_equal(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
                   Handle* handle = NULL);

Matrix add(const Scalar& scalar, const Matrix& matrix, Handle* handle = NULL);
Matrix sub(const Scalar& scalar, const Matrix& matrix, Handle* handle = NULL);
Matrix mul(const Scalar& scalar, const Matrix& matrix, Handle* handle = NULL);
Matrix div(const Scalar& scalar, const Matrix& matrix, Handle* handle = NULL);
Matrix equal(const Scalar& scalar, const Matrix& matrix, Handle* handle = NULL);
Matrix greater(const Scalar& scalar, const Matrix& matrix,
               Handle* handle = NULL);
Matrix greater_equal(const Scalar& scalar, const Matrix& matrix,
                     Handle* handle = NULL);
Matrix less(const Scalar& scalar, const Matrix& matrix, Handle* handle = NULL);
Matrix less_equal(const Scalar& scalar, const Matrix& matrix,
                  Handle* handle = NULL);

Matrix& add(const Scalar& scalar, const Matrix& matrix, Matrix& result,
            Handle* handle = NULL);
Matrix& sub(const Scalar& scalar, const Matrix& matrix, Matrix& result,
            Handle* handle = NULL);
Matrix& mul(const Scalar& scalar, const Matrix& matrix, Matrix& result,
            Handle* handle = NULL);
Matrix& div(const Scalar& scalar, const Matrix& matrix, Matrix& result,
            Handle* handle = NULL);
Matrix& equal(const Scalar& scalar, const Matrix& matrix, Matrix& result,
              Handle* handle = NULL);
Matrix& greater(const Scalar& scalar, const Matrix& matrix, Matrix& result,
                Handle* handle = NULL);
Matrix& greater_equal(const Scalar& scalar, const Matrix& matrix,
                      Matrix& result, Handle* handle = NULL);
Matrix& less(const Scalar& scalar, const Matrix& matrix, Matrix& result,
             Handle* handle = NULL);
Matrix& less_equal(const Scalar& scalar, const Matrix& matrix, Matrix& result,
                   Handle* handle = NULL);

Matrix add(const Matrix& matrix, const Scalar& scalar, Handle* handle = NULL);
Matrix sub(const Matrix& matrix, const Scalar& scalar, Handle* handle = NULL);
Matrix mul(const Matrix& matrix, const Scalar& scalar, Handle* handle = NULL);
Matrix div(const Matrix& matrix, const Scalar& scalar, Handle* handle = NULL);
Matrix equal(const Matrix& matrix, const Scalar& scalar, Handle* handle = NULL);
Matrix greater(const Matrix& matrix, const Scalar& scalar,
               Handle* handle = NULL);
Matrix greater_equal(const Matrix& matrix, const Scalar& scalar,
                     Handle* handle = NULL);
Matrix less(const Matrix& matrix, const Scalar& scalar, Handle* handle = NULL);
Matrix less_equal(const Matrix& matrix, const Scalar& scalar,
                  Handle* handle = NULL);

Matrix& add(const Matrix& matrix, const Scalar& scalar, Matrix& result,
            Handle* handle = NULL);
Matrix& sub(const Matrix& matrix, const Scalar& scalar, Matrix& result,
            Handle* handle = NULL);
Matrix& mul(const Matrix& matrix, const Scalar& scalar, Matrix& result,
            Handle* handle = NULL);
Matrix& div(const Matrix& matrix, const Scalar& scalar, Matrix& result,
            Handle* handle = NULL);
Matrix& equal(const Matrix& matrix, const Scalar& scalar, Matrix& result,
              Handle* handle = NULL);
Matrix& greater(const Matrix& matrix, const Scalar& scalar, Matrix& result,
                Handle* handle = NULL);
Matrix& greater_equal(const Matrix& matrix, const Scalar& scalar,
                      Matrix& result, Handle* handle = NULL);
Matrix& less(const Matrix& matrix, const Scalar& scalar, Matrix& result,
             Handle* handle = NULL);
Matrix& less_equal(const Matrix& matrix, const Scalar& scalar, Matrix& result,
                   Handle* handle = NULL);
}  // namespace starml
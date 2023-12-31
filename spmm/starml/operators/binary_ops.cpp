#include "starml/operators/binary_ops.h"

namespace starml {
STARML_DEFINE_DISPATCHER(add_dispatcher);
STARML_DEFINE_DISPATCHER(sub_dispatcher);
STARML_DEFINE_DISPATCHER(mul_dispatcher);
STARML_DEFINE_DISPATCHER(div_dispatcher);
STARML_DEFINE_DISPATCHER(equal_dispatcher);
STARML_DEFINE_DISPATCHER(greater_dispatcher);
STARML_DEFINE_DISPATCHER(greater_equal_dispatcher);
STARML_DEFINE_DISPATCHER(less_dispatcher);
STARML_DEFINE_DISPATCHER(less_equal_dispatcher);

Shape broadcast(const Matrix& matrix1, const Matrix& matrix2) {
  auto shape1 = matrix1.dims();
  auto shape2 = matrix2.dims();
  int ndims1 = matrix1.ndims();
  int ndims2 = matrix2.ndims();
  // if the trailing dimension of the two matrix are equal
  // or one of them equal to 1
  bool can_broadcast = true;
  for (int i = ndims1 - 1, j = ndims2 - 1; i >= 0 && j >= 0; i--, j--) {
    can_broadcast = can_broadcast && (shape1[i] == shape2[j] ||
                                      shape1[i] == 1 || shape2[j] == 1);
  }
  STARML_CHECK(can_broadcast) << "Operands could not be broadcast.";
  int result_dims = (ndims1 < ndims2) ? ndims2 : ndims1;
  Shape result = Shape(result_dims);
  int k = result_dims - 1;
  int i = ndims1 - 1;
  int j = ndims2 - 1;
  while (i >= 0 && j >= 0) {
    if (shape1[i] > shape2[j]) {
      result[k--] = shape1[i--];
      j--;
    } else {
      result[k--] = shape2[j--];
      i--;
    }
  }
  while (i >= 0) {
    result[k--] = shape1[i--];
  }
  while (j >= 0) {
    result[k--] = shape2[j--];
  }
  return result;
}

Matrix add(const Matrix& matrix1, const Matrix& matrix2, Handle* handle) {
  auto shape = broadcast(matrix1, matrix2);
  auto result_dtype = (matrix1.data_type().type() < matrix2.data_type().type())
                          ? matrix2.data_type().type()
                          : matrix1.data_type().type();
  Matrix result = Matrix(shape, matrix1.device(), result_dtype);
  add_dispatcher(matrix1, matrix2, result, handle);
  return result;
}

Matrix sub(const Matrix& matrix1, const Matrix& matrix2, Handle* handle) {
  auto shape = broadcast(matrix1, matrix2);
  auto result_dtype = (matrix1.data_type().type() < matrix2.data_type().type())
                          ? matrix2.data_type().type()
                          : matrix1.data_type().type();
  Matrix result = Matrix(shape, matrix1.device(), result_dtype);
  sub_dispatcher(matrix1, matrix2, result, handle);
  return result;
}

Matrix mul(const Matrix& matrix1, const Matrix& matrix2, Handle* handle) {
  auto shape = broadcast(matrix1, matrix2);
  auto result_dtype = (matrix1.data_type().type() < matrix2.data_type().type())
                          ? matrix2.data_type().type()
                          : matrix1.data_type().type();
  Matrix result = Matrix(shape, matrix1.device(), result_dtype);
  mul_dispatcher(matrix1, matrix2, result, handle);
  return result;
}

Matrix div(const Matrix& matrix1, const Matrix& matrix2, Handle* handle) {
  auto shape = broadcast(matrix1, matrix2);
  auto result_dtype = (matrix1.data_type().type() < matrix2.data_type().type())
                          ? matrix2.data_type().type()
                          : matrix1.data_type().type();
  Matrix result = Matrix(shape, matrix1.device(), result_dtype);
  div_dispatcher(matrix1, matrix2, result, handle);
  return result;
}

Matrix equal(const Matrix& matrix1, const Matrix& matrix2, Handle* handle) {
  auto shape = broadcast(matrix1, matrix2);
  auto result_dtype = (matrix1.data_type().type() < matrix2.data_type().type())
                          ? matrix2.data_type().type()
                          : matrix1.data_type().type();
  Matrix result = Matrix(shape, matrix1.device(), result_dtype);
  equal_dispatcher(matrix1, matrix2, result, handle);
  return result;
}

Matrix greater(const Matrix& matrix1, const Matrix& matrix2, Handle* handle) {
  auto shape = broadcast(matrix1, matrix2);
  auto result_dtype = (matrix1.data_type().type() < matrix2.data_type().type())
                          ? matrix2.data_type().type()
                          : matrix1.data_type().type();
  Matrix result = Matrix(shape, matrix1.device(), result_dtype);
  greater_dispatcher(matrix1, matrix2, result, handle);
  return result;
}

Matrix greater_equal(const Matrix& matrix1, const Matrix& matrix2,
                     Handle* handle) {
  auto shape = broadcast(matrix1, matrix2);
  auto result_dtype = (matrix1.data_type().type() < matrix2.data_type().type())
                          ? matrix2.data_type().type()
                          : matrix1.data_type().type();
  Matrix result = Matrix(shape, matrix1.device(), result_dtype);
  greater_equal_dispatcher(matrix1, matrix2, result, handle);
  return result;
}

Matrix less(const Matrix& matrix1, const Matrix& matrix2, Handle* handle) {
  auto shape = broadcast(matrix1, matrix2);
  auto result_dtype = (matrix1.data_type().type() < matrix2.data_type().type())
                          ? matrix2.data_type().type()
                          : matrix1.data_type().type();
  Matrix result = Matrix(shape, matrix1.device(), result_dtype);
  less_dispatcher(matrix1, matrix2, result, handle);
  return result;
}

Matrix less_equal(const Matrix& matrix1, const Matrix& matrix2,
                  Handle* handle) {
  auto shape = broadcast(matrix1, matrix2);
  auto result_dtype = (matrix1.data_type().type() < matrix2.data_type().type())
                          ? matrix2.data_type().type()
                          : matrix1.data_type().type();
  Matrix result = Matrix(shape, matrix1.device(), result_dtype);
  less_equal_dispatcher(matrix1, matrix2, result, handle);
  return result;
}

Matrix& add(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
            Handle* handle) {
  auto shape = broadcast(matrix1, matrix2);
  STARML_CHECK(shape == result.dims())
      << "Dimension of result for inplace operator should be well "
         "preallocated.";
  auto result_dtype = (matrix1.data_type().type() < matrix2.data_type().type())
                          ? matrix2.data_type().type()
                          : matrix1.data_type().type();
  STARML_CHECK(result_dtype <= result.data_type().type())
      << "Unexpected result data type";
  add_dispatcher(matrix1, matrix2, result, handle);
  return result;
}

Matrix& sub(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
            Handle* handle) {
  auto shape = broadcast(matrix1, matrix2);
  STARML_CHECK(shape == result.dims())
      << "Dimension of result for inplace operator should be well "
         "preallocated.";
  auto result_dtype = (matrix1.data_type().type() < matrix2.data_type().type())
                          ? matrix2.data_type().type()
                          : matrix1.data_type().type();
  STARML_CHECK(result_dtype <= result.data_type().type())
      << "Unexpected result data type";
  sub_dispatcher(matrix1, matrix2, result, handle);
  return result;
}

Matrix& mul(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
            Handle* handle) {
  auto shape = broadcast(matrix1, matrix2);
  STARML_CHECK(shape == result.dims())
      << "Dimension of result for inplace operator should be well "
         "preallocated.";
  auto result_dtype = (matrix1.data_type().type() < matrix2.data_type().type())
                          ? matrix2.data_type().type()
                          : matrix1.data_type().type();
  STARML_CHECK(result_dtype <= result.data_type().type())
      << "Unexpected result data type";
  mul_dispatcher(matrix1, matrix2, result, handle);
  return result;
}

Matrix& div(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
            Handle* handle) {
  auto shape = broadcast(matrix1, matrix2);
  STARML_CHECK(shape == result.dims())
      << "Dimension of result for inplace operator should be well "
         "preallocated.";
  auto result_dtype = (matrix1.data_type().type() < matrix2.data_type().type())
                          ? matrix2.data_type().type()
                          : matrix1.data_type().type();
  STARML_CHECK(result_dtype <= result.data_type().type())
      << "Unexpected result data type";
  div_dispatcher(matrix1, matrix2, result, handle);
  return result;
}

Matrix& equal(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
              Handle* handle) {
  auto shape = broadcast(matrix1, matrix2);
  STARML_CHECK(shape == result.dims())
      << "Dimension of result for inplace operator should be well "
         "preallocated.";
  auto result_dtype = (matrix1.data_type().type() < matrix2.data_type().type())
                          ? matrix2.data_type().type()
                          : matrix1.data_type().type();
  STARML_CHECK(result_dtype <= result.data_type().type())
      << "Unexpected result data type";
  equal_dispatcher(matrix1, matrix2, result, handle);
  return result;
}

Matrix& greater(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
                Handle* handle) {
  auto shape = broadcast(matrix1, matrix2);
  STARML_CHECK(shape == result.dims())
      << "Dimension of result for inplace operator should be well "
         "preallocated.";
  auto result_dtype = (matrix1.data_type().type() < matrix2.data_type().type())
                          ? matrix2.data_type().type()
                          : matrix1.data_type().type();
  STARML_CHECK(result_dtype <= result.data_type().type())
      << "Unexpected result data type";
  greater_dispatcher(matrix1, matrix2, result, handle);
  return result;
}

Matrix& greater_equal(const Matrix& matrix1, const Matrix& matrix2,
                      Matrix& result, Handle* handle) {
  auto shape = broadcast(matrix1, matrix2);
  STARML_CHECK(shape == result.dims())
      << "Dimension of result for inplace operator should be well "
         "preallocated.";
  auto result_dtype = (matrix1.data_type().type() < matrix2.data_type().type())
                          ? matrix2.data_type().type()
                          : matrix1.data_type().type();
  STARML_CHECK(result_dtype <= result.data_type().type())
      << "Unexpected result data type";
  greater_equal_dispatcher(matrix1, matrix2, result, handle);
  return result;
}

Matrix& less(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
             Handle* handle) {
  auto shape = broadcast(matrix1, matrix2);
  STARML_CHECK(shape == result.dims())
      << "Dimension of result for inplace operator should be well "
         "preallocated.";
  auto result_dtype = (matrix1.data_type().type() < matrix2.data_type().type())
                          ? matrix2.data_type().type()
                          : matrix1.data_type().type();
  STARML_CHECK(result_dtype <= result.data_type().type())
      << "Unexpected result data type";
  less_dispatcher(matrix1, matrix2, result, handle);
  return result;
}

Matrix& less_equal(const Matrix& matrix1, const Matrix& matrix2, Matrix& result,
                   Handle* handle) {
  auto shape = broadcast(matrix1, matrix2);
  STARML_CHECK(shape == result.dims())
      << "Dimension of result for inplace operator should be well "
         "preallocated.";
  auto result_dtype = (matrix1.data_type().type() < matrix2.data_type().type())
                          ? matrix2.data_type().type()
                          : matrix1.data_type().type();
  STARML_CHECK(result_dtype <= result.data_type().type())
      << "Unexpected result data type";
  less_equal_dispatcher(matrix1, matrix2, result, handle);
  return result;
}

Matrix add(const Scalar& scalar, const Matrix& matrix, Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return add(scalar.to_matrix(matrix.device(), result_dtype), matrix, handle);
}

Matrix sub(const Scalar& scalar, const Matrix& matrix, Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return sub(scalar.to_matrix(matrix.device(), result_dtype), matrix, handle);
}

Matrix mul(const Scalar& scalar, const Matrix& matrix, Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return mul(scalar.to_matrix(matrix.device(), result_dtype), matrix, handle);
}

Matrix div(const Scalar& scalar, const Matrix& matrix, Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return div(scalar.to_matrix(matrix.device(), result_dtype), matrix, handle);
}

Matrix equal(const Scalar& scalar, const Matrix& matrix, Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return equal(scalar.to_matrix(matrix.device(), result_dtype), matrix, handle);
}

Matrix greater(const Scalar& scalar, const Matrix& matrix, Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return greater(scalar.to_matrix(matrix.device(), result_dtype), matrix,
                 handle);
}

Matrix greater_equal(const Scalar& scalar, const Matrix& matrix,
                     Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return greater_equal(scalar.to_matrix(matrix.device(), result_dtype), matrix,
                       handle);
}

Matrix less(const Scalar& scalar, const Matrix& matrix, Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return less(scalar.to_matrix(matrix.device(), result_dtype), matrix, handle);
}

Matrix less_equal(const Scalar& scalar, const Matrix& matrix, Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return less_equal(scalar.to_matrix(matrix.device(), result_dtype), matrix,
                    handle);
}

Matrix add(const Matrix& matrix, const Scalar& scalar, Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return add(matrix, scalar.to_matrix(matrix.device(), result_dtype), handle);
}

Matrix sub(const Matrix& matrix, const Scalar& scalar, Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return sub(matrix, scalar.to_matrix(matrix.device(), result_dtype), handle);
}

Matrix mul(const Matrix& matrix, const Scalar& scalar, Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return mul(matrix, scalar.to_matrix(matrix.device(), result_dtype), handle);
}

Matrix div(const Matrix& matrix, const Scalar& scalar, Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return div(matrix, scalar.to_matrix(matrix.device(), result_dtype), handle);
}

Matrix equal(const Matrix& matrix, const Scalar& scalar, Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return equal(matrix, scalar.to_matrix(matrix.device(), result_dtype), handle);
}

Matrix greater(const Matrix& matrix, const Scalar& scalar, Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return greater(matrix, scalar.to_matrix(matrix.device(), result_dtype),
                 handle);
}

Matrix greater_equal(const Matrix& matrix, const Scalar& scalar,
                     Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return greater_equal(matrix, scalar.to_matrix(matrix.device(), result_dtype),
                       handle);
}

Matrix less(const Matrix& matrix, const Scalar& scalar, Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return less(matrix, scalar.to_matrix(matrix.device(), result_dtype), handle);
}

Matrix less_equal(const Matrix& matrix, const Scalar& scalar, Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return less_equal(matrix, scalar.to_matrix(matrix.device(), result_dtype),
                    handle);
}

Matrix& add(const Scalar& scalar, const Matrix& matrix, Matrix& result,
            Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return add(scalar.to_matrix(matrix.device(), result_dtype), matrix, result,
             handle);
}

Matrix& sub(const Scalar& scalar, const Matrix& matrix, Matrix& result,
            Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return sub(scalar.to_matrix(matrix.device(), result_dtype), matrix, result,
             handle);
}

Matrix& mul(const Scalar& scalar, const Matrix& matrix, Matrix& result,
            Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return mul(scalar.to_matrix(matrix.device(), result_dtype), matrix, result,
             handle);
}

Matrix& div(const Scalar& scalar, const Matrix& matrix, Matrix& result,
            Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return div(scalar.to_matrix(matrix.device(), result_dtype), matrix, result,
             handle);
}

Matrix& equal(const Scalar& scalar, const Matrix& matrix, Matrix& result,
              Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return equal(scalar.to_matrix(matrix.device(), result_dtype), matrix, result,
               handle);
}

Matrix& greater(const Scalar& scalar, const Matrix& matrix, Matrix& result,
                Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return greater(scalar.to_matrix(matrix.device(), result_dtype), matrix,
                 result, handle);
}

Matrix& greater_equal(const Scalar& scalar, const Matrix& matrix,
                      Matrix& result, Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return greater_equal(scalar.to_matrix(matrix.device(), result_dtype), matrix,
                       result, handle);
}

Matrix& less(const Scalar& scalar, const Matrix& matrix, Matrix& result,
             Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return less(scalar.to_matrix(matrix.device(), result_dtype), matrix, result,
              handle);
}

Matrix& less_equal(const Scalar& scalar, const Matrix& matrix, Matrix& result,
                   Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return less_equal(scalar.to_matrix(matrix.device(), result_dtype), matrix,
                    result, handle);
}

Matrix& add(const Matrix& matrix, const Scalar& scalar, Matrix& result,
            Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return add(matrix, scalar.to_matrix(matrix.device(), result_dtype), result,
             handle);
}

Matrix& sub(const Matrix& matrix, const Scalar& scalar, Matrix& result,
            Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return sub(matrix, scalar.to_matrix(matrix.device(), result_dtype), result,
             handle);
}

Matrix& mul(const Matrix& matrix, const Scalar& scalar, Matrix& result,
            Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return mul(matrix, scalar.to_matrix(matrix.device(), result_dtype), result,
             handle);
}

Matrix& div(const Matrix& matrix, const Scalar& scalar, Matrix& result,
            Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return div(matrix, scalar.to_matrix(matrix.device(), result_dtype), result,
             handle);
}

Matrix& equal(const Matrix& matrix, const Scalar& scalar, Matrix& result,
              Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return equal(matrix, scalar.to_matrix(matrix.device(), result_dtype), result,
               handle);
}

Matrix& greater(const Matrix& matrix, const Scalar& scalar, Matrix& result,
                Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return greater(matrix, scalar.to_matrix(matrix.device(), result_dtype),
                 result, handle);
}

Matrix& greater_equal(const Matrix& matrix, const Scalar& scalar,
                      Matrix& result, Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return greater_equal(matrix, scalar.to_matrix(matrix.device(), result_dtype),
                       result, handle);
}

Matrix& less(const Matrix& matrix, const Scalar& scalar, Matrix& result,
             Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return less(matrix, scalar.to_matrix(matrix.device(), result_dtype), result,
              handle);
}

Matrix& less_equal(const Matrix& matrix, const Scalar& scalar, Matrix& result,
                   Handle* handle) {
  auto result_dtype = (scalar.type() < matrix.data_type().type())
                          ? matrix.data_type().type()
                          : scalar.type();
  return less_equal(matrix, scalar.to_matrix(matrix.device(), result_dtype),
                    result, handle);
}

}  // namespace starml
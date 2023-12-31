#include "starml/operators/unary_ops.h"

namespace starml {
STARML_DEFINE_DISPATCHER(exp_dispatcher);
STARML_DEFINE_DISPATCHER(cast_dispatcher);

Matrix exp(const Matrix& matrix, Handle* handle) {
  // If the input matrix is int, then the result data type should be float
  // else the data type of result should be consistent with the input
  auto result_dtype =
      (matrix.data_type().is_int()) ? kFloat : matrix.data_type().type();
  Matrix result = Matrix(matrix.dims(), matrix.device(), result_dtype);
  exp_dispatcher(matrix, result, handle);
  return result;
}

Matrix cast(const Matrix& matrix, const DataType& data_type, Handle* handle) {
  Matrix result = Matrix(matrix.dims(), matrix.device(), data_type);
  cast_dispatcher(matrix, result, handle);
  return result;
}

Matrix& exp(const Matrix& matrix, Matrix& result, Handle* handle) {
  exp_dispatcher(matrix, result, handle);
  return result;
}

Matrix& cast(const Matrix& matrix, Matrix& result, Handle* handle) {
  cast_dispatcher(matrix, result, handle);
  return result;
}

}  // namespace starml
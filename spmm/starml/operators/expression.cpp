#include "starml/operators/expression.h"
namespace starml {
Expression::Expression(const Matrix& matrix1, const Matrix& matrix2,
                       Matrix& result) {
  int ndims1 = matrix1.ndims();
  int ndims2 = matrix2.ndims();
  int result_ndims = result.ndims();
  Shape shape1 = matrix1.dims();
  Shape shape2 = matrix2.dims();
  Shape result_shape = result.dims();
  Shape expand_shape1 = Shape(result_ndims, 1);
  Shape expand_shape2 = Shape(result_ndims, 1);
  for (int i = 1; i <= result_ndims; i++) {
    if (ndims1 - i >= 0) {
      expand_shape1[result_ndims - i] = shape1[ndims1 - i];
    }
    if (ndims2 - i >= 0) {
      expand_shape2[result_ndims - i] = shape2[ndims2 - i];
    }
  }
  Shape stride1 = Shape(result_ndims, 1);
  Shape stride2 = Shape(result_ndims, 1);
  Shape result_stride = Shape(result_ndims, 1);
  for (int i = result_ndims - 2; i >= 0; i--) {
    stride1[i] = stride1[i + 1] * expand_shape1[i + 1];
    stride2[i] = stride2[i + 1] * expand_shape2[i + 1];
    result_stride[i] = result_stride[i + 1] * result_shape[i + 1];
  }
  for (int i = result_ndims - 1; i >= 0; i--) {
    stride1[i] = expand_shape1[i] == 1 ? 0 : stride1[i];
    stride2[i] = expand_shape2[i] == 1 ? 0 : stride2[i];
    result_stride[i] = result_shape[i] == 1 ? 0 : result_stride[i];
  }
  strides_list_.push_back(stride1);
  strides_list_.push_back(stride2); 
  strides_list_.push_back(result_stride);
  dims_list_.push_back(expand_shape1);
  dims_list_.push_back(expand_shape2);
  dims_list_.push_back(result_shape);
}

Shape Expression::strides(int i) const { return strides_list_[i]; }
Shape Expression::dims(int i) const { return dims_list_[i]; }
}  // namespace starml
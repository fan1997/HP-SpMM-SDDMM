#pragma once
#include "starml/basic/matrix.h"
namespace starml {
class Expression {
 public:
  Expression(const Matrix& matrix1, const Matrix& matrix2, Matrix& result);
  Shape strides(int i) const ;
  Shape dims(int i) const;
 private:
  std::vector<Shape> strides_list_;
  std::vector<Shape> dims_list_;
};
}  // namespace starml
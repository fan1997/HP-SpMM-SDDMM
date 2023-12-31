#include "starml/operators/copy.h"

namespace starml {
STARML_DEFINE_DISPATCHER(copy_dispatcher);

Matrix deep_copy(const Matrix& src, const Device& new_device, Handle* handle) {
  Matrix result = Matrix(src.dims(), new_device, src.data_type());
  copy_dispatcher(src, result, handle);
  return result;
}
}  // namespace starml
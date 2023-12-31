#include "starml/operators/factories.h"
#include "starml/basic/dispatch.h"

namespace starml {
STARML_DEFINE_DISPATCHER(full_dispatcher);
Matrix full(const Shape& shape, const Device& device, const DataType& data_type,
            const Scalar& init_val) {
  Matrix result(shape, device, data_type);
  full_dispatcher(init_val, result);
  return result;
}
STARML_DEFINE_DISPATCHER(rand_dispatcher);
Matrix rand(const Shape& shape, const Device& device, const DataType& data_type,
            const Scalar& start, const Scalar& end) {
  Matrix result(shape, device, data_type);
  rand_dispatcher(start, end, result);
  return result;
}

Matrix empty(const Shape& shape, const Device& device,
             const DataType& data_type) {
  Matrix result(shape, device, data_type);
  return result;
}

}  // namespace starml
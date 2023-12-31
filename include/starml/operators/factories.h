#pragma once
#include "starml/basic/matrix.h"
#include "starml/basic/scalar.h"
#include "starml/basic/dispatch.h"

namespace starml {
using full_kernel_fn = void (*)(const Scalar& init_val, Matrix& result);
STARML_DECLARE_DISPATCHER(full_dispatcher, full_kernel_fn);
using rand_kernel_fn = void (*)(const Scalar& start, const Scalar& end, Matrix& result);
STARML_DECLARE_DISPATCHER(rand_dispatcher, rand_kernel_fn);

Matrix full(const Shape& shape, const Device& device, const DataType& data_type,
            const Scalar& init_val);
Matrix rand(const Shape& shape, const Device& device, const DataType& data_type,
            const Scalar& start, const Scalar& end);
Matrix empty(const Shape& shape, const Device& device,
             const DataType& data_type);
}  // namespace starml
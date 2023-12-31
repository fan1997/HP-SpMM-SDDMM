#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include "starml/operators/factories.h"
#include <iostream>
namespace starml {
namespace {

void full_impl(const Scalar& init_val, Matrix& result) {
  int size = result.size();
  auto dtype = result.data_type().type();
  STARML_DISPATCH_TYPES(dtype, "FULL_CUDA", [&]() {
    auto data = result.mutable_data<scalar_t>();
    scalar_t value = init_val.value<scalar_t>();
    thrust::device_ptr<scalar_t> dev_ptr(data);
    thrust::fill(dev_ptr, dev_ptr + size, value);
  });
}
}  // namespace
STARML_REGISTER_KERNEL(full_dispatcher, &full_impl, kCUDA);
}  // namespace starml
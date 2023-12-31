#include "starml/operators/factories.h"
#include <random>
namespace starml {
namespace {
void full_impl(const Scalar& init_val, Matrix& result) {
  int size = result.size();
  auto dtype = result.data_type().type();
  STARML_DISPATCH_TYPES(dtype, "FULL_CPU", [&]() {
    auto data = result.mutable_data<scalar_t>();
    scalar_t value = init_val.value<scalar_t>();
    std::fill(data, data + size, value);
  });
}
void rand_impl(const Scalar& start, const Scalar& end, Matrix& result) {
  int size = result.size();
  auto dtype = result.data_type().type();
  STARML_DISPATCH_TYPES(dtype, "RAND_CPU", [&]() {
    auto data = result.mutable_data<scalar_t>();
    scalar_t start_val = start.value<scalar_t>();
    scalar_t end_val = end.value<scalar_t>();
    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_real_distribution<> distr(start_val, end_val); // Define the range
    for (int i = 0; i < size; i++) {
      data[i] = distr(gen);
    }
  });
}
}  // namespace
STARML_REGISTER_KERNEL(full_dispatcher, &full_impl, kCPU);
STARML_REGISTER_KERNEL(rand_dispatcher, &rand_impl, kCPU);
}  // namespace starml
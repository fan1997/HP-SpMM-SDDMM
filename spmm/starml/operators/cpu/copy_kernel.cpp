#include <string.h>
#include "starml/operators/copy.h"

namespace starml {
namespace {
void copy_impl(const Matrix& src, Matrix& dst, Handle* handle) {
  memcpy(dst.raw_mutable_data(), src.raw_data(),
         src.size() * src.data_type().size());
}
}  // namespace
STARML_REGISTER_KERNEL(copy_dispatcher, &copy_impl, kCPU, kCPU);
}  // namespace starml

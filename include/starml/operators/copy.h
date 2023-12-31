#pragma once
#include "starml/basic/matrix.h"
#include "starml/basic/dispatch.h"

namespace starml {
using copy_kernel_fn = void (*)(const Matrix& src, Matrix& dst, Handle* handle);
STARML_DECLARE_DISPATCHER(copy_dispatcher, copy_kernel_fn);

Matrix deep_copy(const Matrix& src, const Device& new_device, Handle* handle);
}  // namespace starml
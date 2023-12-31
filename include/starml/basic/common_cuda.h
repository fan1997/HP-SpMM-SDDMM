#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "starml/utils/loguru.h"

namespace starml {
#define STARML_CUDA_CHECK(condition) \
  cudaError_t error = condition;     \
  STARML_CHECK(error == cudaSuccess) << cudaGetErrorString(error)

}  // namespace starml
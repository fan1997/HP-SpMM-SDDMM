#pragma once
#include <vector>
#include "starml/basic/common_cuda.h"

namespace starml {
class IndexHelper {
 public:
  IndexHelper(const int* dims, const int* strides, int ndims) : ndims_(ndims) {
    for (int i = 0; i < ndims_; i++) {
      dims_[i] = dims[i];
      strides_[i] = strides[i];
    }
  }
  __device__ int index(int idx) {
    int result = 0;
    for (int i = ndims_ - 1; i >= 0; i--) {
      result += (idx % dims_[i]) * strides_[i];
      idx /= dims_[i];
    }
    return result;
  }
  static constexpr int MAX_DIMS_NUM = 2;
 private:
  int ndims_;
  int dims_[MAX_DIMS_NUM];
  int strides_[MAX_DIMS_NUM];
};
}  // namespace starml
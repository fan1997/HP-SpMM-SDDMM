#pragma once
#include <vector>

namespace starml {
class IndexHelper {
 public:
  IndexHelper(const std::vector<int>& dims, const std::vector<int>& strides)
      : dims_(dims), strides_(strides) {}
  int index(int idx) {
    int n = dims_.size();
    int result = 0;
    for (int i = n - 1; i >= 0; i--) {
      result += (idx % dims_[i]) * strides_[i];
      idx /= dims_[i];
    }
    return result;
  }

 private:
  std::vector<int> dims_;
  std::vector<int> strides_;
};
}  // namespace starml
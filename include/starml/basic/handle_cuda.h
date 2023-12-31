#pragma once
#include "starml/basic/common_cuda.h"
#include "starml/basic/handle.h"
#include "starml/basic/device.h"

namespace starml {
class CUDAHandle : public Handle {
 public:
  CUDAHandle(DeviceIndex index_ = 0);
  ~CUDAHandle();
  void synchronized() const override;
  void* stream() const override;
  void switch_device() const override;

 private:
  cudaStream_t stream_;
  DeviceIndex index_;
};

class CUDAHandleEntry : public HandleEntry {
 public:
  Handle* create_handle(DeviceIndex index = 0) override;
};

}  // namespace starml
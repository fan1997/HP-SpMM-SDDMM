#pragma once
#include "starml/basic/handle.h"
#include "starml/basic/device.h"

namespace starml {
class CPUHandle : public Handle {
 public:
  CPUHandle(DeviceIndex index_ = 0);
  ~CPUHandle() = default;
  void synchronized() const override;
  void* stream() const override;
  void switch_device() const override;

 private:
  DeviceIndex index_;
};

class CPUHandleEntry : public HandleEntry {
 public:
  Handle* create_handle(DeviceIndex index = 0) override;
};

}  // namespace starml
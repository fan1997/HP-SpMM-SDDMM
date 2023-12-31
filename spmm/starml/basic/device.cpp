#include "starml/basic/device.h"
#include "starml/utils/loguru.h"

namespace starml {
std::string to_string(DeviceType d, bool lower_case) {
  switch (d) {
    case DeviceType::CPU:
      return lower_case ? "cpu" : "CPU";
    case DeviceType::CUDA:
      return lower_case ? "cuda" : "CUDA";
    default:
      STARML_LOG(ERROR) << "Unknown device: " << static_cast<int>(d);
      return "";
  }
}

std::ostream& operator<<(std::ostream& os, DeviceType type) {
  os << to_string(type, true);
  return os;
}

Device::Device(DeviceType type, int index) : type_(type), index_(index) {
  STARML_CHECK(index_ >= 0)
      << "Device index must be non-negative, got " << index_;
  STARML_CHECK(!is_cpu() || index_ == 0)
      << "CPU device index must be zero, got " << index_;
}

DeviceIndex Device::index() const { return this->index_; }
void Device::set_index(DeviceIndex new_index) { this->index_ = new_index; }

DeviceType Device::type() const { return this->type_; }
void Device::set_type(DeviceType new_type) { this->type_ = new_type; }

bool Device::is_cuda() const { return this->type_ == DeviceType::CUDA; }
bool Device::is_cpu() const { return this->type_ == DeviceType::CPU; }

bool Device::operator==(const Device& rhs) {
  return (this->type_ == rhs.type_) && (this->index_ == rhs.index_);
}
bool Device::operator!=(const Device& rhs) { return !((*this) == rhs); }

}  // namespace starml
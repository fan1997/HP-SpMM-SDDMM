#pragma once
#include <ostream>

namespace starml {
// Supported device type and the total nums of device types
enum class DeviceType : int {
  UNCERTAIN = -1,
  CPU = 0,
  CUDA = 1,
  NumDeviceTypes
};
// The constexpr specifier declares that it is possible to evaluate
// the value of the variable at compile time. Since the value of enum
// will be known during compiling time.
constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::CUDA;
constexpr int kNumDeviceTypes = static_cast<int>(DeviceType::NumDeviceTypes);

// In order to print or return the right device type string
// lower_case is a flag which indicates the spelling style
// of the string.
std::string to_string(DeviceType d, bool lower_case);
std::ostream& operator<<(std::ostream& stream, DeviceType type);

using DeviceIndex = int;

// Maintain the attributes relative to the Device
class Device {
 public:
  // DeviceType should be supplied while creating a new instance.
  // Check whether the input parameter is valid, index is greater or equal
  // than 0 indicates the index when multiple devices(GPU) are available.
  Device(DeviceType type = DeviceType::UNCERTAIN, DeviceIndex index = 0);
  ~Device() = default;
  Device(const Device&) = default;
  Device& operator=(const Device&) = default;
  // Setter/Getter of DeviceType
  DeviceType type() const;
  void set_type(DeviceType new_type);
  // Setter/Getter of index
  DeviceIndex index() const;
  void set_index(int new_index);
  // Return whether the device instance is cuda or cpu
  bool is_cuda() const;
  bool is_cpu() const;
  bool operator==(const Device& rhs);
  bool operator!=(const Device& rhs);

 private:
  DeviceType type_;
  int index_;
};
// Overload of output function for object Device
std::ostream& operator<<(std::ostream& os, const Device& device);

}  // namespace starml
#include "starml/basic/handle.h"
#include "starml/utils/loguru.h"

namespace starml {

HandleEntryRegistry& HandleEntryRegistry::singleton() {
  static HandleEntryRegistry handle_registry;
  return handle_registry;
}

void HandleEntryRegistry::set_handle_entry(DeviceType device_type,
                                           HandleEntry* handle_entry) {
  std::lock_guard<std::mutex> guard(mu_);
  STARML_CHECK_NE(device_type, DeviceType::UNCERTAIN)
      << "The device type is uncertain";
  handle_entries_[static_cast<int>(device_type)] = handle_entry;
}

HandleEntry* HandleEntryRegistry::handle_entry(DeviceType device_type) {
  HandleEntry* handle_entry = handle_entries_[static_cast<int>(device_type)];
  STARML_CHECK_NOTNULL(handle_entry) << "Allocator for " << device_type << " is not set.";
  return handle_entry;
}

HandleEntryRegister::HandleEntryRegister(DeviceType device_type,
                                         HandleEntry* handle_entry) {
  HandleEntryRegistry::singleton().set_handle_entry(device_type, handle_entry);
}

HandleEntry* get_handle_entry(DeviceType device_type) {
  HandleEntry* handle_entry =
      HandleEntryRegistry::singleton().handle_entry(device_type);
  return handle_entry;
}
}  // namespace starml
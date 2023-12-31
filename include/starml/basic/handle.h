#pragma once
#include <mutex>
#include <unordered_map>

#include "starml/basic/device.h"

namespace starml {
class Handle {
 public:
  virtual void* stream() const = 0;
  virtual void synchronized() const = 0;
  virtual void switch_device() const = 0;
};

class HandleEntry {
 public:
  virtual Handle* create_handle(DeviceIndex index = 0) = 0;
};

class HandleEntryRegistry {
 public:
  static HandleEntryRegistry& singleton();
  void set_handle_entry(DeviceType device_type, HandleEntry* handle_entry);
  HandleEntry* handle_entry(DeviceType device_type);

 private:
  HandleEntryRegistry() = default;
  HandleEntryRegistry(const HandleEntryRegistry&) = delete;
  HandleEntryRegistry& operator=(const HandleEntryRegistry&) = delete;
  std::unordered_map<int, HandleEntry*> handle_entries_;
  std::mutex mu_;
};

class HandleEntryRegister {
 public:
  HandleEntryRegister(DeviceType device_type, HandleEntry* handle_entry);
};
HandleEntry* get_handle_entry(DeviceType device_type);
}  // namespace starml
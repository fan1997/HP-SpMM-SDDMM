#include "starml/basic/allocator.h"
#include "starml/basic/device.h"
#include "starml/utils/loguru.h"

namespace starml {
void Allocator::deallocate_raw(void* ptr) const {
  auto deleter = raw_deleter();
  deleter(ptr);
}

DataPtr Allocator::allocate(size_t num_bytes) const {
  // Although using pure pointer to initial shared_pointer is not
  // suitable, here ensure a pure pointer matches a shared_pointer
  // the error causing by two shared_ptr point to a same pure pointer
  // will not happen. 
  void* raw_ptr = allocate_raw(num_bytes);
  auto deleter = raw_deleter();
  return {raw_ptr, deleter};
}

AllocatorRegistry& AllocatorRegistry::singleton() {
  static AllocatorRegistry alloc_registry;
  return alloc_registry;
}

void AllocatorRegistry::set_allocator(DeviceType device_type,
                                      Allocator* allocator) {
  std::lock_guard<std::mutex> guard(mu_);
  STARML_CHECK_NE(device_type, DeviceType::UNCERTAIN)
      << "The device type is uncertain";
  allocators_[static_cast<int>(device_type)] = allocator;
}

Allocator* AllocatorRegistry::allocator(DeviceType device_type) {
  Allocator* allocator = allocators_[static_cast<int>(device_type)];
  STARML_CHECK_NOTNULL(allocator) << "Allocator for " << device_type << " is not set.";
  return allocator;
}

AllocatorRegister::AllocatorRegister(DeviceType device_type,
                                     Allocator* allocator) {
  AllocatorRegistry::singleton().set_allocator(device_type, allocator);
}

Allocator* get_allocator(DeviceType device_type) {
  Allocator *alloc = AllocatorRegistry::singleton().allocator(device_type);
  return alloc;
}
}  // namespace starml
#pragma once
#include <cstddef>
#include <unordered_map>
#include <mutex>
#include <memory>
#include "starml/basic/device.h"

namespace starml {
// Unified function pointer for free allocated memory, 
// the only input parameter is the pointer to be free.
using DeleterFnPtr = void (*)(void*);
// In order to avoid memory leak, using shared_ptr as the underground
// type for the data of Matrix.
typedef std::shared_ptr<void> DataPtr; 

// Abstract base class of all device-allocators, in order to manage
// allocate/deallocate polymorphically.
class Allocator {
 public:
  // Deconstructor should be virtual since base class has virtual functions. 
  virtual ~Allocator() = default;
  // Allocate a given `num_bytes` 1D-linear space on specific device, return a pointer 
  // which point to the new allocated space.
  virtual void* allocate_raw(size_t num_bytes) const = 0;
  // Define and return the function pointer which used to free the space on given device. 
  virtual DeleterFnPtr raw_deleter() const = 0;
  // Using the proper deleter function to free the space where the parameter pointer point to.
  void deallocate_raw(void* ptr) const; 
  // Initial a shared_ptr with suitable deleter to manage the given space.
  DataPtr allocate(size_t num_bytes) const;
};

// Singleton pattern used to maintain the only AlocatorRegistry table instance.
// Device id and its corresponding derived pointer will be put into the table
// for further used, polymorphic is used here.
// Previous set for derived pointer will separate the device correlation while compiling.
class AllocatorRegistry {
 public:
  // Get the reference of the only instance. Static object instance will be instantiate
  // only once if it is not exists. (static instance pointer will cause memory leak)
  static AllocatorRegistry& singleton(); 
  // Set the table entry using the given device_type and allocator.
  void set_allocator(DeviceType device_type, Allocator* allocator);
  // Get the derived pointer of the given device type from table.
  Allocator* allocator(DeviceType device_type);
 private:
  // Prevent explicit construct, copy assignment, copy constructor
  AllocatorRegistry() = default;
  AllocatorRegistry(const AllocatorRegistry&) = delete;
  AllocatorRegistry& operator=(const AllocatorRegistry&) = delete;
  // Hash table contains device id and its derived pointer
  std::unordered_map<int, Allocator*> allocators_;
  // Lock to ensure modified thread safe 
  std::mutex mu_;
};

// Since functions cannot be eval in global scope.
// A wrapper is needed to set AllocatorRegistry using given device type.
class AllocatorRegister {
 public:
  AllocatorRegister(DeviceType device_type, Allocator* allocator);
};

// A wrapper function to get the derived pointer of 
// the given device type from AllocatorRegistry table. 
Allocator* get_allocator(DeviceType device_type);
}  // namespace starml
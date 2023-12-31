#pragma once
#include "starml/basic/allocator.h"

namespace starml {
// Derived class inherit the public Allocator, implement
// the allocator on CPU.
class CPUAllocator : public Allocator {
 public:
  // Use the default constructor and deconstructor function.
  CPUAllocator() = default;
  ~CPUAllocator() = default;
  // Override the virtual `allocate_raw` from base class.
  void* allocate_raw(size_t num_bytes) const override;
  // Override the virtual `raw_deleter` from base class.
  DeleterFnPtr raw_deleter() const override;
  // Define the corresponding delete function of given device.
  static void delete_fn(void *ptr);
};
// Explicit get the cpu allocator pointer.
Allocator* cpu_allocator();
}  // namespace starml
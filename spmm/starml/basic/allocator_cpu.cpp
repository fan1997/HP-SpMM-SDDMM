#include <stdlib.h>
#include "starml/basic/allocator_cpu.h"
#include "starml/utils/loguru.h"
#include <iostream>

namespace starml {

void* CPUAllocator::allocate_raw(size_t num_bytes) const{
  void* ptr = malloc(num_bytes);
  STARML_CHECK_NOTNULL(ptr)
      << "Fail to allocate " << num_bytes << " bytes on CPU.";
  return ptr;
}

void CPUAllocator::delete_fn(void* ptr) { free(ptr); }

DeleterFnPtr CPUAllocator::raw_deleter() const { return delete_fn; }

Allocator* cpu_allocator() {
  static Allocator* allocator =
      AllocatorRegistry::singleton().allocator(DeviceType::CPU);
  STARML_CHECK_NOTNULL(allocator) << "Allocator for cpu is not set.";
  return allocator;
}

// Register the derived allocator pointer into the AllocatorRegistry.
static CPUAllocator g_cpu_allocator;
static AllocatorRegister g_allocator_register(kCPU, &g_cpu_allocator);

}  // namespace starml
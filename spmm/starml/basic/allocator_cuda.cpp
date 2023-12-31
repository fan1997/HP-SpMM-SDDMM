#include "starml/basic/allocator_cuda.h" 
#include "starml/basic/common_cuda.h"

namespace starml {
void *CUDAAllocator::allocate_raw(size_t num_bytes) const {
  void *d_ptr = 0;
  STARML_CUDA_CHECK(cudaMalloc(&d_ptr, num_bytes));
  return d_ptr;
}

DeleterFnPtr CUDAAllocator::raw_deleter() const { return &delete_fn; }

void CUDAAllocator::delete_fn(void *ptr) { STARML_CUDA_CHECK(cudaFree(ptr)); }

Allocator* cuda_allocator() {
  static Allocator* allocator = 
      AllocatorRegistry::singleton().allocator(DeviceType::CUDA);
  STARML_CHECK_NOTNULL(allocator) << "Allocator for cuda is not set.";
  return allocator;
}

// Register the derived allocator pointer into the AllocatorRegistry.
static CUDAAllocator g_cuda_allocator;
static AllocatorRegister g_allocator_register(kCUDA, &g_cuda_allocator);

}  // namespace starml
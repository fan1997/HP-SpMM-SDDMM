#include "starml/basic/handle_cpu.h"

namespace starml {
CPUHandle::CPUHandle(DeviceIndex index) { index_ = index; }
void CPUHandle::synchronized() const {}
void* CPUHandle::stream() const {}
void CPUHandle::switch_device() const {}

Handle* CPUHandleEntry::create_handle(DeviceIndex index) {
  CPUHandle* handle = new CPUHandle(index);
  return handle;
}

static CPUHandleEntry g_cpu_handle_entry;
static HandleEntryRegister g_cpu_handle_entry_register(kCPU,
                                                       &g_cpu_handle_entry);

}  // namespace starml
#pragma once

#include "madspace/driver/tensor.hpp"

#include <memory>
#include <string>

namespace madspace {
namespace gpu {

// A monitor of the SM activity of CUDA device *device_index* from CUPTI PM
// sampling, or nullptr with the reason in *error* if it is not available. The
// sampling runs in the process madspace_gpu_monitor (gpu_monitor_main.cpp),
// installed next to the madspace CUDA library, because initializing CUPTI
// profiling in a process slows down its kernel launches from several threads
// (3x on an A100), while a separate process costs nothing measurable. The
// helper loads CUPTI at runtime
// (MADSPACE_CUPTI_LIBRARY, libcupti.so.12, libcupti.so, then the library of the
// CUDA toolkit madspace was built with), so that it is not a dependency.
std::unique_ptr<DeviceActivityMonitor>
cupti_activity_monitor(int device_index, std::string* error);

} // namespace gpu
} // namespace madspace

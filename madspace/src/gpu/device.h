#pragma once

#include "gpu_abstraction.h"
#include "madspace/runtime/tensor.h"

#include <format>

namespace madspace {
namespace gpu {

inline void check_error(gpublasStatus_t status) {
    if (status != GPUBLAS_STATUS_SUCCESS) {
        const char* error_str = gpublasGetStatusString(status);
        throw std::runtime_error(std::format("BLAS error: {}", error_str));
    }
}

inline void check_error(gpurandStatus_t status) {
    if (status != GPURAND_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::format("RAND error: error code {}", static_cast<int>(status))
        );
    }
}

inline void check_error(gpuError_t error) {
    if (error != gpuSuccess) {
        const char* error_str = gpuGetErrorString(error);
        throw std::runtime_error(std::format("GPU error: {}", error_str));
    }
}

inline void check_error() { check_error(gpuGetLastError()); }

class GpuDevice : public Device {
public:
    void* allocate(std::size_t size) const override;
    void free(void* ptr) const override;
    void memcpy(void* to, void* from, std::size_t size) const override;

    void tensor_copy(const Tensor& source, Tensor& target) const override;
    void tensor_zero(Tensor& tensor) const override;
    void tensor_add(const Tensor& source, Tensor& target) const override;
    void tensor_cpu(const Tensor& source, Tensor& target) const override;
    DevicePtr device_ptr() const override { return &instance(); }
    DeviceType device_type() const override {
#ifdef __CUDACC__
        return DeviceType::cuda;
#else
        return DeviceType::hip;
#endif
    }

    GpuDevice(const GpuDevice&) = delete;
    GpuDevice& operator=(GpuDevice&) = delete;
    static const GpuDevice& instance() {
        static GpuDevice device;
        return device;
    }

private:
    GpuDevice() = default;
};

class AsyncGpuDevice {
public:
    AsyncGpuDevice(gpuStream_t stream) : _stream(stream) {}

    void* allocate(std::size_t size) const;
    void free(void* ptr) const;
    void memcpy(void* to, void* from, std::size_t size) const;

    void tensor_copy(const Tensor& source, Tensor& target) const;
    void tensor_zero(Tensor& tensor) const;
    void tensor_add(const Tensor& source, Tensor& target) const;
    void tensor_cpu(const Tensor& source, Tensor& target) const;
    DevicePtr device_ptr() const { return &GpuDevice::instance(); }
    void sync_barrier() const {};
    gpuStream_t stream() const { return _stream; }

private:
    gpuStream_t _stream;
};

extern "C" DevicePtr get_device();

} // namespace gpu
} // namespace madspace

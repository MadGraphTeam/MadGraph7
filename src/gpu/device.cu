#include "../kernels/kernels.h"
#include "device.h"
#include "tensor.h"

using namespace madspace;
using namespace madspace::gpu;
using namespace madspace::kernels;

void* GpuDevice::allocate(std::size_t size) const {
    void* ptr;
    check_error(gpuMalloc(&ptr, size));
    return ptr;
}

void GpuDevice::free(void* ptr) const { check_error(gpuFree(ptr)); }

void GpuDevice::memcpy(void* to, void* from, std::size_t size) const {
    check_error(gpuMemcpy(to, from, size, gpuMemcpyDefault));
}

void GpuDevice::tensor_copy(const Tensor& source, Tensor& target) const {
    AsyncGpuDevice(gpuStreamPerThread).tensor_copy(source, target);
    check_error(gpuStreamSynchronize(gpuStreamPerThread));
}

void GpuDevice::tensor_zero(Tensor& tensor) const {
    AsyncGpuDevice(gpuStreamPerThread).tensor_zero(tensor);
    check_error(gpuStreamSynchronize(gpuStreamPerThread));
}

void GpuDevice::tensor_add(const Tensor& source, Tensor& target) const {
    AsyncGpuDevice(gpuStreamPerThread).tensor_add(source, target);
    check_error(gpuStreamSynchronize(gpuStreamPerThread));
}

void GpuDevice::tensor_cpu(const Tensor& source, Tensor& target) const {
    check_error(
        gpuMemcpy(target.data(), source.data(), source.byte_size(), gpuMemcpyDefault)
    );
}

void* AsyncGpuDevice::allocate(std::size_t size) const {
    void* ptr;
    check_error(gpuMallocAsync(&ptr, size, _stream));
    return ptr;
}

void AsyncGpuDevice::free(void* ptr) const { check_error(gpuFreeAsync(ptr, _stream)); }

void AsyncGpuDevice::memcpy(void* to, void* from, std::size_t size) const {
    check_error(gpuMemcpyAsync(to, from, size, gpuMemcpyDefault, _stream));
}

void AsyncGpuDevice::tensor_copy(const Tensor& source, Tensor& target) const {
    if (source.dtype() == DataType::dt_float && target.dtype() == DataType::dt_float) {
        tensor_foreach_dynamic<kernel_copy<GpuTypes>, 1, 1>(
            {&source}, {&target}, target.size(0), *this
        );
    } else if (source.dtype() == DataType::dt_int &&
               target.dtype() == DataType::dt_int) {
        tensor_foreach_dynamic<kernel_copy_int<GpuTypes>, 1, 1>(
            {&source}, {&target}, target.size(0), *this
        );
    } else {
        throw std::runtime_error("invalid dtype in copy");
    }
}

void AsyncGpuDevice::tensor_zero(Tensor& tensor) const {
    if (tensor.dtype() == DataType::dt_float) {
        tensor_foreach_dynamic<kernel_zero<GpuTypes>, 1, 1>(
            {&tensor}, {&tensor}, tensor.size(0), *this
        );
    } else if (tensor.dtype() == DataType::dt_int) {
        tensor_foreach_dynamic<kernel_zero_int<GpuTypes>, 1, 1>(
            {&tensor}, {&tensor}, tensor.size(0), *this
        );
    } else {
        throw std::runtime_error("invalid dtype in zero");
    }
}

void AsyncGpuDevice::tensor_add(const Tensor& source, Tensor& target) const {
    tensor_foreach_dynamic<kernel_add_inplace<GpuTypes>, 1, 1>(
        {&source}, {&target}, target.size(0), *this
    );
}

void AsyncGpuDevice::tensor_cpu(const Tensor& source, Tensor& target) const {
    check_error(gpuMemcpyAsync(
        target.data(), source.data(), source.byte_size(), gpuMemcpyDefault, _stream
    ));
}

extern "C" DevicePtr get_device() { return &GpuDevice::instance(); }

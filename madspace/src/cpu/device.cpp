#include "device.h"
#include "../kernels/kernels.h"
#include "tensor.h"

using namespace madspace;
using namespace madspace::cpu;
using namespace madspace::kernels;

namespace {

template <typename D>
void tensor_copy_impl(const Tensor& source, Tensor& target, const D& device) {
    if (source.dtype() == DataType::dt_float && target.dtype() == DataType::dt_float) {
        tensor_foreach_dynamic_single<
            kernel_copy<CpuTypes>,
            kernel_copy<SimdTypes>,
            1,
            1>({&source}, {&target}, target.size(0), device);
    } else if (source.dtype() == DataType::dt_int &&
               target.dtype() == DataType::dt_int) {
        tensor_foreach_dynamic_single<
            kernel_copy_int<CpuTypes>,
            kernel_copy_int<SimdTypes>,
            1,
            1>({&source}, {&target}, target.size(0), device);
    } else {
        throw std::runtime_error("invalid dtype in copy");
    }
}

template <typename D>
void tensor_zero_impl(Tensor& tensor, const D& device) {
    if (tensor.dtype() == DataType::dt_float) {
        tensor_foreach_dynamic_single<
            kernel_zero<CpuTypes>,
            kernel_zero<SimdTypes>,
            1,
            1>({&tensor}, {&tensor}, tensor.size(0), device);
    } else if (tensor.dtype() == DataType::dt_int) {
        tensor_foreach_dynamic_single<
            kernel_zero_int<CpuTypes>,
            kernel_zero_int<SimdTypes>,
            1,
            1>({&tensor}, {&tensor}, tensor.size(0), device);
    } else {
        throw std::runtime_error("invalid dtype in zero");
    }
}

template <typename D>
void tensor_add_impl(const Tensor& source, Tensor& target, const D& device) {
    tensor_foreach_dynamic_single<
        kernel_add_inplace<CpuTypes>,
        kernel_add_inplace<SimdTypes>,
        1,
        1>({&source}, {&target}, target.size(0), device);
}

} // namespace

void CpuDevice::tensor_copy(const Tensor& source, Tensor& target) const {
    tensor_copy_impl(source, target, *this);
}

void CpuDevice::tensor_zero(Tensor& tensor) const { tensor_zero_impl(tensor, *this); }

void CpuDevice::tensor_add(const Tensor& source, Tensor& target) const {
    tensor_add_impl(source, target, *this);
}

void AsyncCpuDevice::tensor_copy(const Tensor& source, Tensor& target) const {
    tensor_copy_impl(source, target, *this);
}

void AsyncCpuDevice::tensor_zero(Tensor& tensor) const {
    tensor_zero_impl(tensor, *this);
}

void AsyncCpuDevice::tensor_add(const Tensor& source, Tensor& target) const {
    tensor_add_impl(source, target, *this);
}

extern "C" DevicePtr get_device() { return &CpuDevice::instance(); }

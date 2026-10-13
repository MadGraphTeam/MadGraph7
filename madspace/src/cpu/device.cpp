#include "device.hpp"
#include "../kernels/kernels.hpp"
#include "tensor.hpp"

using namespace madspace;
using namespace madspace::cpu;
using namespace madspace::kernels;

namespace {

template <typename D>
void tensor_copy_impl(const Tensor& source, Tensor& target, const D& device) {
    if (source.dtype() != target.dtype()) {
        throw std::runtime_error("invalid dtype in copy");
    }
    switch (source.dtype()) {
    case DataType::dt_float64:
        tensor_foreach_dynamic_single<
            kernel_copy<CpuTypes>,
            kernel_copy<SimdTypes>,
            1,
            1>({&source}, {&target}, target.size(0), device);
        break;
    case DataType::dt_float32:
        tensor_foreach_dynamic_single<
            kernel_copy<CpuTypes32>,
            kernel_copy<SimdTypes32>,
            1,
            1>({&source}, {&target}, target.size(0), device);
        break;
    case DataType::dt_int32:
        tensor_foreach_dynamic_single<
            kernel_copy_int<CpuTypes>,
            kernel_copy_int<SimdTypes>,
            1,
            1>({&source}, {&target}, target.size(0), device);
        break;
    default:
        throw std::runtime_error("invalid dtype in copy");
    }
}

template <typename D>
void tensor_zero_impl(Tensor& tensor, const D& device) {
    switch (tensor.dtype()) {
    case DataType::dt_float64:
        tensor_foreach_dynamic_single<
            kernel_zero<CpuTypes>,
            kernel_zero<SimdTypes>,
            1,
            1>({&tensor}, {&tensor}, tensor.size(0), device);
        break;
    case DataType::dt_float32:
        tensor_foreach_dynamic_single<
            kernel_zero<CpuTypes32>,
            kernel_zero<SimdTypes32>,
            1,
            1>({&tensor}, {&tensor}, tensor.size(0), device);
        break;
    case DataType::dt_int32:
        tensor_foreach_dynamic_single<
            kernel_zero_int<CpuTypes>,
            kernel_zero_int<SimdTypes>,
            1,
            1>({&tensor}, {&tensor}, tensor.size(0), device);
        break;
    default:
        throw std::runtime_error("invalid dtype in zero");
    }
}

template <typename D>
void tensor_add_impl(const Tensor& source, Tensor& target, const D& device) {
    // a broadcast target (e.g. the gradient of a literal or global) accumulates
    // the whole source batch
    std::size_t batch_size = std::max(source.size(0), target.size(0));
    if (source.dtype() != target.dtype()) {
        throw std::runtime_error("invalid dtype in add");
    }
    switch (source.dtype()) {
    case DataType::dt_float64:
        tensor_foreach_dynamic_single<
            kernel_add_inplace<CpuTypes>,
            kernel_add_inplace<SimdTypes>,
            1,
            1>({&source}, {&target}, batch_size, device);
        break;
    case DataType::dt_float32:
        tensor_foreach_dynamic_single<
            kernel_add_inplace<CpuTypes32>,
            kernel_add_inplace<SimdTypes32>,
            1,
            1>({&source}, {&target}, batch_size, device);
        break;
    default:
        throw std::runtime_error("invalid dtype in add");
    }
}

} // namespace

void CpuDevice::tensor_copy(const Tensor& source, Tensor& target) const {
    tensor_copy_impl(source, target, *this);
}

void CpuDevice::tensor_zero(Tensor& tensor) const { tensor_zero_impl(tensor, *this); }

void CpuDevice::tensor_add(const Tensor& source, Tensor& target) const {
    tensor_add_impl(source, target, *this);
}

void CpuDevice::adam_step(
    const Tensor& gradient,
    Tensor& parameter,
    Tensor& exp_avg,
    Tensor& exp_avg_sq,
    double step_size,
    double beta1,
    double beta2,
    double eps,
    double bias_corr2_sqrt,
    double weight_decay
) const {
    tensor_foreach_dynamic<
        kernel_adam_step<CpuTypes>,
        kernel_adam_step<SimdTypes>,
        1,
        3>(
        {&gradient},
        {&parameter, &exp_avg, &exp_avg_sq},
        gradient.size(0),
        *this,
        step_size,
        beta1,
        beta2,
        eps,
        bias_corr2_sqrt,
        weight_decay
    );
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

extern "C" int device_count() { return 1; }
extern "C" DevicePtr get_device(int index) { return &CpuDevice::instance(); }

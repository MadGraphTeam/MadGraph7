#include "madspace/runtime/tensor.h"

using namespace madspace;

Tensor Tensor::select(std::size_t axis, std::size_t index) const {
    check_impl();
    auto new_dim = impl->shape.size() - 1;
    Sizes new_shape(new_dim), new_stride(new_dim);
    std::copy(impl->shape.begin(), impl->shape.begin() + axis, new_shape.begin());
    std::copy(
        impl->shape.begin() + axis + 1, impl->shape.end(), new_shape.begin() + axis
    );
    std::copy(impl->stride.begin(), impl->stride.begin() + axis, new_stride.begin());
    std::copy(
        impl->stride.begin() + axis + 1, impl->stride.end(), new_stride.begin() + axis
    );
    return Tensor(new Tensor::TensorImpl{
        impl->dtype,
        new_shape,
        impl->device,
        impl->data,
        false,
        std::nullopt,
        impl,
        1,
        new_stride,
        impl->offset + index * impl->stride[axis],
        std::min(impl->contiguous_dims, axis)
    });
}

Tensor Tensor::slice(std::size_t axis, std::size_t start, std::size_t stop) const {
    check_impl();
    auto new_shape = impl->shape;
    new_shape[axis] = stop - start;
    return Tensor(new Tensor::TensorImpl{
        impl->dtype,
        new_shape,
        impl->device,
        impl->data,
        false,
        std::nullopt,
        impl,
        1,
        impl->stride,
        impl->offset + start * impl->stride[axis],
        std::min(impl->contiguous_dims, axis)
    });
}

std::vector<Tensor> Tensor::split(std::size_t axis, const SizeVec& sizes) const {
    check_impl();
    std::vector<Tensor> tensors;
    auto split_count = sizes.size();
    tensors.reserve(split_count);
    std::size_t offset = 0;
    for (std::size_t size : sizes) {
        auto next_offset = offset + size;
        tensors.push_back(slice(axis, offset, next_offset));
        offset = next_offset;
    }
    return tensors;
}

std::vector<Tensor> Tensor::unstack(std::size_t axis) const {
    check_impl();
    std::vector<Tensor> tensors;
    auto axis_size = size(axis);
    tensors.reserve(axis_size);
    for (std::size_t i = 0; i < axis_size; ++i) {
        tensors.push_back(select(axis, i));
    }
    return tensors;
}

Tensor Tensor::unsqueeze(std::size_t axis) const {
    check_impl();
    // TODO: check if correct for non-contiguous tensor
    auto new_dim = impl->shape.size() + 1;
    Sizes new_shape(new_dim), new_stride(new_dim);
    std::copy(impl->shape.begin(), impl->shape.begin() + axis, new_shape.begin());
    new_shape[axis] = 1;
    std::copy(
        impl->shape.begin() + axis, impl->shape.end(), new_shape.begin() + axis + 1
    );
    std::copy(impl->stride.begin(), impl->stride.begin() + axis, new_stride.begin());
    new_stride[axis] = axis == 0 ? 1 : impl->stride[axis - 1];
    std::copy(
        impl->stride.begin() + axis, impl->stride.end(), new_stride.begin() + axis + 1
    );
    return Tensor(new Tensor::TensorImpl{
        impl->dtype,
        new_shape,
        impl->device,
        impl->data,
        false,
        std::nullopt,
        impl,
        1,
        new_stride,
        impl->offset,
        impl->contiguous_dims + (axis <= impl->contiguous_dims)
    });
}

Tensor Tensor::expand(const Sizes& shape) const {
    check_impl();
    return Tensor(new Tensor::TensorImpl{
        impl->dtype,
        shape,
        impl->device,
        impl->data,
        false,
        std::nullopt,
        impl,
        1,
        Sizes(shape.size(), 0),
        impl->offset,
        0
    });
}

Tensor Tensor::factor_dim(std::size_t axis, std::size_t factor) {
    check_impl();
    auto new_dim = impl->shape.size() + 1;
    Sizes new_shape(new_dim), new_stride(new_dim);

    std::copy(impl->shape.begin(), impl->shape.begin() + axis, new_shape.begin());
    new_shape[axis] = factor;
    std::copy(
        impl->shape.begin() + axis, impl->shape.end(), new_shape.begin() + axis + 1
    );
    new_shape[axis + 1] /= factor;

    std::copy(
        impl->stride.begin(), impl->stride.begin() + axis + 1, new_stride.begin()
    );
    new_stride[axis + 1] = new_stride[axis] * factor;
    std::copy(
        impl->stride.begin() + axis + 1,
        impl->stride.end(),
        new_stride.begin() + axis + 2
    );

    return Tensor(new Tensor::TensorImpl{
        impl->dtype,
        new_shape,
        impl->device,
        impl->data,
        false,
        std::nullopt,
        impl,
        1,
        new_stride,
        impl->offset,
        impl->contiguous_dims + (axis <= impl->contiguous_dims)
    });
}

std::size_t Tensor::init_stride() {
    std::size_t stride_prod = 1;
    bool first = true;
    for (auto size : impl->shape) {
        if (first && size == 1) {
            impl->stride.push_back(0);
        } else {
            impl->stride.push_back(stride_prod);
        }
        stride_prod *= size;
        first = false;
    }
    impl->contiguous_dims = impl->shape.size();
    return stride_prod * dtype_size();
}

#include "runtime.h"

#include <algorithm>
#include <array>
#include <format>
#include <functional>
#include <random>
#include <tuple>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>

#include "../kernels/kernels.h"
#include "../kernels/operations.h"
#include "device.h"
#include "madspace/util.h"
#include "tensor.h"

using namespace madspace;
using namespace madspace::gpu;
using namespace madspace::kernels;

namespace {

void op_matrix_element(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device
) {
    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    std::size_t me_index = locals[instruction.input_indices[0]].index_value();
    std::size_t input_count = locals[instruction.input_indices[1]].index_value();
    std::size_t output_count = locals[instruction.input_indices[2]].index_value();
    TensorVec contiguous_inputs(input_count);
    std::vector<UmamiInputKey> input_keys(input_count + 1);
    std::vector<UmamiOutputKey> output_keys(output_count);
    std::vector<void*> input_ptrs(input_count + 1), output_ptrs(output_count);
    for (std::size_t i = 0; i < input_count; ++i) {
        input_keys[i] = static_cast<UmamiInputKey>(
            locals[instruction.input_indices[3 + 2 * i]].index_value()
        );
        contiguous_inputs[i] =
            locals[instruction.input_indices[3 + 2 * i + 1]].contiguous(
                batch_size, device
            );
        input_ptrs[i] = contiguous_inputs[i].data();
    }
    input_keys[input_count] = UMAMI_IN_GPU_STREAM;
    input_ptrs[input_count] = device.stream();
    std::size_t output_offset = 3 + 2 * input_count;
    for (std::size_t i = 0; i < output_count; ++i) {
        output_keys[i] = static_cast<UmamiOutputKey>(
            locals[instruction.input_indices[output_offset + 2 * i]].index_value()
        );
        auto& output = locals[instruction.output_indices[i]];
        auto& output_shape = instruction.output_shapes[i];
        Sizes shape(output_shape.size() + 1);
        shape[0] = batch_size;
        std::copy(output_shape.begin(), output_shape.end(), shape.begin() + 1);
        output = Tensor(instruction.output_dtypes[i], shape, device);
        output_ptrs[i] = output.data();
        if (me_index == 0xBADCAFE) {
            // flat dummy matrix element for testing purposes
            // implemented at LUMI hackathon where the coffee was indeed terrible
            switch (output_keys[i]) {
            case UMAMI_OUT_MATRIX_ELEMENT:
                thrust::fill_n(
                    thrust_par.on(device.stream()),
                    thrust::device_pointer_cast(static_cast<double*>(output_ptrs[i])),
                    batch_size,
                    1.0
                );
                break;
            case UMAMI_OUT_DIAGRAM_AMP2:
                thrust::fill_n(
                    thrust_par.on(device.stream()),
                    thrust::device_pointer_cast(static_cast<double*>(output_ptrs[i])),
                    batch_size * shape[1],
                    1. / shape[1]
                );
                break;
            default:
                output.zero(device);
                break;
            }
        }
    }
    if (me_index == 0xBADCAFE || batch_size == 0) {
        return;
    }
    auto& matrix_element = instruction.runtime.context().matrix_element(me_index);
    if (matrix_element.device() != get_device()) {
        throw std::runtime_error("Incompatible device");
    }
    device.sync_barrier();

    matrix_element.call(
        matrix_element.process_instance(ThreadPool::thread_index()),
        batch_size,
        batch_size,
        0,
        input_count,
        input_keys.data(),
        input_ptrs.data(),
        output_count,
        output_keys.data(),
        output_ptrs.data()
    );
    for (auto& input : contiguous_inputs) {
        input.reset(device);
    }
}

void op_matmul(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device
) {
    auto input = locals[instruction.input_indices[0]].contiguous(device);
    auto weight = locals[instruction.input_indices[1]].contiguous(device);
    auto bias = locals[instruction.input_indices[2]].contiguous(device);
    auto& output = locals[instruction.output_indices[0]];
    std::size_t batch_size = input.size(0);
    std::size_t dims_in = input.size(1);
    std::size_t dims_out = weight.size(1);
    output = Tensor(DataType::dt_float, {batch_size, dims_out}, device);
    output.copy_from(bias, device);
    if (batch_size == 0) {
        return;
    }

    gpublasHandle_t handle = instruction.runtime.gpublas_handle();
    check_error(gpublasSetStream(handle, device.stream()));
    double alpha = 1., beta = 1.;
    check_error(gpublasDgemm(
        handle,
        GPUBLAS_OP_N,
        GPUBLAS_OP_T,
        batch_size,
        dims_out,
        dims_in,
        &alpha,
        static_cast<double*>(input.data()),
        batch_size,
        static_cast<double*>(weight.data()),
        dims_out,
        &beta,
        static_cast<double*>(output.data()),
        batch_size
    ));
    input.reset(device);
    weight.reset(device);
    bias.reset(device);
}

void backward_op_matmul(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    TensorVec& local_grads,
    const AsyncGpuDevice& device
) {
    auto input = locals[instruction.input_indices[0]].contiguous(device);
    auto weight = locals[instruction.input_indices[1]].contiguous(device);
    auto output_grad = local_grads[instruction.output_indices[0]].contiguous(device);
    auto& input_grad = local_grads[instruction.input_indices[0]];
    auto& weight_grad = local_grads[instruction.input_indices[1]];
    auto& bias_grad = local_grads[instruction.input_indices[2]];
    std::size_t batch_size = input.size(0);
    std::size_t dims_in = input.size(1);
    std::size_t dims_out = weight.size(1);

    if (!input_grad) {
        input_grad = Tensor(DataType::dt_float, input.shape(), device);
        input_grad.zero();
    }
    if (!weight_grad) {
        weight_grad = Tensor(DataType::dt_float, weight.shape(), device);
        weight_grad.zero();
    }
    if (!bias_grad) {
        bias_grad = Tensor(DataType::dt_float, {1, dims_out}, device);
        bias_grad.zero();
    }
    if (batch_size == 0) {
        return;
    }

    double alpha = 1., beta = 1.;
    gpublasHandle_t handle = instruction.runtime.gpublas_handle();
    gpuStream_t stream = device.stream();
    check_error(gpublasSetStream(handle, stream));

    // compute input_grad += output_grad * weight
    check_error(gpublasDgemm(
        handle,
        GPUBLAS_OP_N,
        GPUBLAS_OP_N,
        batch_size,
        dims_in,
        dims_out,
        &alpha,
        static_cast<double*>(output_grad.data()),
        batch_size,
        static_cast<double*>(weight.data()),
        dims_out,
        &beta,
        static_cast<double*>(input_grad.data()),
        batch_size
    ));

    // compute weight_grad += output_grad.T * input
    check_error(gpublasDgemm(
        handle,
        GPUBLAS_OP_T,
        GPUBLAS_OP_N,
        dims_out,
        dims_in,
        batch_size,
        &alpha,
        static_cast<double*>(output_grad.data()),
        batch_size,
        static_cast<double*>(input.data()),
        batch_size,
        &beta,
        static_cast<double*>(weight_grad.data()),
        dims_out
    ));

    // compute bias_grad += sum_i output_grad_ij
    double* ones;
    check_error(gpuMallocAsync(&ones, batch_size * sizeof(double), stream));
    thrust::fill_n(
        thrust_par.on(stream), thrust::device_pointer_cast(ones), batch_size, 1.0
    );
    check_error(gpublasDgemv(
        handle,
        GPUBLAS_OP_T,
        batch_size,
        dims_out,
        &alpha,
        static_cast<double*>(output_grad.data()),
        batch_size,
        static_cast<double*>(ones),
        1,
        &beta,
        static_cast<double*>(bias_grad.data()),
        1
    ));
    check_error(gpuFreeAsync(ones, stream));
    input.reset(device);
    weight.reset(device);
    output_grad.reset(device);
}

struct NotMinusOne {
    __device__ bool operator()(me_int_t val) { return val != -1; }
};

__global__ void kernel_nonzero(
    std::size_t batch_size,
    GpuTensorView<double, 1, true> input,
    GpuTensorView<me_int_t, 1, true> output
) {
    me_int_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < batch_size) {
        output[i] = input[i] == 0. ? -1 : i;
    }
}

void op_nonzero(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device
) {
    auto& input = locals[instruction.input_indices[0]];
    auto batch_size = input.size(0);
    auto& output = locals[instruction.output_indices[0]];
    Tensor indices_tmp(DataType::dt_int, {batch_size}, device);
    Tensor output_tmp(DataType::dt_int, {batch_size}, device);
    launch_kernel(
        kernel_nonzero,
        batch_size,
        device.stream(),
        batch_size,
        input.view<double, 1>(),
        indices_tmp.view<me_int_t, 1>()
    );

    auto indices_ptr =
        thrust::device_pointer_cast(static_cast<me_int_t*>(indices_tmp.data()));
    auto output_ptr =
        thrust::device_pointer_cast(static_cast<me_int_t*>(output_tmp.data()));
    check_error(gpuStreamSynchronize(device.stream()));
    auto count = thrust::copy_if(
                     indices_ptr, indices_ptr + batch_size, output_ptr, NotMinusOne()
                 ) -
        output_ptr;
    output = output_tmp.slice(0, 0, count);
    indices_tmp.reset(device);
    output_tmp.reset(device);
}

template <int dim>
__global__ void batch_gather_kernel(
    std::size_t batch_size,
    GpuTensorView<me_int_t, 1, true> indices,
    GpuTensorView<double, dim, true> values,
    GpuTensorView<double, dim, true> selection
) {
    std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < batch_size) {
        recursive_for<kernel_copy<GpuTypes>, dim - 1>(values[indices[i]], selection[i]);
    }
}

template <int dim>
__global__ void batch_gather_kernel_int(
    std::size_t batch_size,
    GpuTensorView<me_int_t, 1, true> indices,
    GpuTensorView<me_int_t, dim, true> values,
    GpuTensorView<me_int_t, dim, true> selection
) {
    std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < batch_size) {
        recursive_for<kernel_copy_int<GpuTypes>, dim - 1>(
            values[indices[i]], selection[i]
        );
    }
}

template <int dim>
void batch_gather_impl(
    Tensor& indices, Tensor& values, Tensor& selection, const AsyncGpuDevice& device
) {
    auto batch_size = indices.size(0);
    Sizes out_shape = values.shape();
    out_shape[0] = batch_size;

    if (values.dtype() == DataType::dt_float) {
        selection = Tensor(DataType::dt_float, out_shape, device);
        launch_kernel(
            batch_gather_kernel<dim>,
            batch_size,
            device.stream(),
            batch_size,
            indices.view<me_int_t, 1>(),
            values.view<double, dim>(),
            selection.view<double, dim>()
        );
    } else if (values.dtype() == DataType::dt_int) {
        selection = Tensor(DataType::dt_int, out_shape, device);
        launch_kernel(
            batch_gather_kernel_int<dim>,
            batch_size,
            device.stream(),
            batch_size,
            indices.view<me_int_t, 1>(),
            values.view<me_int_t, dim>(),
            selection.view<me_int_t, dim>()
        );
    } else {
        throw std::runtime_error("invalid dtype in batch_gather");
    }
}

void op_batch_gather(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device
) {
    auto& indices = locals[instruction.input_indices[0]];
    auto& values = locals[instruction.input_indices[1]];
    auto& selection = locals[instruction.output_indices[0]];
    switch (values.shape().size()) {
    case 1:
        batch_gather_impl<1>(indices, values, selection, device);
        break;
    case 2:
        batch_gather_impl<2>(indices, values, selection, device);
        break;
    case 3:
        batch_gather_impl<3>(indices, values, selection, device);
        break;
    case 4:
        batch_gather_impl<4>(indices, values, selection, device);
        break;
    default:
        throw std::runtime_error("The number of dimensions must be between 1 and 4");
    }
}

template <int dim>
__global__ void batch_scatter_kernel(
    std::size_t batch_size,
    GpuTensorView<me_int_t, 1, true> indices,
    GpuTensorView<double, dim, true> source,
    GpuTensorView<double, dim, true> output
) {
    std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < batch_size) {
        recursive_for<kernel_copy<GpuTypes>, dim - 1>(source[i], output[indices[i]]);
    }
}

template <int dim>
__global__ void batch_scatter_kernel_int(
    std::size_t batch_size,
    GpuTensorView<me_int_t, 1, true> indices,
    GpuTensorView<me_int_t, dim, true> source,
    GpuTensorView<me_int_t, dim, true> output
) {
    std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < batch_size) {
        recursive_for<kernel_copy_int<GpuTypes>, dim - 1>(
            source[i], output[indices[i]]
        );
    }
}

template <int dim>
void batch_scatter_impl(
    Tensor& indices, Tensor& source, Tensor& output, const AsyncGpuDevice& device
) {
    if (source.dtype() == DataType::dt_float) {
        auto batch_size = indices.size(0);
        launch_kernel(
            batch_scatter_kernel<dim>,
            batch_size,
            device.stream(),
            batch_size,
            indices.view<me_int_t, 1>(),
            source.view<double, dim>(),
            output.view<double, dim>()
        );
    } else if (source.dtype() == DataType::dt_int) {
        auto batch_size = indices.size(0);
        launch_kernel(
            batch_scatter_kernel_int<dim>,
            batch_size,
            device.stream(),
            batch_size,
            indices.view<me_int_t, 1>(),
            source.view<me_int_t, dim>(),
            output.view<me_int_t, dim>()
        );
    } else {
        throw std::runtime_error("invalid dtype in batch_scatter");
    }
}

void op_batch_scatter(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device
) {
    auto& indices = locals[instruction.input_indices[0]];
    auto& target = locals[instruction.input_indices[1]];
    auto& source = locals[instruction.input_indices[2]];

    auto& output = locals[instruction.output_indices[0]];
    output = target.copy(device);
    switch (target.shape().size()) {
    case 1:
        batch_scatter_impl<1>(indices, source, output, device);
        break;
    case 2:
        batch_scatter_impl<2>(indices, source, output, device);
        break;
    case 3:
        batch_scatter_impl<3>(indices, source, output, device);
        break;
    case 4:
        batch_scatter_impl<4>(indices, source, output, device);
        break;
    default:
        throw std::runtime_error("The number of dimensions must be between 1 and 4");
    }
}

void op_offset_indices(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device
) {
    auto& sizes_offset = locals[instruction.input_indices[0]].batch_sizes();
    auto& sizes_out = locals[instruction.input_indices[1]].batch_sizes();
    std::size_t total_size = std::accumulate(sizes_out.begin(), sizes_out.end(), 0);
    auto& output = locals[instruction.output_indices[0]];
    output = Tensor(DataType::dt_int, {total_size}, device);
    std::size_t sum_offset = 0, sum_out = 0;
    for (auto [size_offset, size_out] : zip(sizes_offset, sizes_out)) {
        thrust::fill_n(
            thrust_par.on(device.stream()),
            thrust::device_pointer_cast(
                static_cast<me_int_t*>(output.data()) + sum_out
            ),
            size_out,
            sum_offset
        );
        sum_offset += size_offset;
        sum_out += size_out;
    }
}

void op_random(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device
) {
    auto batch_size = locals[instruction.input_indices[0]].batch_sizes()[0];
    auto& output = locals[instruction.output_indices[0]];
    auto dim = instruction.output_shapes[0][0];
    output = Tensor(DataType::dt_float, {batch_size, dim}, device);
    gpurandGenerator_t generator = instruction.runtime.gpurand_generator();
    check_error(gpurandSetStream(generator, device.stream()));
    check_error(gpurandGenerateUniformDouble(
        generator, static_cast<double*>(output.data()), batch_size * dim
    ));
}

__global__ void kernel_unweight(
    std::size_t batch_size,
    GpuTensorView<double, 1, true> rand_in,
    GpuTensorView<double, 1, true> weights_in,
    GpuTensorView<double, 1, true> max_weights_in,
    GpuTensorView<double, 1, true> weights_out,
    GpuTensorView<me_int_t, 1, true> indices_out
) {
    me_int_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= batch_size) {
        return;
    }

    auto rand = rand_in[i], weight = weights_in[i], max_weight = max_weights_in[i];
    bool accepted = max_weight * rand < weight;
    auto weight_clipped = weight < max_weight ? max_weight : weight;
    weights_out[i] = accepted ? weight_clipped : 0.;
    indices_out[i] = accepted ? i : -1;
}

void op_unweight(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device
) {
    auto& weights = locals[instruction.input_indices[0]];
    auto& max_weight = locals[instruction.input_indices[1]];
    auto& indices = locals[instruction.output_indices[0]];
    auto& uw_weights = locals[instruction.output_indices[1]];
    auto batch_size = weights.size(0);
    gpuStream_t stream = device.stream();

    Tensor rand(DataType::dt_float, {batch_size}, device);
    gpurandGenerator_t generator = instruction.runtime.gpurand_generator();
    check_error(gpurandSetStream(generator, stream));
    check_error(gpurandGenerateUniformDouble(
        generator, static_cast<double*>(rand.data()), batch_size
    ));

    Tensor indices_tmp(DataType::dt_int, {batch_size}, device);
    Tensor uw_weights_tmp(DataType::dt_float, {batch_size}, device);
    launch_kernel(
        kernel_unweight,
        batch_size,
        stream,
        batch_size,
        rand.view<double, 1>(),
        weights.view<double, 1>(),
        max_weight.view<double, 1>(),
        uw_weights_tmp.view<double, 1>(),
        indices_tmp.view<me_int_t, 1>()
    );

    Tensor indices_compacted(DataType::dt_int, {batch_size}, device);
    check_error(gpuStreamSynchronize(stream));
    auto ptr_all =
        thrust::device_pointer_cast(static_cast<me_int_t*>(indices_tmp.data()));
    auto ptr_compacted =
        thrust::device_pointer_cast(static_cast<me_int_t*>(indices_compacted.data()));
    auto ptr_compacted_end =
        thrust::copy_if(ptr_all, ptr_all + batch_size, ptr_compacted, NotMinusOne());

    std::size_t count = ptr_compacted_end - ptr_compacted;
    indices = indices_compacted.slice(0, 0, count);
    uw_weights = Tensor(DataType::dt_float, {count}, device);
    auto ptr_all_weights =
        thrust::device_pointer_cast(static_cast<double*>(uw_weights_tmp.data()));
    auto ptr_uw_weights =
        thrust::device_pointer_cast(static_cast<double*>(uw_weights.data()));
    thrust::gather(
        thrust_par.on(stream),
        ptr_compacted,
        ptr_compacted_end,
        ptr_all_weights,
        ptr_uw_weights
    );
    rand.reset(device);
    indices_tmp.reset(device);
    uw_weights_tmp.reset(device);
    indices_compacted.reset(device);
}

struct Decrement {
    __device__ void operator()(me_int_t& val) { --val; }
};

void histogram_common(
    const AsyncGpuDevice& device,
    std::size_t padded_size,
    std::size_t n_dims,
    std::size_t n_bins,
    Tensor& indices_tmp,
    Tensor& weights_tmp,
    Tensor& counts,
    Tensor& values
) {
    auto policy = thrust_par.on(device.stream());
    Tensor reduce_tmp(DataType::dt_float, {n_dims * n_bins}, device);
    auto indices_ptr =
        thrust::device_pointer_cast(static_cast<me_int_t*>(indices_tmp.data()));
    auto weights_ptr =
        thrust::device_pointer_cast(static_cast<double*>(weights_tmp.data()));
    auto counts_ptr =
        thrust::device_pointer_cast(static_cast<me_int_t*>(counts.data()));
    auto values_ptr = thrust::device_pointer_cast(static_cast<double*>(values.data()));
    auto reduce_tmp_ptr =
        thrust::device_pointer_cast(static_cast<double*>(reduce_tmp.data()));

    std::size_t flat_size = padded_size * n_dims;
    thrust::sort_by_key(policy, indices_ptr, indices_ptr + flat_size, weights_ptr);
    thrust::reduce_by_key(
        policy,
        indices_ptr,
        indices_ptr + flat_size,
        thrust::constant_iterator<me_int_t>(1),
        reduce_tmp_ptr,
        counts_ptr
    );
    thrust::for_each_n(policy, counts_ptr, n_dims * n_bins, Decrement{});
    thrust::reduce_by_key(
        policy,
        indices_ptr,
        indices_ptr + flat_size,
        weights_ptr,
        reduce_tmp_ptr,
        values_ptr
    );
    reduce_tmp.reset(device);
}

__global__ void kernel_prepare_vegas_hist(
    std::size_t batch_size,
    std::size_t n_bins,
    GpuTensorView<double, 2, true> input,
    GpuTensorView<double, 1, true> weights_in,
    GpuTensorView<me_int_t, 2, true> indices,
    GpuTensorView<double, 2, true> weights_out
) {
    me_int_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= batch_size + n_bins) {
        return;
    }
    std::size_t n_dims = input.size(1);
    double bin_count_f = n_bins;
    double w2 = i < batch_size ? weights_in[i] * weights_in[i] : 0.;
    for (std::size_t j = 0; j < n_dims; ++j) {
        indices[i][j] = j +
            n_dims *
                (i < batch_size ? static_cast<me_int_t>(input[i][j] * bin_count_f)
                                : i - batch_size);
        weights_out[i][j] = w2;
    }
}

void op_vegas_histogram(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device
) {
    auto& input = locals[instruction.input_indices[0]];
    auto& weights = locals[instruction.input_indices[1]];
    auto& values = locals[instruction.output_indices[0]];
    auto& counts = locals[instruction.output_indices[1]];

    auto out_shape = instruction.output_shapes[0];
    Sizes shape(out_shape.size() + 1);
    shape[0] = 1;
    std::copy(out_shape.begin(), out_shape.end(), shape.begin() + 1);
    values = Tensor(DataType::dt_float, shape, device);
    counts = Tensor(DataType::dt_int, shape, device);

    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    std::size_t n_dims = input.size(1);
    std::size_t n_bins = values.size(2);

    std::size_t padded_size = batch_size + n_bins;
    Tensor indices_tmp(DataType::dt_int, {padded_size, n_dims}, device);
    Tensor weights_tmp(DataType::dt_float, {padded_size, n_dims}, device);
    launch_kernel(
        kernel_prepare_vegas_hist,
        padded_size,
        device.stream(),
        batch_size,
        n_bins,
        input.view<double, 2>(),
        weights.view<double, 1>(),
        indices_tmp.view<me_int_t, 2>(),
        weights_tmp.view<double, 2>()
    );
    histogram_common(
        device, padded_size, n_dims, n_bins, indices_tmp, weights_tmp, counts, values
    );
    indices_tmp.reset(device);
    weights_tmp.reset(device);
}

__global__ void kernel_prepare_discrete_hist(
    std::size_t batch_size,
    std::size_t n_opts,
    GpuTensorView<me_int_t, 1, true> input,
    GpuTensorView<double, 1, true> weights_in,
    GpuTensorView<me_int_t, 1, true> indices,
    GpuTensorView<double, 1, true> weights_out
) {
    me_int_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= batch_size + n_opts) {
        return;
    }
    indices[i] = i < batch_size ? input[i] : i - batch_size;
    weights_out[i] = i < batch_size ? weights_in[i] : 0.;
}

void op_discrete_histogram(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device
) {
    auto& input = locals[instruction.input_indices[0]];
    auto& weights = locals[instruction.input_indices[1]];
    auto& values = locals[instruction.output_indices[0]];
    auto& counts = locals[instruction.output_indices[1]];

    auto out_shape = instruction.output_shapes[0];
    Sizes shape(out_shape.size() + 1);
    shape[0] = 1;
    std::copy(out_shape.begin(), out_shape.end(), shape.begin() + 1);
    values = Tensor(DataType::dt_float, shape, device);
    counts = Tensor(DataType::dt_int, shape, device);

    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    std::size_t n_opts = values.size(1);

    std::size_t padded_size = batch_size + n_opts;
    Tensor indices_tmp(DataType::dt_int, {padded_size}, device);
    Tensor weights_tmp(DataType::dt_float, {padded_size}, device);
    launch_kernel(
        kernel_prepare_discrete_hist,
        padded_size,
        device.stream(),
        batch_size,
        n_opts,
        input.view<me_int_t, 1>(),
        weights.view<double, 1>(),
        indices_tmp.view<me_int_t, 1>(),
        weights_tmp.view<double, 1>()
    );
    histogram_common(
        device, padded_size, 1, n_opts, indices_tmp, weights_tmp, counts, values
    );
    indices_tmp.reset(device);
    weights_tmp.reset(device);
}

__global__ void kernel_prepare_hist(
    std::size_t batch_size,
    std::size_t n_bins,
    GpuTensorView<double, 1, true> input,
    GpuTensorView<double, 1, true> min,
    GpuTensorView<double, 1, true> max,
    GpuTensorView<double, 1, true> weights_in,
    GpuTensorView<me_int_t, 1, true> indices,
    GpuTensorView<double, 1, true> weights_out,
    GpuTensorView<double, 1, true> square_weights_out
) {
    me_int_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= batch_size + n_bins + 2) {
        return;
    }
    double bin_count_f = n_bins;
    double w;
    me_int_t index;
    if (i < batch_size) {
        w = weights_in[i];
        me_int_t index_rounded = (input[i] - min[i]) / (max[i] - min[i]) * bin_count_f;
        if (index_rounded < 0) {
            index = 0;
        } else if (index_rounded >= n_bins) {
            index = n_bins + 1;
        } else {
            index = index_rounded + 1;
        }
    } else {
        w = 0.;
        index = i - batch_size;
    }
    indices[i] = index;
    weights_out[i] = w;
    square_weights_out[i] = w * w;
}

void op_histogram(
    const GpuRuntime::Instruction& instruction,
    TensorVec& locals,
    const AsyncGpuDevice& device
) {
    Tensor input = locals[instruction.input_indices[0]].contiguous(device);
    auto& weights = locals[instruction.input_indices[1]];
    auto& hist_min = locals[instruction.input_indices[2]];
    auto& hist_max = locals[instruction.input_indices[3]];
    auto& values = locals[instruction.output_indices[0]];
    auto& square_values = locals[instruction.output_indices[1]];

    auto out_shape = instruction.output_shapes[0];
    Sizes shape(out_shape.size() + 1);
    shape[0] = 1;
    std::copy(out_shape.begin(), out_shape.end(), shape.begin() + 1);
    values = Tensor(DataType::dt_float, shape, device);
    square_values = Tensor(DataType::dt_float, shape, device);

    std::size_t batch_size = locals[instruction.batch_size_index].size(0);
    std::size_t n_bins = values.size(1) - 2;
    std::size_t padded_size = batch_size + n_bins + 2;
    Tensor indices_tmp(DataType::dt_int, {padded_size}, device);
    Tensor weights_tmp(DataType::dt_float, {padded_size}, device);
    Tensor square_weights_tmp(DataType::dt_float, {padded_size}, device);
    Tensor reduce_tmp(DataType::dt_float, {n_bins}, device);

    launch_kernel(
        kernel_prepare_hist,
        padded_size,
        device.stream(),
        batch_size,
        n_bins,
        input.view<double, 1>(),
        hist_min.view<double, 1>(),
        hist_max.view<double, 1>(),
        weights.view<double, 1>(),
        indices_tmp.view<me_int_t, 1>(),
        weights_tmp.view<double, 1>(),
        square_weights_tmp.view<double, 1>()
    );

    auto policy = thrust_par.on(device.stream());
    auto indices_ptr =
        thrust::device_pointer_cast(static_cast<me_int_t*>(indices_tmp.data()));
    auto reduce_tmp_ptr =
        thrust::device_pointer_cast(static_cast<double*>(reduce_tmp.data()));

    auto weights_ptr =
        thrust::device_pointer_cast(static_cast<double*>(weights_tmp.data()));
    auto values_ptr = thrust::device_pointer_cast(static_cast<double*>(values.data()));
    thrust::sort_by_key(policy, indices_ptr, indices_ptr + padded_size, weights_ptr);
    thrust::reduce_by_key(
        policy,
        indices_ptr,
        indices_ptr + padded_size,
        weights_ptr,
        reduce_tmp_ptr,
        values_ptr
    );

    auto square_weights_ptr =
        thrust::device_pointer_cast(static_cast<double*>(square_weights_tmp.data()));
    auto square_values_ptr =
        thrust::device_pointer_cast(static_cast<double*>(square_values.data()));
    thrust::sort_by_key(
        policy, indices_ptr, indices_ptr + padded_size, square_weights_ptr
    );
    thrust::reduce_by_key(
        policy,
        indices_ptr,
        indices_ptr + padded_size,
        square_weights_ptr,
        reduce_tmp_ptr,
        square_values_ptr
    );

    reduce_tmp.reset(device);
    indices_tmp.reset(device);
    weights_tmp.reset(device);
    square_weights_tmp.reset(device);
}

} // namespace

GpuRuntime::GpuRuntime(const Function& function, ContextPtr context) :
    _context(context), input_count(function.inputs().size()) {
    check_error(
        gpurandCreateGenerator(&_gpurand_generator, GPURAND_RNG_PSEUDO_DEFAULT)
    );
    std::random_device rand_dev;
    check_error(gpurandSetPseudoRandomGeneratorSeed(_gpurand_generator, rand_dev()));
    check_error(gpublasCreate(&_gpublas_handle));

    locals_init.resize(function.locals().size());
    requires_grad_init.resize(function.locals().size());
    LastUseOfLocals last_use(function);
    InstructionDependencies dependencies(function);

    std::size_t instr_index = 0;

    gpuStream_t new_stream;
    check_error(gpuStreamCreate(&new_stream));
    streams.push_back(new_stream);

    // std::vector<int> forward_streams;
    // std::vector<int> backward_streams;
    std::vector<int> local_sources(function.locals().size(), -1);
    for (auto& instr : function.instructions()) {
        SizeVec input_indices;
        std::size_t batch_size_index = instr.inputs.at(0).local_index;
        // int forward_stream_index = -1, backward_stream_index = -1;
        for (auto& in : instr.inputs) {
            input_indices.push_back(in.local_index);
            if (in.type.batch_size != BatchSize::one) {
                batch_size_index = in.local_index;
            }
            /*int local_source = local_sources.at(in.local_index);
            if (local_source != -1) {
                if (forward_streams.at(local_source)) {

                }

            }*/
        }
        SizeVec output_indices;
        std::vector<DataType> output_dtypes;
        std::vector<SizeVec> output_shapes;
        for (auto& out : instr.outputs) {
            output_indices.push_back(out.local_index);
            output_dtypes.push_back(out.type.dtype);
            output_shapes.push_back({out.type.shape.begin(), out.type.shape.end()});
            local_sources.at(out.local_index) = instr_index;
        }

        /*if (forward_stream_index >= streams.size() || backward_stream_index >=
        streams.size()) { gpuStream_t new_stream;
            check_error(gpuStreamCreate(&new_stream));
            streams.push_back(new_stream);
        }*/

        instructions.push_back({
            instr.instruction->opcode(),
            input_indices,
            output_indices,
            output_dtypes,
            output_shapes,
            batch_size_index,
            *this,
            instr.instruction->differentiable(),
            new_stream, // streams.at(forward_stream_index),
            new_stream, // streams.at(backward_stream_index),
        });
        for (std::size_t local_index : last_use.local_indices(instr_index)) {
            instructions.push_back(
                {-1, {local_index}, {}, {}, {}, 0, *this, false, new_stream, new_stream}
            );
        }
        ++instr_index;
    }

    for (auto& [name, value] : function.globals()) {
        Tensor global = context->global(name);
        auto& global_shape = value.type.shape;
        Sizes full_shape(global_shape.size() + 1);
        full_shape[0] = 1;
        std::copy(global_shape.begin(), global_shape.end(), full_shape.begin() + 1);
        if (value.type.dtype != global.dtype() || full_shape != global.shape()) {
            throw std::invalid_argument(
                std::format("Global {} has wrong dtype or shape", name)
            );
        }
        locals_init.at(value.local_index) = global;
        if (context->global_requires_grad(name)) {
            requires_grad_init.at(value.local_index) = true;
            grad_global_indices.push_back({name, value.local_index});
        }
    }

    for (auto& local : function.locals()) {
        std::visit(
            Overloaded{
                [&](auto val) {
                    Tensor tensor(val, &GpuDevice::instance());
                    locals_init[local.local_index] = tensor;
                },
                [](std::monostate val) {}
            },
            local.literal_value
        );
    }

    for (auto& out : function.outputs()) {
        output_indices.push_back(out.local_index);
    }
}

GpuRuntime::~GpuRuntime() {
    check_error(gpurandDestroyGenerator(_gpurand_generator));
    check_error(gpublasDestroy(_gpublas_handle));
    for (auto event : events) {
        check_error(gpuEventDestroy(event));
    }
    for (auto stream : streams) {
        check_error(gpuStreamDestroy(stream));
    }
}

TensorVec GpuRuntime::run(const TensorVec& inputs) const {
    auto locals = locals_init;
    std::copy(inputs.begin(), inputs.end(), locals.begin());

    for (auto& instr : instructions) {
        AsyncGpuDevice device(instr.stream);
        for (auto event : instr.wait_events) {
            check_error(gpuStreamWaitEvent(instr.stream, event));
        }
        switch (instr.opcode) {
        case -1: // free memory
            locals[instr.input_indices[0]].reset(device);
            break;
#include "runtime_mixin.h"
        }
        if (instr.record_event) {
            check_error(gpuEventRecord(instr.record_event, instr.stream));
        }
    }
    TensorVec outputs;
    for (auto index : output_indices) {
        outputs.push_back(locals[index]);
    }
    check_error(gpuDeviceSynchronize());
    return outputs;
}

std::tuple<TensorVec, TensorVec, std::vector<bool>> GpuRuntime::run_with_grad(
    const TensorVec& inputs, const std::vector<bool>& input_requires_grad
) const {
    auto locals = locals_init;
    auto requires_grad = requires_grad_init;
    std::vector<bool> store_local(locals.size());
    std::vector<bool> eval_grad(instructions.size());
    std::copy(inputs.begin(), inputs.end(), locals.begin());
    std::copy(
        input_requires_grad.begin(), input_requires_grad.end(), requires_grad.begin()
    );

    for (auto [instr, instr_eval_grad] : zip(instructions, eval_grad)) {
        AsyncGpuDevice device(instr.stream);
        if (instr.differentiable) {
            for (auto input_index : instr.input_indices) {
                if (requires_grad[input_index]) {
                    instr_eval_grad = true;
                    break;
                }
            }
            if (instr_eval_grad) {
                // TODO: only store necessary
                for (auto input_index : instr.input_indices) {
                    store_local[input_index] = true;
                }
                for (auto output_index : instr.output_indices) {
                    store_local[output_index] = true;
                    requires_grad[output_index] = true;
                }
            }
        }
        for (auto event : instr.wait_events) {
            check_error(gpuStreamWaitEvent(instr.stream, event));
        }
        switch (instr.opcode) {
        case -1: { // free memory
            auto input_index = instr.input_indices[0];
            if (!store_local[input_index]) {
                locals[input_index].reset(device);
            }
            break;
        }
#include "runtime_mixin.h"
        }
        if (instr.record_event) {
            check_error(gpuEventRecord(instr.record_event, instr.stream));
        }
    }
    TensorVec outputs;
    for (auto index : output_indices) {
        outputs.push_back(locals[index]);
    }
    check_error(gpuDeviceSynchronize());
    return {outputs, locals, eval_grad};
}

std::tuple<TensorVec, std::vector<std::tuple<std::string, Tensor>>>
GpuRuntime::run_backward(
    const TensorVec& output_grads,
    const TensorVec& stored_locals,
    const std::vector<bool>& eval_grad
) const {
    TensorVec local_grads(stored_locals.size());
    TensorVec locals(stored_locals);
    for (auto [index, grad] : zip(output_indices, output_grads)) {
        local_grads[index] = grad;
    }
    for (auto [instr, instr_eval_grad] :
         zip(std::views::reverse(instructions), std::views::reverse(eval_grad))) {
        if (!instr_eval_grad) {
            continue;
        }
        AsyncGpuDevice device(instr.backward_stream);
        for (auto [output_index, output_dtype] :
             zip(instr.output_indices, instr.output_dtypes)) {
            auto& grad = local_grads[output_index];
            if (!grad && output_dtype == DataType::dt_float) {
                grad = Tensor(DataType::dt_float, locals[output_index].shape(), device);
                grad.zero(device);
            }
        }
        for (auto event : instr.backward_wait_events) {
            check_error(gpuStreamWaitEvent(instr.backward_stream, event));
        }
        switch (instr.opcode) {
#include "runtime_backward_mixin.h"
        }
        if (instr.backward_record_event) {
            check_error(
                gpuEventRecord(instr.backward_record_event, instr.backward_stream)
            );
        }
    }
    std::vector<std::tuple<std::string, Tensor>> global_grads;
    for (auto& [name, index] : grad_global_indices) {
        global_grads.push_back({name, local_grads[index]});
    }
    check_error(gpuDeviceSynchronize());
    return {{local_grads.begin(), local_grads.begin() + input_count}, global_grads};
}

extern "C" Runtime*
build_runtime(const Function& function, ContextPtr context, bool concurrent) {
    return new GpuRuntime(function, context);
}

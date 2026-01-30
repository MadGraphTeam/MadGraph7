#pragma once

#include "gpu_abstraction.h"
#include "madspace/madcode/function.h"
#include "madspace/runtime/runtime_base.h"
#include "madspace/runtime/tensor.h"

#include <memory>

namespace madspace {
namespace gpu {

class GpuRuntime : public Runtime {
public:
    struct Instruction {
        int opcode;
        SizeVec input_indices;
        SizeVec output_indices;
        std::vector<DataType> output_dtypes;
        std::vector<SizeVec> output_shapes;
        std::size_t batch_size_index;
        GpuRuntime& runtime;
        bool differentiable;
        gpuStream_t stream;
        gpuStream_t backward_stream;
        std::vector<gpuEvent_t> wait_events;
        gpuEvent_t record_event;
        std::vector<gpuEvent_t> backward_wait_events;
        gpuEvent_t backward_record_event;
    };

    GpuRuntime(const Function& function, ContextPtr context);
    ~GpuRuntime();
    TensorVec run(const TensorVec& inputs) const override;
    std::tuple<TensorVec, TensorVec, std::vector<bool>> run_with_grad(
        const TensorVec& inputs, const std::vector<bool>& input_requires_grad
    ) const override;
    std::tuple<TensorVec, std::vector<std::tuple<std::string, Tensor>>> run_backward(
        const TensorVec& output_grads,
        const TensorVec& stored_locals,
        const std::vector<bool>& eval_grad
    ) const override;
    Context& context() { return *_context; }
    gpublasHandle_t gpublas_handle() { return _gpublas_handle; }
    gpurandGenerator_t gpurand_generator() { return _gpurand_generator; }

private:
    std::vector<Instruction> instructions;
    SizeVec output_indices;
    std::size_t input_count;
    TensorVec locals_init;
    std::vector<bool> requires_grad_init;
    std::vector<std::tuple<std::string, std::size_t>> grad_global_indices;
    ContextPtr _context;
    std::vector<gpuStream_t> streams;
    std::vector<gpuEvent_t> events;
    gpublasHandle_t _gpublas_handle;
    gpurandGenerator_t _gpurand_generator;
};

extern "C" Runtime*
build_runtime(const Function& function, ContextPtr context, bool concurrent);

} // namespace gpu
} // namespace madspace

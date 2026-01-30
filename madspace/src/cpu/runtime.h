#pragma once

#include <random>

#include "madspace/madcode/function.h"
#include "madspace/runtime/context.h"
#include "madspace/runtime/runtime_base.h"
#include "madspace/runtime/tensor.h"

namespace madspace {
namespace cpu {

class CpuRuntime : public Runtime {
public:
    struct Instruction {
        int opcode;
        SizeVec input_indices;
        SizeVec output_indices;
        std::vector<DataType> output_dtypes;
        std::vector<SizeVec> output_shapes;
        std::size_t batch_size_index;
        CpuRuntime& runtime;
        bool differentiable;
        SizeVec dependent_instructions;
        std::size_t dependency_count;
        SizeVec dependent_instructions_backward;
        std::size_t dependency_count_backward;
    };

    CpuRuntime(const Function& function, ContextPtr context, bool concurrent);

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
    std::mt19937& rand_gen(std::size_t thread_id) { return _rand_gens.get(thread_id); }

private:
    TensorVec run_single(const TensorVec& inputs) const;
    std::tuple<TensorVec, TensorVec, std::vector<bool>> run_with_grad_single(
        const TensorVec& inputs, const std::vector<bool>& input_requires_grad
    ) const;
    std::tuple<TensorVec, TensorVec, std::vector<bool>> run_concurrent(
        const TensorVec& inputs,
        const std::vector<bool>& input_requires_grad,
        bool with_grad
    ) const;
    std::tuple<TensorVec, std::vector<std::tuple<std::string, Tensor>>>
    run_backward_single(
        const TensorVec& output_grads,
        const TensorVec& stored_locals,
        const std::vector<bool>& eval_grad
    ) const;
    std::tuple<TensorVec, std::vector<std::tuple<std::string, Tensor>>>
    run_backward_concurrent(
        const TensorVec& output_grads,
        const TensorVec& stored_locals,
        const std::vector<bool>& eval_grad
    ) const;

    std::vector<Instruction> _instructions;
    SizeVec _output_indices;
    std::size_t _input_count;
    TensorVec _locals_init;
    std::vector<bool> _requires_grad_init;
    std::vector<std::tuple<std::string, std::size_t>> _grad_global_indices;
    ContextPtr _context;
    ThreadResource<std::mt19937> _rand_gens;
    bool _concurrent;
    SizeVec _ready_instructions_init;
    SizeVec _ready_instructions_backward_init;
};

extern "C" Runtime*
build_runtime(const Function& function, ContextPtr context, bool concurrent);

} // namespace cpu
} // namespace madspace

#pragma once

#include <cstdint>
#include <optional>

#include "madspace/compgraphs.hpp"
#include "madspace/driver/context.hpp"
#include "madspace/driver/tensor.hpp"

namespace madspace {

class Runtime {
public:
    virtual ~Runtime() = default;
    // `seed`, when set, makes the random numbers drawn during this call a
    // deterministic function of `seed` alone, independent of which thread executes
    // it. Backends that cannot support this (e.g. GPU, or a CPU runtime executing
    // concurrently) should reject a non-null seed rather than silently ignore it.
    virtual TensorVec
    run(const TensorVec& inputs, std::optional<std::uint64_t> seed = std::nullopt) = 0;
    virtual std::tuple<TensorVec, TensorVec, std::vector<bool>> run_with_grad(
        const TensorVec& inputs,
        const std::vector<bool>& input_requires_grad,
        std::optional<std::uint64_t> seed = std::nullopt
    ) = 0;
    virtual std::pair<TensorVec, TensorVec> run_backward(
        const TensorVec& output_grads,
        const TensorVec& stored_locals,
        const std::vector<bool>& eval_grad,
        bool return_contiguous_grads = false
    ) = 0;

    friend std::unique_ptr<Runtime>
    build_runtime(const Function& function, ContextPtr context, bool concurrent);

private:
    std::shared_ptr<void> shared_lib;
};

using RuntimePtr = std::unique_ptr<Runtime>;
RuntimePtr
build_runtime(const Function& function, ContextPtr context, bool concurrent = true);
std::vector<std::string> available_backends();
DevicePtr cpu_device();
DevicePtr cuda_device(std::size_t index = 0);
DevicePtr hip_device(std::size_t index = 0);
void set_lib_path(const std::string& lib_path);
void set_simd_vector_size(int vector_size);

} // namespace madspace

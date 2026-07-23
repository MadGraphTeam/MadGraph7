#pragma once

#include <cstdint>

#include "madspace/compgraphs.hpp"
#include "madspace/driver/context.hpp"
#include "madspace/driver/tensor.hpp"

namespace madspace {

class Runtime {
public:
    virtual ~Runtime() = default;
    virtual TensorVec run(const TensorVec& inputs) = 0;
    virtual std::tuple<TensorVec, TensorVec, std::vector<bool>> run_with_grad(
        const TensorVec& inputs, const std::vector<bool>& input_requires_grad
    ) = 0;
    virtual std::pair<TensorVec, TensorVec> run_backward(
        const TensorVec& output_grads,
        const TensorVec& stored_locals,
        const std::vector<bool>& eval_grad,
        bool return_contiguous_grads = false
    ) = 0;

    // Deterministic per-job RNG. Between begin_job_rng(seed) and end_job_rng(), all
    // randomness consumed by run() on the CALLING thread is drawn from a generator
    // seeded by `seed`, so a job's random numbers depend only on `seed` and not on
    // which pool thread executes it. Backends that cannot support this (e.g. GPU)
    // keep the default no-op and fall back to their usual per-thread streams.
    virtual void begin_job_rng(std::uint64_t seed) {}
    virtual void end_job_rng() {}

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

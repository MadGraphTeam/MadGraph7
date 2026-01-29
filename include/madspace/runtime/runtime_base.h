#pragma once

#include "madspace/madcode.h"
#include "madspace/runtime/context.h"
#include "madspace/runtime/tensor.h"

namespace madspace {

class Runtime {
public:
    virtual ~Runtime() = default;
    virtual TensorVec run(const TensorVec& inputs) const = 0;
    virtual std::tuple<TensorVec, TensorVec, std::vector<bool>> run_with_grad(
        const TensorVec& inputs, const std::vector<bool>& input_requires_grad
    ) const = 0;
    virtual std::
        tuple<TensorVec, std::vector<std::tuple<std::string, madspace::Tensor>>>
        run_backward(
            const TensorVec& output_grads,
            const TensorVec& stored_locals,
            const std::vector<bool>& eval_grad
        ) const = 0;
    friend std::unique_ptr<Runtime>
    build_runtime(const Function& function, ContextPtr context, bool concurrent);

private:
    std::shared_ptr<void> shared_lib;
};

using RuntimePtr = std::unique_ptr<Runtime>;
RuntimePtr
build_runtime(const Function& function, ContextPtr context, bool concurrent = true);
DevicePtr cpu_device();
DevicePtr cuda_device();
DevicePtr hip_device();
void set_lib_path(const std::string& lib_path);
void set_simd_vector_size(int vector_size);

} // namespace madspace

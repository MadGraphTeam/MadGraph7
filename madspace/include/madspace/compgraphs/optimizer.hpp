#pragma once

#include "madspace/compgraphs/function.hpp"

#include <vector>

namespace madspace {

class InstructionDependencies {
public:
    InstructionDependencies(const Function& function);
    bool depends(std::size_t test_index, std::size_t dependency_index) {
        return _matrix[test_index * _size + dependency_index];
    }
    const std::vector<int>& ranks() const { return _ranks; }

private:
    std::size_t _size;
    std::vector<bool> _matrix;
    std::vector<int> _ranks;
};

class LastUseOfLocals {
public:
    LastUseOfLocals(const Function& function);
    std::vector<std::size_t>& local_indices(std::size_t index) {
        return _last_used.at(index);
    }

private:
    std::vector<std::vector<std::size_t>> _last_used;
};

Function sort_breadth_first(const Function& function);

/**
 * Lowering pass for single-precision networks, run by the GPU backend when it
 * builds a runtime. Fuses a precision conversion into the instruction that
 * produces the converted value:
 *  - `convert_float32(v)` where `v` is an output of `select`, `rqs_forward` or
 *    `rqs_inverse` becomes the `*_with_float32` variant of the producer, which
 *    writes the single-precision copy itself;
 *  - `convert_float64(matmul_float32(x, w, b))`, with the matmul result used only
 *    by the conversion, becomes `matmul_to_float64`.
 * The graph keeps only the generic instructions and so stays portable; the fused
 * instructions never appear in a saved function.
 */
Function fuse_conversions(const Function& function);

} // namespace madspace

#pragma once

#include "madspace/madcode/function.h"

#include <vector>

namespace madspace {

class InstructionDependencies {
public:
    InstructionDependencies(const Function& function);
    bool depends(std::size_t test_index, std::size_t dependency_index) {
        return matrix[test_index * size + dependency_index];
    }

private:
    std::size_t size;
    std::vector<bool> matrix;
    std::vector<int> ranks;
};

class LastUseOfLocals {
public:
    LastUseOfLocals(const Function& function);
    std::vector<int>& local_indices(std::size_t index) { return last_used[index]; }

private:
    std::vector<std::vector<int>> last_used;
};

} // namespace madspace

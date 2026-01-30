#include "madspace/madcode/optimizer.h"

#include <algorithm>
#include <numeric>
#include <ranges>
#include <unordered_map>

#include "madspace/util.h"

using namespace madspace;

InstructionDependencies::InstructionDependencies(const Function& function) :
    size(function.instructions().size()), matrix(size * size) {
    std::vector<int> local_source(function.locals().size(), -1);
    int index = 0;
    for (auto& instr : function.instructions()) {
        int rank = 0;
        for (auto& input : instr.inputs) {
            auto source_index = local_source.at(input.local_index);
            if (source_index == -1) {
                continue;
            }
            matrix.at(index * size + source_index) = true;
            for (int i = 0; i < size; ++i) {
                matrix.at(index * size + i) =
                    matrix.at(index * size + i) | matrix.at(source_index * size + i);
            }
            int source_rank = ranks.at(source_index);
            if (rank < source_rank) {
                rank = source_rank;
            }
        }
        for (auto& output : instr.outputs) {
            local_source.at(output.local_index) = index;
        }
        ranks.push_back(rank + 1);
        ++index;
    }
}

LastUseOfLocals::LastUseOfLocals(const Function& function) :
    last_used(function.instructions().size()) {
    std::vector<bool> seen_locals;
    for (auto& local : function.locals()) {
        seen_locals.push_back(
            !std::holds_alternative<std::monostate>(local.literal_value)
        );
    }
    for (auto& output : function.outputs()) {
        seen_locals.at(output.local_index) = true;
    }
    auto instr = function.instructions().rbegin();
    auto indices = last_used.begin();
    for (; instr != function.instructions().rend(); ++instr, ++indices) {
        for (auto& input : instr->inputs) {
            auto index = input.local_index;
            if (!seen_locals.at(index)) {
                indices->push_back(index);
                seen_locals.at(index) = true;
            }
        }
    }
    std::reverse(last_used.begin(), last_used.end());
}

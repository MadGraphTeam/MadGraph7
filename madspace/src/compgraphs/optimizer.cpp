#include "madspace/compgraphs/optimizer.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <numeric>
#include <ranges>
#include <unordered_map>

#include "madspace/util.hpp"

using namespace madspace;

InstructionDependencies::InstructionDependencies(const Function& function) :
    _size(function.instructions().size()), _matrix(_size * _size) {
    std::vector<int> local_source(function.locals().size(), -1);
    int index = 0;
    for (auto& instr : function.instructions()) {
        int rank = 0;
        for (auto& input : instr.inputs) {
            auto source_index = local_source.at(input.local_index);
            if (source_index == -1) {
                continue;
            }
            _matrix.at(index * _size + source_index) = true;
            for (int i = 0; i < _size; ++i) {
                _matrix.at(index * _size + i) = _matrix.at(index * _size + i) |
                    _matrix.at(source_index * _size + i);
            }
            int source_rank = _ranks.at(source_index);
            if (rank < source_rank) {
                rank = source_rank;
            }
        }
        for (auto& output : instr.outputs) {
            local_source.at(output.local_index) = index;
        }
        _ranks.push_back(rank + 1);
        ++index;
    }
}

LastUseOfLocals::LastUseOfLocals(const Function& function) :
    _last_used(function.instructions().size()) {
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
    auto indices = _last_used.begin();
    for (; instr != function.instructions().rend(); ++instr, ++indices) {
        for (auto& input : instr->inputs) {
            auto index = input.local_index;
            if (!seen_locals.at(index)) {
                indices->push_back(index);
                seen_locals.at(index) = true;
            }
        }
    }
    std::reverse(_last_used.begin(), _last_used.end());
}

Function madspace::sort_breadth_first(const Function& function) {
    Function func_out = function;
    InstructionDependencies dependencies(function);
    auto order = dependencies.ranks();
    std::vector<std::size_t> instruction_perm(function.instructions().size());
    std::iota(instruction_perm.begin(), instruction_perm.end(), 0);
    std::stable_sort(
        instruction_perm.begin(),
        instruction_perm.end(),
        [&](std::size_t i, std::size_t j) { return order.at(i) < order.at(j); }
    );
    func_out._instructions.clear();
    for (std::size_t index : instruction_perm) {
        func_out._instructions.push_back(function._instructions.at(index));
    }
    return func_out;
}

Function madspace::fuse_conversions(const Function& function) {
    Function func_out = function;
    auto& instructions = func_out._instructions;
    std::vector<int> producer(function.locals().size(), -1);
    std::vector<int> use_count(function.locals().size(), 0);
    for (std::size_t i = 0; i < instructions.size(); ++i) {
        for (auto& output : instructions[i].outputs) {
            producer.at(output.local_index) = i;
        }
        for (auto& input : instructions[i].inputs) {
            if (input.local_index != -1) {
                ++use_count.at(input.local_index);
            }
        }
    }
    for (auto& output : function.outputs()) {
        ++use_count.at(output.local_index);
    }

    auto find_instruction = [](const std::string& name) {
        auto find = instruction_set.find(name);
        if (find == instruction_set.end()) {
            throw std::runtime_error(std::format("Unknown instruction '{}'", name));
        }
        return find->second.get();
    };

    // the precision of an activation instruction that fuses into a matmul, or empty
    auto activation_precision = [](const std::string& name) -> std::string {
        static const std::array<std::string, 6> activations{
            "relu", "leaky_relu", "elu", "gelu", "sigmoid", "softplus"
        };
        for (const auto& activation : activations) {
            for (std::string precision : {"float32", "float64"}) {
                if (name == activation + "_" + precision) {
                    return precision;
                }
            }
        }
        return "";
    };

    std::vector<bool> removed(instructions.size());
    bool any_fused = false;
    for (std::size_t i = 0; i < instructions.size(); ++i) {
        auto& call = instructions[i];
        const std::string& name = call.instruction->name();
        std::string precision = activation_precision(name);
        if (name != "convert_float32" && name != "convert_float64" &&
            precision.empty()) {
            continue;
        }
        auto& input = call.inputs.at(0);
        if (input.local_index == -1) {
            continue;
        }
        int source_index = producer.at(input.local_index);
        if (source_index == -1 || removed.at(source_index)) {
            continue;
        }
        auto& source = instructions.at(source_index);
        // the fused instruction takes the place of the producer
        if (source.stream_index != call.stream_index) {
            continue;
        }
        const std::string& source_name = source.instruction->name();
        if (!precision.empty()) {
            if (source_name != "matmul_" + precision) {
                continue;
            }
            // the matmul output stays an output, the activation needs it backward
            source.instruction = find_instruction("matmul_" + name);
            source.outputs.push_back(call.outputs.at(0));
        } else if (name == "convert_float32") {
            bool fusable = source_name == "select" || source_name == "rqs_forward" ||
                source_name == "rqs_inverse";
            if (!fusable || source.outputs.at(0).local_index != input.local_index ||
                input.type.dtype != DataType::dt_float64) {
                continue;
            }
            source.instruction = find_instruction(source_name + "_with_float32");
            source.outputs.push_back(call.outputs.at(0));
        } else {
            if (source_name != "matmul_float32" ||
                use_count.at(input.local_index) != 1) {
                continue;
            }
            source.instruction = find_instruction("matmul_to_float64");
            source.outputs = {call.outputs.at(0)};
        }
        producer.at(call.outputs.at(0).local_index) = source_index;
        removed.at(i) = true;
        any_fused = true;
    }
    if (!any_fused) {
        return func_out;
    }

    std::vector<InstructionCall> kept;
    kept.reserve(instructions.size());
    for (std::size_t i = 0; i < instructions.size(); ++i) {
        if (!removed.at(i)) {
            kept.push_back(std::move(instructions[i]));
        }
    }
    instructions = std::move(kept);
    return func_out;
}

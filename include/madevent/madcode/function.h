#pragma once

#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "instruction.h"

namespace madevent {

struct InstructionCall {
    InstructionPtr instruction;
    ValueVec inputs;
    ValueVec outputs;
};

class Function {
public:
    friend class FunctionBuilder;

    Function() = default;

    const ValueVec& inputs() const { return _inputs; }
    const ValueVec& outputs() const { return _outputs; }
    const ValueVec& locals() const { return _locals; }
    const std::unordered_map<std::string, Value>& globals() const { return _globals; }
    const std::vector<InstructionCall>& instructions() const { return _instructions; }

    void save(const std::string& file) const;
    static Function load(const std::string& file);

private:
    Function(
        const ValueVec& inputs,
        const ValueVec& outputs,
        const ValueVec& locals,
        const std::unordered_map<std::string, Value>& globals,
        const std::vector<InstructionCall>& instructions
    ) :
        _inputs(inputs),
        _outputs(outputs),
        _locals(locals),
        _globals(globals),
        _instructions(instructions) {}

    ValueVec _inputs;
    ValueVec _outputs;
    ValueVec _locals;
    std::unordered_map<std::string, Value> _globals;
    std::vector<InstructionCall> _instructions;
};

std::ostream& operator<<(std::ostream& out, const Value& value);
std::ostream& operator<<(std::ostream& out, const ValueVec& list);
std::ostream& operator<<(std::ostream& out, const InstructionCall& call);
std::ostream& operator<<(std::ostream& out, const Function& func);

void to_json(nlohmann::json& j, const InstructionCall& call);
void to_json(nlohmann::json& j, const Function& call);
void from_json(const nlohmann::json& j, Function& call);

class FunctionBuilder {
public:
    FunctionBuilder(
        const std::vector<Type> _input_types, const std::vector<Type> _output_types
    );
    FunctionBuilder(const Function& function);
    Value input(int index) const;
    ValueVec input_range(int start_index, int end_index) const;
    void output(int index, Value value);
    void output_range(int start_index, const ValueVec& values);
    Value
    global(const std::string& name, DataType dtype, const std::vector<int>& shape);
    ValueVec instruction(const std::string& name, const ValueVec& args);
    ValueVec instruction(InstructionPtr instruction, const ValueVec& args);
    Function function();

    Value sum(const ValueVec& values);
    Value product(const ValueVec& values);

#include "function_builder_mixin.h"

private:
    std::vector<Type> output_types;
    ValueVec inputs;
    std::vector<std::optional<Value>> outputs;
    std::map<LiteralValue, Value> literals;
    ValueVec locals;
    std::unordered_map<std::string, Value> globals;
    std::vector<InstructionCall> instructions;
    std::map<std::vector<std::size_t>, std::vector<std::size_t>> instruction_cache;
    std::vector<int> local_sources;
    std::vector<bool> instruction_used;

    void register_local(Value& val);
};

} // namespace madevent

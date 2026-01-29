#pragma once

#include <format>
#include <vector>

#include "madspace/phasespace/base.h"
#include "madspace/phasespace/phasespace.h"

namespace madspace {

class MultiChannelMapping : public Mapping {
public:
    MultiChannelMapping(const std::vector<std::shared_ptr<Mapping>>& mappings);

private:
    Result build_impl(
        FunctionBuilder& fb,
        const ValueVec& inputs,
        const ValueVec& conditions,
        bool inverse
    ) const;
    Result build_forward_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override {
        return build_impl(fb, inputs, conditions, false);
    }
    Result build_inverse_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override {
        return build_impl(fb, inputs, conditions, true);
    }

    std::vector<std::shared_ptr<Mapping>> _mappings;
};

class MultiChannelFunction : public FunctionGenerator {
public:
    MultiChannelFunction(
        const std::vector<std::shared_ptr<FunctionGenerator>>& functions
    );

private:
    ValueVec
    build_function_impl(FunctionBuilder& fb, const ValueVec& args) const override;

    std::vector<std::shared_ptr<FunctionGenerator>> _functions;
};

} // namespace madspace

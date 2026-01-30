#pragma once

#include "madspace/phasespace/base.h"
#include "madspace/phasespace/mlp.h"

namespace madspace {

class MomentumPreprocessing : public FunctionGenerator {
public:
    MomentumPreprocessing(std::size_t particle_count);
    std::size_t output_dim() const { return _output_dim; };

private:
    ValueVec
    build_function_impl(FunctionBuilder& fb, const ValueVec& args) const override;

    std::size_t _output_dim;
};

class ChannelWeightNetwork : public FunctionGenerator {
public:
    ChannelWeightNetwork(
        std::size_t channel_count,
        std::size_t particle_count,
        std::size_t hidden_dim = 32,
        std::size_t layers = 3,
        MLP::Activation activation = MLP::leaky_relu,
        const std::string& prefix = ""
    );

    const MLP& mlp() const { return _mlp; }
    const MomentumPreprocessing& preprocessing() const { return _preprocessing; }
    void initialize_globals(ContextPtr context) const;
    const std::string& mask_name() const { return _mask_name; }

private:
    ValueVec
    build_function_impl(FunctionBuilder& fb, const ValueVec& args) const override;

    MomentumPreprocessing _preprocessing;
    MLP _mlp;
    std::size_t _channel_count;
    std::string _mask_name;
};

} // namespace madspace

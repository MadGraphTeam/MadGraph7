#pragma once

#include "madspace/phasespace/base.h"
#include "madspace/phasespace/invariants.h"

namespace madspace {

class TwoBodyDecay : public Mapping {
public:
    TwoBodyDecay(bool com) :
        Mapping(
            "TwoBodyDecay",
            [&] {
                TypeVec input_types(5, batch_float);
                if (!com) {
                    input_types.push_back(batch_four_vec);
                }
                return input_types;
            }(),
            {batch_four_vec, batch_four_vec},
            {}
        ),
        _com(com) {}
    std::size_t random_dim() const { return 2; }

private:
    Result build_forward_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;

    bool _com;
};

class TwoToTwoParticleScattering : public Mapping {
public:
    TwoToTwoParticleScattering(
        bool com, double invariant_power = 0, double mass = 0, double width = 0
    ) :
        Mapping(
            "TwoToTwoParticleScattering",
            {batch_float, batch_float, batch_float, batch_float},
            {batch_four_vec, batch_four_vec},
            {batch_four_vec, batch_four_vec}
        ),
        _com(com),
        _invariant(invariant_power, mass, width) {}

private:
    Result build_forward_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;

    bool _com;
    Invariant _invariant;
};

} // namespace madspace

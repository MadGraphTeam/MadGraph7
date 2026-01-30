#pragma once

#include "madspace/phasespace/base.h"
#include "madspace/phasespace/invariants.h"

namespace madspace {

class ThreeBodyDecay : public Mapping {
public:
    ThreeBodyDecay(bool com) :
        Mapping(
            "ThreeBodyDecay",
            [&] {
                TypeVec input_types(9, batch_float);
                if (!com) {
                    input_types.push_back(batch_four_vec);
                }
                return input_types;
            }(),
            {batch_four_vec, batch_four_vec, batch_four_vec},
            {}
        ),
        _com(com) {}
    std::size_t random_dim() const { return 5; }

private:
    Result build_forward_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;

    bool _com;
};

class TwoToThreeParticleScattering : public Mapping {
public:
    TwoToThreeParticleScattering(
        double t_invariant_power = 0,
        double t_mass = 0,
        double t_width = 0,
        double s_invariant_power = 0,
        double s_mass = 0,
        double s_width = 0
    ) :
        Mapping(
            "TwoToThreeParticleScattering",
            {batch_float, batch_float, batch_float, batch_float, batch_float},
            {batch_four_vec, batch_four_vec},
            {batch_four_vec, batch_four_vec, batch_four_vec}
        ),
        _t_invariant(t_invariant_power, t_mass, t_width),
        _s_invariant(s_invariant_power, s_mass, s_width) {}

private:
    Result build_forward_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;

    Invariant _t_invariant;
    Invariant _s_invariant;
};

} // namespace madspace

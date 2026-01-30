#pragma once

#include <vector>

#include "madspace/phasespace/base.h"
#include "madspace/phasespace/invariants.h"
#include "madspace/phasespace/topology.h"
#include "madspace/phasespace/two_particle.h"

namespace madspace {

class TPropagatorMapping : public Mapping {
public:
    TPropagatorMapping(
        const std::vector<std::size_t>& integration_order, double invariant_power = 0.8
    );
    std::size_t random_dim() const { return 3 * _integration_order.size() - 1; }

private:
    Result build_forward_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;

    std::vector<std::size_t> _integration_order;
    std::vector<bool> _sample_sides;
    Invariant _uniform_invariant;
    TwoToTwoParticleScattering _com_scattering;
    TwoToTwoParticleScattering _lab_scattering;
};

} // namespace madspace

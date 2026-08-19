#pragma once

#include <vector>

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/invariants.hpp"
#include "madspace/phasespace/topology.hpp"
#include "madspace/phasespace/two_particle.hpp"

namespace madspace {

class TPropagatorMapping : public Mapping {
public:
    TPropagatorMapping(
        const std::vector<std::size_t>& integration_order,
        double invariant_power = 0.8,
        const std::vector<double>& pt_min = {},
        bool return_invariants = false
    );
    std::size_t random_dim() const { return 3 * _integration_order.size() - 1; }

    // Masks for every invariant sampled by TPropagatorMapping: one per t-channel
    // propagator (physical order along the chain), followed by one per
    // intermediate s-channel propagator. outgoing_masks are the masks for the
    // attached legs
    std::vector<int> invariant_masks(const std::vector<int>& outgoing_masks) const;

    static std::size_t invariant_count(std::size_t t_propagator_count) {
        return 2 * t_propagator_count - 1;
    }

private:
    Result build_forward_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override;

    // pt^2 of the outgoing particle at position i (0 if no pt cut).
    double pt2(std::size_t i) const;

    std::vector<std::size_t> _integration_order;
    std::vector<bool> _sample_sides;
    std::vector<double> _pt_min;
    bool _has_cut;
    bool _return_invariants;
    Invariant _uniform_invariant;
    TwoToTwoParticleScattering _com_scattering;
    TwoToTwoParticleScattering _lab_scattering;
};

} // namespace madspace

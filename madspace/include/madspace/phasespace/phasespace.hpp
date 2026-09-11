#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/chili.hpp"
#include "madspace/phasespace/color_ordered_mapping.hpp"
#include "madspace/phasespace/cuts.hpp"
#include "madspace/phasespace/invariants.hpp"
#include "madspace/phasespace/luminosity.hpp"
#include "madspace/phasespace/rambo.hpp"
#include "madspace/phasespace/t_propagator_mapping.hpp"
#include "madspace/phasespace/three_particle.hpp"
#include "madspace/phasespace/topology.hpp"

namespace madspace {

class PhaseSpaceMapping : public Mapping {
public:
    enum TChannelMode { propagator, rambo, chili, color_ordered };

    PhaseSpaceMapping(
        const Topology& topology,
        double cm_energy,
        bool leptonic = false,
        double invariant_power = 0.8,
        TChannelMode t_channel_mode = propagator,
        const std::optional<Cuts>& cuts = std::nullopt,
        const std::vector<std::vector<std::size_t>>& permutations = {},
        const std::optional<std::vector<std::size_t>>& color_order = std::nullopt
    );

    PhaseSpaceMapping(
        const std::vector<double>& external_masses,
        double cm_energy,
        bool leptonic = false,
        double invariant_power = 0.8,
        TChannelMode mode = rambo,
        const std::optional<Cuts>& cuts = std::nullopt,
        const std::optional<std::vector<std::size_t>>& color_order = std::nullopt
    );

    // A 1 -> n decay and a leptonic (fixed-s) 2 -> n collision both have 3n-4
    // degrees of freedom; a hadronic 2 -> n adds the beam momentum fractions,
    // i.e. one further sampled invariant on top of s_hat -> 3n-2.
    static std::size_t random_dim_for(const Topology& topology, bool leptonic) {
        return 3 * topology.outgoing_masses().size() -
            ((leptonic || topology.is_decay()) ? 4 : 2);
    }
    std::size_t random_dim() const { return random_dim_for(_topology, _leptonic); }
    std::size_t discrete_dim() const override { return _n_discrete; }
    std::size_t particle_count() const {
        return _topology.outgoing_masses().size() + _topology.incoming_masses().size();
    }
    std::size_t channel_count() const { return _permutations.size(); }

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

    Topology _topology;
    Cuts _cuts;
    double _pi_factors;
    double _sqrt_s_lab;
    bool _leptonic;
    bool _map_luminosity;
    std::size_t _n_discrete;
    std::vector<Invariant> _s_invariants;
    std::variant<
        TPropagatorMapping,
        FastRamboMapping,
        ChiliMapping,
        ColorOrderedMapping,
        std::monostate>
        _t_mapping;
    std::vector<std::variant<TwoBodyDecay, ThreeBodyDecay, FastRamboMapping>> _s_decays;
    nested_vector2<me_int_t> _permutations;
};

} // namespace madspace

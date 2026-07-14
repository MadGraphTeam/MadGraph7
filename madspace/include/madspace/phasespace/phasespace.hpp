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
        const std::optional<std::vector<std::size_t>>& color_order = std::nullopt,
        bool produce_virtuality = false
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

    std::size_t random_dim() const {
        return 3 * _topology.outgoing_masses().size() - (_leptonic ? 4 : 2);
    }
    std::size_t discrete_dim() const override { return _n_discrete; }
    std::size_t particle_count() const {
        return _topology.outgoing_masses().size() + 2;
    }
    std::size_t channel_count() const { return _permutations.size(); }
    bool produce_virtuality() const { return _produce_virtuality; }

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

    // Virtuality passthrough: when enabled, build_forward additionally outputs a
    // flattened (npar x npar) real matrix v[i][j] = p^2 - M^2 for every first-level
    // propagator that is the fusion of exactly two external legs (i,j). All three kinds
    // are covered: s-channel two-outgoing and the two-incoming s_hat current use the
    // numerically clean *sampled* invariant (a Breit-Wigner resonance keeps its clean
    // value); first-level t-channel (one incoming + one outgoing) uses p^2 = t recomputed
    // from the momenta in double (spacelike, no cancellation). Entries that are not such a
    // propagator in the sampled channel are left at 0, which the ME treats as "recompute
    // from momenta". See UMAMI_IN_VIRTUALITY / wf_fixp2_map.
    bool _produce_virtuality;
    // for each 2-external-leg propagator k: the decay index (to read its sampled p^2 =
    // mass2) and the propagator mass squared M^2
    std::vector<std::size_t> _virt_decay_index;
    std::vector<double> _virt_mass2;
    // _virt_slot_of_pos[i * npar + j][channel] = index k of the propagator feeding output
    // position (i,j) in that channel (permutation), or _virt_decay_index.size() (sentinel)
    // if no propagator sits there. Used with select() to scatter the values per event.
    nested_vector2<me_int_t> _virt_slot_of_pos;

    // First-level t-channel propagators (one incoming + one outgoing external leg). Their
    // p^2 = t is spacelike (no Breit-Wigner cancellation), so it is recomputed from the
    // momenta in double via invariants_from_momenta rather than sampled. _t_momentum_factors
    // is the deduplicated list of per-channel permuted momentum-coefficient vectors; for each
    // output position (i,j) the tables give, per channel, the index into that invariants
    // vector (sentinel _t_momentum_factors.size() if none) and the propagator mass squared.
    nested_vector2<double> _t_momentum_factors;
    nested_vector2<me_int_t> _t_factor_of_pos;
    nested_vector2<double> _t_mass2_of_pos;
};

} // namespace madspace

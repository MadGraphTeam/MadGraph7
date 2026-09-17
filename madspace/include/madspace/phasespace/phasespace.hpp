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
        double beam_rapidity = 0.,
        bool mirror_beams = false
    );

    PhaseSpaceMapping(
        const std::vector<double>& external_masses,
        double cm_energy,
        bool leptonic = false,
        double invariant_power = 0.8,
        TChannelMode mode = rambo,
        const std::optional<Cuts>& cuts = std::nullopt,
        const std::optional<std::vector<std::size_t>>& color_order = std::nullopt,
        double beam_rapidity = 0.,
        bool mirror_beams = false
    );

    // mirror_beams: add a "mirror_index" condition (after permutation_index).
    // For index 1 the event is rotated by pi about the x axis in the beams'
    // centre-of-mass frame -- leg 1 then comes from beam 2 and leg 2 from beam
    // 1 -- before the boost into the lab frame and the cuts. With identical
    // beams that is a rotation of the lab event as well, which is why the
    // Integrand can otherwise mirror accepted events after the cuts.
    //
    // beam_rapidity: rapidity of the beams' centre-of-mass frame in the frame
    // the momenta are returned in (the lab frame), 0.5 ln(E1 / E2) for beam
    // energies E1 (along +z) and E2. The phase space is generated in the beams'
    // centre-of-mass frame, of energy cm_energy = 2 sqrt(E1 E2), and boosted
    // into the lab frame before the cuts, so rapidity cuts act in the lab.

    // A 1 -> n decay and a leptonic (fixed-s) 2 -> n collision both have 3n-4
    // degrees of freedom; a hadronic 2 -> n adds the beam momentum fractions,
    // i.e. one further sampled invariant on top of s_hat -> 3n-2.
    // A hadronic 2 -> 1 collision keeps a single random number, the rapidity
    // of the produced particle; s_hat is fixed at its mass.
    static std::size_t random_dim_for(const Topology& topology, bool leptonic) {
        if (topology.outgoing_masses().size() == 1 &&
            (leptonic || topology.is_decay())) {
            throw std::invalid_argument(
                "a 2 -> 1 process needs beams with a PDF: with fixed beam "
                "energies s_hat is fixed and there is nothing to integrate"
            );
        }
        return 3 * topology.outgoing_masses().size() -
            ((leptonic || topology.is_decay()) ? 4 : 2);
    }
    std::size_t random_dim() const { return random_dim_for(_topology, _leptonic); }
    std::size_t discrete_dim() const override { return _n_discrete; }
    std::size_t particle_count() const {
        return _topology.outgoing_masses().size() + _topology.incoming_masses().size();
    }
    std::size_t channel_count() const { return _permutations.size(); }
    double beam_rapidity() const { return _beam_rapidity; }
    bool mirror_beams() const { return _mirror_beams; }
    const Cuts& cuts() const { return _cuts; }

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
    // partonic centre-of-mass frame <-> lab frame (partonic boost, mirror,
    // beam_rapidity boost)
    Value to_lab(
        FunctionBuilder& fb,
        Value momenta,
        Value x1,
        Value x2,
        const NamedVector<Value>& conditions
    ) const;
    Value from_lab(
        FunctionBuilder& fb,
        Value momenta,
        Value x1,
        Value x2,
        const NamedVector<Value>& conditions
    ) const;

    Topology _topology;
    Cuts _cuts;
    double _pi_factors;
    double _sqrt_s_lab;
    double _beam_rapidity;
    bool _mirror_beams;
    bool _leptonic;
    bool _map_luminosity;
    bool _two_to_one;
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

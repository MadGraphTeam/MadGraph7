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

/**
 * Full @f$2 \to n@f$ phase-space mapping for one diagram topology.
 *
 * Assembles a complete mapping from the elementary building blocks along the
 * recursive decomposition of the tree-level phase space (Sec. 2.2 of [1]),
 *
 * @f[
 *   \int \mathrm{d}\Phi_n =
 *   \Big[\textstyle\prod_i \int \mathrm{d}s_i\Big]
 *   \Big[\textstyle\prod_j \int \mathrm{d}\Phi^{(\phi,t)}_{2,j}\Big]
 *   \Big[\textstyle\prod_k \int \mathrm{d}\Phi^{(\tilde s,t)}_{2,k}\Big]
 *   \Big[\textstyle\prod_l \int \mathrm{d}\Phi_{3,l}\Big]
 *   \Big[\textstyle\prod_m \int \mathrm{d}\Phi^{(\phi,\theta)}_{2,m}\Big].
 * @f]
 *
 * The time-like invariants are sampled with @ref Invariant, the s-channel
 * decays with @ref TwoBodyDecay / @ref ThreeBodyDecay, and the PDF convolution
 * with @ref Luminosity. @ref TChannelMode selects the t-channel strategy. The
 * second constructor takes a flat list of external masses instead of a
 * @ref Topology and is used for the topology-free @ref TChannelMode::rambo and
 * @ref TChannelMode::chili modes.
 *
 * `batch` is the leading batch dimension. `n_out` is the number of outgoing
 * particles.
 *
 * **Inputs**
 * - `random` – `float`, shape `(batch, random_dim())` – the unit-hypercube
 *   coordinates.
 * - `discrete` – `int`, shape `(batch, discrete_dim())` – the discrete channel
 *   choices. Present only when `discrete_dim()` is nonzero.
 *
 * **Conditions**
 * - `permutation_index` – `int`, shape `(batch,)` – which permutation channel
 *   to use. Present only when more than one permutation is given.
 *
 * **Outputs**
 * - `momenta` – `float`, shape `(batch, n_out + 2, 4)` – all momenta (incoming
 *   beams first, then outgoing).
 * - `x1` – `float`, shape `(batch,)` – first parton momentum fraction.
 * - `x2` – `float`, shape `(batch,)` – second parton momentum fraction.
 *
 * In addition every mapping returns a `weight` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 2.2)
 */
class PhaseSpaceMapping : public Mapping {
public:
    /// How the space-like (t-channel) part of the phase space is generated.
    enum TChannelMode {
        propagator,   ///< A chain of 2->2 blocks (@ref TPropagatorMapping).
        rambo,        ///< @ref FastRamboMapping on the t-channel legs.
        chili,        ///< Collider coordinates (@ref ChiliMapping).
        color_ordered ///< A color-ordered chain (@ref ColorOrderedMapping).
    };

    /**
     * @param topology        The diagram topology to follow. See @ref Topology.
     * @param cm_energy        Total collision energy.
     * @param leptonic         If true, skip the PDF convolution (no `x1`/`x2`
     *                         sampling).
     * @param invariant_power  Exponent of the `1/s^p` sampling of every
     *                         time-like invariant. See @ref Invariant.
     * @param t_channel_mode   The @ref TChannelMode strategy.
     * @param cuts             Fiducial cuts to fold into the sampling. See
     *                         @ref Cuts.
     * @param permutations     One outgoing-particle permutation per channel.
     *                         More than one enables the `permutation_index`
     *                         condition.
     * @param color_order      Color ordering for @ref TChannelMode::color_ordered.
     */
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

    /**
     * @param external_masses  Masses of the outgoing particles. A trivial
     *                         topology is built from them.
     * @param cm_energy        Total collision energy.
     * @param leptonic         If true, skip the PDF convolution.
     * @param invariant_power  Exponent of the `1/s^p` invariant sampling. See
     *                         @ref Invariant.
     * @param mode             The @ref TChannelMode strategy (default
     *                         @ref TChannelMode::rambo).
     * @param cuts             Fiducial cuts. See @ref Cuts.
     * @param color_order      Color ordering for @ref TChannelMode::color_ordered.
     */
    PhaseSpaceMapping(
        const std::vector<double>& external_masses,
        double cm_energy,
        bool leptonic = false,
        double invariant_power = 0.8,
        TChannelMode mode = rambo,
        const std::optional<Cuts>& cuts = std::nullopt,
        const std::optional<std::vector<std::size_t>>& color_order = std::nullopt
    );

    /// Number of continuous unit-hypercube inputs consumed by the forward map.
    std::size_t random_dim() const {
        return 3 * _topology.outgoing_masses().size() - (_leptonic ? 4 : 2);
    }
    /// Number of discrete channel choices.
    std::size_t discrete_dim() const override { return _n_discrete; }
    /// Total number of particles, incoming and outgoing.
    std::size_t particle_count() const {
        return _topology.outgoing_masses().size() + 2;
    }
    /// Number of permutation channels.
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

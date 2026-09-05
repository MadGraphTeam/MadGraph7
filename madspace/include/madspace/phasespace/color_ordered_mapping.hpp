#pragma once

#include <vector>

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/invariants.hpp"
#include "madspace/phasespace/three_particle.hpp"
#include "madspace/phasespace/topology.hpp"
#include "madspace/phasespace/two_particle.hpp"

namespace madspace {

/**
 * Leading-color phase-space mapping following a color-ordered particle chain.
 *
 * Generates the full final state as a chain of @f$(\tilde s, t)@f$ two-particle
 * blocks (@ref TwoToThreeParticleScattering) laid out along a color-ordered
 * permutation of the particles, peeling one on-shell momentum off at a time
 * from each beam, with a @ref TwoToTwoParticleScattering or @ref DoubleT
 * central block joining the two sides [1]. It encodes no specific diagram
 * topology. The `discrete_i` inputs select, per @f$2 \to 3@f$ peel, which of
 * the two branches to take. See Sec. 2.2.5 of [1], following [2].
 *
 * `batch` is the leading batch dimension. `n_out` is the number of outgoing
 * particles and `n = n_out + 2` the total.
 *
 * **Inputs**
 * - `random_i` – `float`, shape `(batch,)` – the continuous random numbers
 *   (`random_dim()` of them).
 * - `discrete_i` – `int`, shape `(batch,)` – the two-solution choices
 *   (`discrete_dim()` of them).
 *
 * **Conditions**
 * - `com_energy` – `float`, shape `(batch,)` – total collision energy.
 * - `mass_i` – `float`, shape `(batch,)` – the `n_out` outgoing masses.
 *
 * **Outputs**
 * - `momentum_i` – `float`, shape `(batch, 4)` – the `n` momenta (incoming
 *   beams first, then outgoing).
 *
 * In addition every mapping returns a `weight` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 2.2.5)
 * - [2] R. Frederix, T. Vitos, "Leading-colour-based unweighted event
 *   generation for multi-parton tree-level processes",
 *   https://arxiv.org/abs/2409.12128
 */
class ColorOrderedMapping : public Mapping {
public:
    /**
     * @param color_order       0-based permutation of `{0, ..., n-1}`
     *                          (`n = n_out + 2`) giving the chain order;
     *                          particles 0 and 1 are the incoming beams. See
     *                          @ref Topology.
     * @param t_invariant_power Exponent of the `1/|t|^p` sampling of every
     *                          momentum transfer; see @ref Invariant.
     * @param s_invariant_power Exponent of the `1/s^p` sampling of every
     *                          time-like invariant; see @ref Invariant.
     * @param pt_min            Per-outgoing-particle minimum transverse
     *                          momentum; empty disables the cut. See @ref Cuts.
     * @param m_inv_min         Symmetric matrix of minimum pair invariant
     *                          masses; empty disables the cut. See @ref Cuts.
     * @param dr_min            Symmetric matrix of minimum pair `delta_r`
     *                          separations; empty disables the cut. See
     *                          @ref Cuts.
     */
    ColorOrderedMapping(
        const std::vector<std::size_t>& color_order,
        double t_invariant_power = 0.8,
        double s_invariant_power = 0.8,
        const std::vector<double>& pt_min = {},
        const std::vector<std::vector<double>>& m_inv_min = {},
        const std::vector<std::vector<double>>& dr_min = {}
    );

    /// Number of continuous unit-hypercube inputs consumed by the forward map.
    std::size_t random_dim() const { return _random_dim; }
    /// Number of discrete two-solution choices (one per 2->3 peel).
    std::size_t discrete_dim() const override { return _discrete_dim; }

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

    // pt^2 of outgoing particle i (0 if no pt cut on it).
    double pt2(std::size_t i) const;
    // Cut-derived invariant-mass^2 floor (gen23 invm_min, without the mass^2
    // term, which is applied separately) for a subset of outgoing particles.
    // Returns 0 when cuts are disabled or the subset has fewer than 2 members.
    double cut_floor(const std::vector<std::size_t>& subset) const;

    // 0-indexed outgoing-particle indices (values in {0,...,n_out-1}).
    // _set1 contains the outgoing particles attached to beam 0's side,
    // _set2 those attached to beam 1's side, in peel order.
    std::vector<std::size_t> _set1;
    std::vector<std::size_t> _set2;
    std::size_t _n_out;
    std::size_t _random_dim;
    // Number of discrete two-solution choices (one per 2->3 peel). These are
    // supplied/recovered as a separate batch_int channel, not through the
    // continuous random_dim() block (opt-in r_disc).
    std::size_t _discrete_dim;
    // True iff exactly one of (set1, set2) has size 1 (and the other >= 2).
    // In that case the central block is DoubleT instead of 2->2.
    bool _use_double_t;
    // True iff one of (set1, set2) is empty, i.e. particles 0 and 1 are
    // adjacent in the color order and all outgoing particles sit on one side.
    // In that case there is no central block at all: the full final state is
    // produced as a single t-channel chain seeded directly off the beams.
    bool _use_single_chain;

    // Cut configuration (empty => all bounds resolve to 0 = no cut).
    std::vector<double> _pt_min;
    std::vector<std::vector<double>> _m_inv_min;
    std::vector<std::vector<double>> _dr_min;
    bool _has_cut;

    Invariant _uniform_invariant;
    TwoToTwoParticleScattering _com_scattering;
    TwoToTwoParticleScattering _lab_scattering;
    TwoToThreeParticleScattering _two_to_three;
    DoubleT _double_t;
};

} // namespace madspace

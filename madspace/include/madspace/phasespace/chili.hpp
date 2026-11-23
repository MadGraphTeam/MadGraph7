#pragma once

#include "madspace/phasespace/base.hpp"

#include <vector>

namespace madspace {

/**
 * Chili mapping: the final state directly in collider coordinates.
 *
 * Generates the momenta in transverse momentum, rapidity and azimuth rather
 * than along a diagram topology, reconstructing the incoming momentum fractions
 * from the final-state kinematics [1] (conceptually the ALPGEN construction,
 * [4]). Dropping overall factors, the measure is
 *
 * @f[
 *   \mathrm{d}\Phi_n = \delta^{(4)}\!\Big(p_a + p_b - \sum_i p_i\Big)
 *   \prod_{i=1}^{n} \frac{\mathrm{d}p_{\mathrm{T},i}^2\,\mathrm{d}y_i\,
 *   \mathrm{d}\phi_i}{4}.
 * @f]
 *
 * The first @f$n-1@f$ momenta are generated explicitly. The transverse
 * components of the last are fixed by recoil, and its rapidity is sampled
 * uniformly. Each @f$p_\mathrm{T}^2@f$ is drawn from
 *
 * @f[
 *   p_\mathrm{T}^2 = \Big(\tfrac{r}{p_{\mathrm{T},\max}^2}
 *   + \tfrac{1-r}{p_{\mathrm{T},\min}^2}\Big)^{-1}
 * @f]
 *
 * when a lower cut applies, and from a small-@f$p_\mathrm{T}@f$-regulated map
 * otherwise. Azimuths are sampled relative to the running recoil direction,
 * @f$\phi = 2\pi r_\phi + \phi_\mathrm{rec}@f$ (Sec. 2.4 of [1], following
 * [2, 3]). Chili does not guarantee physical configurations, so unphysical
 * points are removed by a technical cut.
 *
 * With fixed beams (lepton colliders, `fixed_beams`) the incoming momenta add
 * up to @f$P = (\sqrt{s}, 0, 0, 0)@f$, and the final state cannot fix them.
 * The first @f$n-2@f$ momenta are generated as above, and the last two as the
 * two-body decay of what is left, @f$Q = P - \sum_{i<n-1} p_i@f$, in the rest
 * frame of @f$Q@f$, with
 *
 * @f[
 *   \mathrm{d}\Phi_n = \prod_{i=1}^{n-2} \frac{\mathrm{d}p_{\mathrm{T},i}^2\,
 *   \mathrm{d}y_i\,\mathrm{d}\phi_i}{4}\;
 *   \frac{|\vec p^{\,*}|}{4 M_Q}\,\mathrm{d}\Omega^*,
 * @f]
 *
 * which takes @f$3n-4@f$ random numbers. A point with @f$Q@f$ below the
 * threshold of the last two masses gets zero weight. @ref PhaseSpaceMapping
 * also takes it for dressed-lepton beams, whose split samples
 * @f$x_1, x_2@f$: chili then builds the final state for fixed beams at
 * @f$\sqrt{\hat s}@f$ in their partonic frame.
 *
 * `batch` is the leading batch dimension.
 *
 * **Inputs**
 * - `random_i` – `float`, shape `(batch,)` – the `random_dim()` random
 *   numbers.
 *
 * **Conditions**
 * - `com_energy` – `float`, shape `(batch,)` – total collision energy.
 * - `mass_i` – `float`, shape `(batch,)` – the `n_particles` outgoing masses.
 *
 * **Outputs**
 * - `momentum_i` – `float`, shape `(batch, 4)` – the `n_particles + 2` momenta
 *   (incoming beams first, then outgoing). With fixed beams the incoming ones
 *   are the massless beams in their centre-of-mass frame.
 *
 * In addition every mapping returns a `det` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 2.4)
 * - [2] E. Bothmann et al., "Efficient phase-space generation for hadron
 *   collider event simulation", https://arxiv.org/abs/2302.10449
 * - [3] E. Bothmann et al., "A portable parton-level event generator for the
 *   high-luminosity LHC", https://arxiv.org/abs/2311.06198
 * - [4] M. L. Mangano et al., "ALPGEN, a generator for hard multiparton
 *   processes in hadronic collisions", https://arxiv.org/abs/hep-ph/0206293
 */
class ChiliMapping : public Mapping {
public:
    /**
     * @param n_particles Number of outgoing particles.
     * @param y_max       Per-outgoing-particle maximum rapidity, one entry per
     *                    particle; infinity disables the cut. See @ref Cuts.
     * @param pt_min      Per-outgoing-particle minimum transverse momentum, one
     *                    entry per particle (the last is unused, and with
     *                    fixed beams the last two); 0 disables the cut. See
     *                    @ref Cuts.
     * @param fixed_beams If true, the beams have fixed momenta (see above);
     *                    needs at least two outgoing particles.
     */
    ChiliMapping(
        std::size_t n_particles,
        const std::vector<double>& y_max,
        const std::vector<double>& pt_min,
        bool fixed_beams = false
    );

    /// Number of uniform random inputs consumed by the forward mapping:
    /// `3 * n_particles - 2`, or `3 * n_particles - 4` with fixed beams.
    std::size_t random_dim() const { return 3 * _n_particles - (_fixed_beams ? 4 : 2); }
    /// Whether the beams have fixed momenta.
    bool fixed_beams() const { return _fixed_beams; }

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

    std::size_t _n_particles;
    std::vector<double> _y_max;
    std::vector<double> _pt_min;
    bool _fixed_beams;
};

} // namespace madspace

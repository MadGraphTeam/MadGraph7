#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/invariants.hpp"

namespace madspace {

/**
 * Elementary @f$1 \to 2@f$ decay block, angular parametrization.
 *
 * Splits a parent momentum @f$p_0@f$ into two on-shell momenta
 * @f$p_0 = p_1 + p_2@f$ [1]. Two-body kinematics fix the energies and
 * three-momentum magnitudes from the masses. The remaining freedom is the
 * polar and azimuthal angles of @f$\vec p_1@f$ in the @f$p_0@f$ rest frame,
 * sampled uniformly,
 *
 * @f[
 *   \phi = 2\pi\,r_\phi, \qquad \cos\theta = 2\,r_{\cos\theta} - 1.
 * @f]
 *
 * The momenta are then boosted back (Sec. 2.2.3 of [1]). The measure is flat
 * in @f$(\phi, \cos\theta)@f$.
 *
 * `batch` is the leading batch dimension. The masses are inputs here. They are
 * supplied on the forward call and recovered by the inverse.
 *
 * **Inputs**
 * - `random_phi` – `float`, shape `(batch,)` – azimuthal-angle random number.
 * - `random_cos_theta` – `float`, shape `(batch,)` – polar-angle random number.
 * - `mass0` – `float`, shape `(batch,)` – parent mass @f$m_0@f$.
 * - `mass1` – `float`, shape `(batch,)` – first daughter mass.
 * - `mass2` – `float`, shape `(batch,)` – second daughter mass.
 * - `com_momentum` – `float`, shape `(batch, 4)` – parent four-momentum.
 *   Present only when @p com is false.
 *
 * **Conditions**
 * - None.
 *
 * **Outputs**
 * - `momentum1` – `float`, shape `(batch, 4)` – first daughter momentum.
 * - `momentum2` – `float`, shape `(batch, 4)` – second daughter momentum.
 *
 * In addition every mapping returns a `weight` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 2.2.3)
 */
class TwoBodyDecay : public Mapping {
public:
    /// @param com  If true the decay is generated in the parent rest frame,
    ///             otherwise the parent momentum is taken from the
    ///             `com_momentum` input.
    TwoBodyDecay(bool com);
    /// Number of uniform random inputs consumed by the forward mapping (2).
    std::size_t random_dim() const { return 2; }

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

    bool _com;
};

/**
 * Two-particle phase-space block parametrized by the azimuth and the momentum
 * transfer @f$t@f$.
 *
 * Same two-particle phase space as @ref TwoBodyDecay. The polar angle is traded
 * for the Mandelstam invariant @f$t = (p_\mathrm{in,1} - p_1)^2 < 0@f$, which
 * is linear in @f$\cos\theta@f$ [1]. The azimuth is uniform,
 * @f$\phi = 2\pi\,r_\phi@f$. @f$|t|@f$ is drawn with an @ref Invariant (from
 * @p invariant_power / @p mass / @p width) so a t-channel propagator pole is
 * flattened (Sec. 2.2.4 of [1]). The momenta are built in the scattering
 * center-of-mass frame and boosted out.
 *
 * `batch` is the leading batch dimension. The masses are inputs here. They are
 * supplied on the forward call and recovered by the inverse.
 *
 * **Inputs**
 * - `random_phi` – `float`, shape `(batch,)` – azimuthal-angle random number.
 * - `random_inv` – `float`, shape `(batch,)` – random number for @f$|t|@f$.
 * - `mass1` – `float`, shape `(batch,)` – first outgoing mass.
 * - `mass2` – `float`, shape `(batch,)` – second outgoing mass.
 *
 * **Conditions**
 * - `momentum_in1` – `float`, shape `(batch, 4)` – first incoming momentum.
 * - `momentum_in2` – `float`, shape `(batch, 4)` – second incoming momentum.
 * - `etmin_1` – `float`, shape `(batch,)` – transverse-energy cut on the first
 *   outgoing particle. Present only when @p has_cut is true.
 * - `etmin_2` – `float`, shape `(batch,)` – transverse-energy cut on the
 *   second outgoing particle. Present only when @p has_cut is true.
 *
 * **Outputs**
 * - `momentum1` – `float`, shape `(batch, 4)` – first outgoing momentum.
 * - `momentum2` – `float`, shape `(batch, 4)` – second outgoing momentum.
 *
 * In addition every mapping returns a `weight` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 2.2.4)
 */
class TwoToTwoParticleScattering : public Mapping {
public:
    /**
     * @param com             If true, momenta are built in the scattering
     *                        center-of-mass frame.
     * @param invariant_power Exponent of the @f$1/|t|^{\,p}@f$ sampling of the
     *                        momentum transfer; see @ref Invariant.
     * @param mass            t-channel propagator mass; see @ref Invariant.
     * @param width           t-channel propagator width; a nonzero value
     *                        selects Breit-Wigner sampling. See @ref Invariant.
     * @param has_cut         If true, the `etmin_*` conditions restrict
     *                        @f$|t|@f$ to the region passing the transverse
     *                        cuts; see @ref Cuts.
     */
    TwoToTwoParticleScattering(
        bool com,
        double invariant_power = 0,
        double mass = 0,
        double width = 0,
        bool has_cut = false
    );

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

    bool _com;
    Invariant _invariant;
    bool _has_cut;
};

/**
 * Central two-particle block with both legs sampled in a momentum transfer.
 *
 * A variant of the @f$(\phi, t)@f$ two-particle block (Sec. 2.2.4 of [1]) in
 * which both outgoing legs are attached to a t-channel propagator, so two
 * momentum transfers @f$t_1, t_2@f$ are drawn (each with its own
 * @ref Invariant) in addition to the shared azimuth. It is used as the central
 * block of the @ref ColorOrderedMapping chain.
 *
 * `batch` is the leading batch dimension.
 *
 * **Inputs**
 * - `random_phi` – `float`, shape `(batch,)` – azimuthal-angle random number.
 * - `random_t1` – `float`, shape `(batch,)` – random number for @f$|t_1|@f$.
 * - `random_t2` – `float`, shape `(batch,)` – random number for @f$|t_2|@f$.
 *
 * **Conditions**
 * - `momentum_in1` – `float`, shape `(batch, 4)` – first incoming momentum.
 * - `momentum_in2` – `float`, shape `(batch, 4)` – second incoming momentum.
 * - `mass1` – `float`, shape `(batch,)` – mass of the first outgoing particle.
 * - `mass_rest_min` – `float`, shape `(batch,)` – minimum invariant mass of the
 *   remaining system.
 * - `etmin_i` – `float`, shape `(batch,)` – transverse-energy cut on the
 *   outgoing particle. Present only when @p has_cut is true.
 * - `etmin_ir` – `float`, shape `(batch,)` – transverse-energy cut on the
 *   recoil system. Present only when @p has_cut is true.
 *
 * **Outputs**
 * - `momentum1` – `float`, shape `(batch, 4)` – first outgoing momentum.
 * - `momentum2` – `float`, shape `(batch, 4)` – recoil momentum.
 *
 * In addition every mapping returns a `weight` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 2.2.4)
 */
class DoubleT : public Mapping {
public:
    /**
     * @param t1_invariant_power Exponent of the @f$1/|t_1|^{\,p}@f$ sampling;
     *                           see @ref Invariant.
     * @param t1_mass            First t-channel propagator mass; see
     *                           @ref Invariant.
     * @param t1_width           First t-channel propagator width; see
     *                           @ref Invariant.
     * @param t2_invariant_power Exponent of the @f$1/|t_2|^{\,p}@f$ sampling;
     *                           see @ref Invariant.
     * @param t2_mass            Second t-channel propagator mass; see
     *                           @ref Invariant.
     * @param t2_width           Second t-channel propagator width; see
     *                           @ref Invariant.
     * @param has_cut            If true, the `etmin_*` conditions restrict the
     *                           momentum transfers to the region passing the
     *                           transverse cuts; see @ref Cuts.
     */
    DoubleT(
        double t1_invariant_power = 0,
        double t1_mass = 0,
        double t1_width = 0,
        double t2_invariant_power = 0,
        double t2_mass = 0,
        double t2_width = 0,
        bool has_cut = false
    );

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

    Invariant _t1_invariant;
    Invariant _t2_invariant;
    bool _has_cut;
};

} // namespace madspace

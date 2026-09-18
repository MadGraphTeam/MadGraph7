#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/invariants.hpp"

namespace madspace {

/**
 * Genuine @f$1 \to 3@f$ decay block.
 *
 * Splits a parent momentum into three on-shell momenta,
 * @f$p_0 = p_1 + p_2 + p_3@f$ [1]. In the parent rest frame the five degrees of
 * freedom are the daughter energies @f$E_1, E_2@f$, the polar and azimuthal
 * angles of @f$\vec p_1@f$, and an azimuth @f$\beta@f$ of @f$\vec p_2@f$ about
 * @f$\vec p_1@f$; the opening angle between @f$\vec p_1@f$ and @f$\vec p_2@f$ is
 * fixed by energy-momentum conservation. All five variables are sampled flat
 * within their kinematic ranges, giving a constant @f$1/8@f$ measure
 * (Sec. 2.2.6 of [1], following [2]).
 *
 * `batch` is the leading batch dimension. Masses are inputs here: supplied on
 * the forward call and recovered by the inverse.
 *
 * **Inputs**
 * - `random_energy1` – `float`, shape `(batch,)` – random number for @f$E_1@f$.
 * - `random_energy2` – `float`, shape `(batch,)` – random number for @f$E_2@f$.
 * - `random_phi` – `float`, shape `(batch,)` – azimuth of @f$\vec p_1@f$.
 * - `random_cos_theta` – `float`, shape `(batch,)` – polar angle of
 *   @f$\vec p_1@f$.
 * - `random_beta` – `float`, shape `(batch,)` – azimuth of @f$\vec p_2@f$ about
 *   @f$\vec p_1@f$.
 * - `mass0` – `float`, shape `(batch,)` – parent mass.
 * - `mass1`, `mass2`, `mass3` – `float`, shape `(batch,)` – daughter masses.
 * - `com_momentum` – `float`, shape `(batch, 4)` – parent four-momentum.
 *   Present only when @p com is false.
 *
 * **Conditions**
 * - None.
 *
 * **Outputs**
 * - `momentum1`, `momentum2`, `momentum3` – `float`, shape `(batch, 4)` – the
 *   daughter momenta.
 *
 * In addition every mapping returns a `weight` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 2.2.6)
 * - [2] G. Knippen, PhD thesis, University of Freiburg (2019),
 *   https://doi.org/10.6094/UNIFR/154629
 */
class ThreeBodyDecay : public Mapping {
public:
    /// @param com  If true the decay is generated in the parent rest frame,
    ///             otherwise the parent momentum is taken from the
    ///             `com_momentum` input.
    ThreeBodyDecay(bool com);
    /// Number of uniform random inputs consumed by the forward mapping (5).
    std::size_t random_dim() const { return 5; }

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
 * Two-particle block in the double-invariant @f$(\tilde s, t)@f$ parametrization.
 *
 * Replaces both angles of the two-particle block by Lorentz invariants. The
 * polar angle becomes the momentum transfer @f$t_{i-1}@f$ and the azimuth an
 * extra time-like invariant @f$\tilde s_i@f$ [1]. The pair @f$(\tilde s, t)@f$
 * alone does not fix the momenta. A recoil momentum (`momentum3`) supplies the
 * scattering plane. This block therefore peels one on-shell momentum off a
 * composite cluster and appears inside a recursive chain. The measure carries
 * an inverse square-root Gram determinant,
 *
 * @f[
 *   \int \mathrm{d}\Phi_2^{(\tilde s, t)} =
 *   \int \mathrm{d}\tilde s_i \int \mathrm{d}|t_{i-1}|\;
 *   \frac{1}{8\sqrt{-\Delta_4}}.
 * @f]
 *
 * @f$|t_{i-1}|@f$ is drawn with an @ref Invariant (from @p t_invariant_power /
 * @p t_mass / @p t_width) and @f$\tilde s_i@f$ with another (from
 * @p s_invariant_power / @p s_mass / @p s_width). The `discrete_choice` input
 * selects one of the two @f$\cos\phi@f$ branches. See Sec. 2.2.5 of [1],
 * following [2, 3].
 *
 * `batch` is the leading batch dimension.
 *
 * **Inputs**
 * - `discrete_choice` – `int`, shape `(batch,)` – which of the two two-body
 *   solutions to take.
 * - `random_s23` – `float`, shape `(batch,)` – random number for
 *   @f$\tilde s_i@f$.
 * - `random_t1` – `float`, shape `(batch,)` – random number for @f$|t_{i-1}|@f$.
 * - `mass1`, `mass2` – `float`, shape `(batch,)` – the two resolved masses.
 *
 * **Conditions**
 * - `momentum_in1` – `float`, shape `(batch, 4)` – first incoming momentum.
 * - `momentum_in2` – `float`, shape `(batch, 4)` – second incoming momentum.
 * - `momentum3` – `float`, shape `(batch, 4)` – recoil momentum defining the
 *   scattering plane.
 * - `etmin_1`, `etmin_2` – `float`, shape `(batch,)` – transverse-energy cuts.
 *   Present only when @p has_cut is true.
 * - `drcut` – `float`, shape `(batch,)` – minimum angular separation.
 *   Present only when @p has_cut is true.
 * - `s23_min_cut` – `float`, shape `(batch,)` – minimum @f$\tilde s@f$.
 *   Present only when @p has_cut is true.
 *
 * **Outputs**
 * - `momentum1`, `momentum2` – `float`, shape `(batch, 4)` – the two resolved
 *   momenta.
 *
 * In addition every mapping returns a `weight` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 2.2.5)
 * - [2] E. Byckling, K. Kajantie, Phys. Rev. 187 (1969) 2008,
 *   https://doi.org/10.1103/PhysRev.187.2008
 * - [3] R. Frederix, T. Vitos, "Leading-colour-based unweighted event
 *   generation for multi-parton tree-level processes",
 *   https://arxiv.org/abs/2409.12128
 */
class TwoToThreeParticleScattering : public Mapping {
public:
    /**
     * @param t_invariant_power Exponent of the @f$1/|t|^{\,p}@f$ sampling; see
     *                          @ref Invariant.
     * @param t_mass            t-channel propagator mass; see @ref Invariant.
     * @param t_width           t-channel propagator width; see @ref Invariant.
     * @param s_invariant_power Exponent of the @f$1/\tilde s^{\,p}@f$ sampling;
     *                          see @ref Invariant.
     * @param s_mass            s-channel resonance mass; see @ref Invariant.
     * @param s_width           s-channel resonance width; see @ref Invariant.
     * @param has_cut           If true, the `etmin_*`, `drcut` and `s23_min_cut`
     *                          conditions restrict the invariants to the region
     *                          passing the cuts; see @ref Cuts.
     */
    TwoToThreeParticleScattering(
        double t_invariant_power = 0,
        double t_mass = 0,
        double t_width = 0,
        double s_invariant_power = 0,
        double s_mass = 0,
        double s_width = 0,
        bool has_cut = false
    );

    /// Number of discrete inputs (1): which of the two two-body solutions.
    std::size_t discrete_dim() const override { return 1; }

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

    Invariant _t_invariant;
    Invariant _s_invariant;
    bool _has_cut;
};

} // namespace madspace

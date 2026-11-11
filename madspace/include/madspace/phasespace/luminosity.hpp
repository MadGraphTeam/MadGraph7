#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/invariants.hpp"

namespace madspace {

/**
 * PDF-convolution mapping: parton momentum fractions from two random numbers.
 *
 * Trades the parton momentum fractions @f$x_1, x_2@f$ for the partonic
 * invariant @f$\hat s = x_1 x_2 s_\mathrm{lab}@f$ and one angular variable [1].
 * @f$\hat s@f$ is drawn in `[s_hat_min, s_hat_max]` with an @ref Invariant,
 * which forwards @p invariant_power, @p mass and @p width. The momentum
 * fractions then follow from logarithmic sampling,
 *
 * @f[
 *   x_1 = \tau^{\,r}, \qquad x_2 = \tau^{\,1-r},
 *   \qquad \tau = \hat s / s_\mathrm{lab},
 * @f]
 *
 * which cancels the @f$1/x_1@f$ of the convolution measure. Choosing
 * @p invariant_power @f$\simeq 1@f$ (the default) also absorbs the @f$1/\hat s@f$
 * flux factor. A nonzero @p width instead resolves an s-channel resonance in
 * @f$\hat s@f$ directly (Sec. 2.2.2 of [1]). This parametrization is specific
 * to hadron collisions.
 *
 * `batch` is the leading batch dimension.
 *
 * **Inputs**
 * - `r_s` – `float`, shape `(batch,)` – random number for @f$\hat s@f$.
 * - `r_x` – `float`, shape `(batch,)` – random number for the @f$x_1/x_2@f$
 *   split.
 *
 * **Conditions**
 * - None.
 *
 * **Outputs**
 * - `x1` – `float`, shape `(batch,)` – first parton momentum fraction.
 * - `x2` – `float`, shape `(batch,)` – second parton momentum fraction.
 * - `s_hat` – `float`, shape `(batch,)` – partonic invariant
 *   @f$\hat s = x_1 x_2 s_\mathrm{lab}@f$.
 *
 * In addition every mapping returns a `det` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 2.2.2)
 */
class Luminosity : public Mapping {
public:
    /**
     * @param s_lab           Hadronic invariant @f$s_\mathrm{lab}@f$ (squared
     *                        collider energy).
     * @param s_hat_min       Lower bound on @f$\hat s@f$, set by the final-state
     *                        masses and analysis cuts.
     * @param s_hat_max       Upper bound on @f$\hat s@f$; `0` uses
     *                        @f$s_\mathrm{lab}@f$.
     * @param invariant_power Exponent of the @f$1/\hat s^{\,p}@f$ sampling of
     *                        @f$\hat s@f$; `1` absorbs the flux factor. See
     *                        @ref Invariant.
     * @param mass            Resonance mass for @f$\hat s@f$; see @ref Invariant.
     * @param width           Resonance width; a nonzero value selects
     *                        Breit-Wigner sampling of @f$\hat s@f$. See
     *                        @ref Invariant.
     */
    Luminosity(
        double s_lab,
        double s_hat_min,
        double s_hat_max = 0,
        double invariant_power = 1,
        double mass = 0,
        double width = 0
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

    double _s_lab, _s_hat_min, _s_hat_max;
    Invariant _invariant;
};

/**
 * Momentum fractions of two lepton beams with PDFs, as generate_ee_tau_y of
 * MadGraph's genps_fks.f samples them (lpp = +-3, +-4, without its
 * Breit-Wigner channel).
 *
 * A lepton PDF peaks as @f$(1 - x)^{\beta - 1}@f$ at @f$x \to 1@f$. Each
 * momentum fraction is drawn from its own random number (generate_x_ee),
 * @f[
 *   1 - x_k = (1 - x_{k,\min})\, r_k^{1 / (1 - \beta)},
 * @f]
 * with @f$x_{1,\min} = \tau_\min = \hat s_\min / s_\mathrm{lab}@f$ and
 * @f$x_{2,\min} = \tau_\min / x_1@f$, so that
 * @f$\hat s = x_1 x_2 s_\mathrm{lab} \ge \hat s_\min@f$. The Jacobian of
 * each, @f$(1 - x_{k,\min})^{1 - \beta} (1 - x_k)^\beta / (1 - \beta)@f$,
 * cancels the peak of the PDF for @f$\beta@f$ close to its exponent (the
 * Fortran leaves the factor @f$(1 - x_k)^\beta@f$ to the PDF). The fractions
 * also come as @f$1 - x_k@f$, exact up to the smallest normal double, for the
 * PDFs close to @f$x = 1@f$.
 *
 * `batch` is the leading batch dimension.
 *
 * **Inputs**
 * - `r1`, `r2` – `float`, shape `(batch,)` – random numbers of @f$x_1@f$ and
 *   @f$x_2@f$.
 *
 * **Conditions**
 * - `s_hat_min` – `float`, shape `(batch,)` – @f$\hat s_\min@f$ of the event,
 *   in place of that of the constructor. Present only when
 *   `s_hat_min_condition` is set.
 *
 * **Outputs**
 * - `x1`, `x2` – `float`, shape `(batch,)` – momentum fractions of the beams.
 * - `s_hat` – `float`, shape `(batch,)` – partonic invariant
 *   @f$\hat s = x_1 x_2 s_\mathrm{lab}@f$.
 * - `one_minus_x1`, `one_minus_x2` – `float`, shape `(batch,)` –
 *   @f$1 - x_1@f$ and @f$1 - x_2@f$. The inverse takes the fractions from
 *   them.
 *
 * In addition every mapping returns a `det` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 */
class FKSLeptonLuminosity : public Mapping {
public:
    /**
     * @param s_lab                Squared collider energy
     *                             @f$s_\mathrm{lab}@f$.
     * @param beta                 Exponent @f$\beta@f$ of the sampling, below
     *                             1 (get_ee_expo of the PDF of MadGraph, 0.975
     *                             for its built-in lepton PDFs).
     * @param s_hat_min            Lower bound on @f$\hat s@f$, from the masses
     *                             and the cuts, below @f$s_\mathrm{lab}@f$.
     * @param s_hat_min_condition  Take the lower bound per event from the
     *                             `s_hat_min` condition.
     * @throws std::invalid_argument for parameters out of their range.
     */
    FKSLeptonLuminosity(
        double s_lab,
        double beta,
        double s_hat_min = 0.,
        bool s_hat_min_condition = false
    );
    /// The squared collider energy.
    double s_lab() const { return _s_lab; }
    /// The exponent of the sampling.
    double beta() const { return _beta; }
    /// The lower bound on s_hat of the constructor.
    double s_hat_min() const { return _s_hat_min; }
    /// Whether the mapping takes the `s_hat_min` condition.
    bool s_hat_min_condition() const { return _s_hat_min_condition; }

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
    // tau_min of the event
    Value tau_min(FunctionBuilder& fb, const NamedVector<Value>& conditions) const;

    double _s_lab, _beta, _s_hat_min;
    bool _s_hat_min_condition;
};

} // namespace madspace

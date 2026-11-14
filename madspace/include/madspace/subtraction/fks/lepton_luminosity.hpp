#pragma once

#include "madspace/phasespace/base.hpp"

namespace madspace {

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
 * The fractions are those of @ref LeptonLuminosity, which the dressed-lepton
 * beams use. This mapping differs in three ways. Its Jacobian keeps the
 * factors @f$(1 - x_k)^\beta@f$ that LeptonLuminosity leaves to the reduced
 * density, @f$\hat s_\min@f$ can come per event from a condition, and its
 * outputs have other names and another order.
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

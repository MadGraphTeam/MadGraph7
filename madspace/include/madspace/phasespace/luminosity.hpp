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
 * In addition every mapping returns a `weight` (`float`, shape `(batch,)`), the
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
 * Momentum fractions for dressed-lepton beams, sampled directly.
 *
 * @ref Luminosity trades @f$x_1, x_2@f$ for @f$\hat s@f$ and a splitting
 * variable, which suits a hadron collider. A dressed-lepton density instead
 * diverges as @f$x \to 1@f$ on *each* beam, so the two fractions are drawn
 * separately from that endpoint,
 *
 * @f[
 *   1-x = r^{1/(1-p)} \,(1 - x_\mathrm{min}),
 * @f]
 * i.e. @f$x@f$ distributed as @f$(1-x)^{-p}@f$ over
 * @f$[x_\mathrm{min}, 1]@f$, with @f$x_\mathrm{min}@f$ set by
 * @f$\hat s_\mathrm{min}@f$ (and, for the second beam, by the first fraction).
 *
 * **@p sampling_power must equal the `ps_expo` the density was reduced with.**
 * The returned Jacobian deliberately omits the @f$(1-x)^{p}@f$ of the
 * transformation, because @ref EEPartonDensity carries exactly that factor:
 * the two cancel analytically, which is the whole point of the reduction and
 * avoids evaluating a divergence against a vanishing weight. Using a
 * different exponent here leaves the cross section wrong by
 * @f$(1-x)^{\Delta p}@f$ with nothing to flag it. (This mirrors `genps.f`,
 * where the beam whose @f$x@f$ is *derived* rather than generated has the
 * factor put back by hand.)
 *
 * @f$1-x@f$ is returned alongside @f$x@f$: it is the quantity the density
 * needs, and recovering it as @f$1-x@f$ from a rounded @f$x@f$ throws away
 * the precision the sampling had in hand.
 *
 * `batch` is the leading batch dimension.
 *
 * **Inputs**
 * - `r1`, `r2` – `float`, shape `(batch,)` – random numbers, one per beam.
 *
 * **Conditions**
 * - None.
 *
 * **Outputs**
 * - `x1`, `x2` – `float`, shape `(batch,)` – momentum fractions.
 * - `omx1`, `omx2` – `float`, shape `(batch,)` – @f$1-x_1@f$, @f$1-x_2@f$.
 * - `s_hat` – `float`, shape `(batch,)` – @f$x_1 x_2 s_\mathrm{lab}@f$.
 */
class LeptonLuminosity : public Mapping {
public:
    /**
     * @param s_lab          Squared collider energy.
     * @param s_hat_min      Lower bound on @f$\hat s@f$, from the final-state
     *                       masses and the cuts.
     * @param sampling_power @f$p@f$ of the @f$(1-x)^{-p}@f$ sampling. Must
     *                       match the density's reduction exponent; the
     *                       default is `get_ee_expo()` of MadEvent.
     */
    LeptonLuminosity(double s_lab, double s_hat_min, double sampling_power = 0.96);

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

    double _s_lab, _s_hat_min, _power;
};

/// Result of @ref lepton_split: the two fractions and the Jacobian.
struct LeptonSplitResult {
    Value x1, x2, omx1, omx2, det;
};

/**
 * Split a given @f$\hat s@f$ into two dressed-lepton momentum fractions.
 *
 * madspace samples @f$\hat s@f$ first -- the topology's root invariant is
 * what carries the resonance structure -- and only then splits it, so this is
 * the shape a dressed-lepton run needs. The hadronic split
 * (@f$x_1 = \tau^r@f$) puts no weight at @f$x \to 1@f$, which is exactly
 * where a dressed-lepton density lives.
 *
 * The larger fraction is drawn from the endpoint over
 * @f$[\sqrt\tau, 1]@f$ and the other follows as @f$\tau/x@f$; which beam
 * carries which is decided per event, so both orderings appear.
 *
 * **The derived fraction's @f$(1-x)^{-p}@f$ is put back explicitly.** The
 * drawn one gets that factor by omission -- the density cancels it -- but the
 * derived one never went through the transformation, so nothing cancels it
 * there. `genps.f` makes the same correction in the same place.
 *
 * @param fb       Builder to emit into.
 * @param s_hat    The invariant to split.
 * @param r_x      Random number for the drawn fraction.
 * @param r_swap   Random number deciding which beam carries it.
 * @param s_lab    Squared collider energy.
 * @param power    @f$p@f$; must match the density's reduction exponent.
 *
 * The returned Jacobian is that of @f$(\hat s, r) \to (x_1, x_2)@f$, the same
 * convention `r_to_x1x2` uses, so it already carries the
 * @f$d\hat s \to d\tau@f$ conversion.
 */
LeptonSplitResult lepton_split(
    FunctionBuilder& fb,
    Value s_hat,
    Value r_x,
    Value r_swap,
    double s_lab,
    double power
);

/**
 * Dressed-lepton fractions sampled through a resonance in @f$\hat s@f$.
 *
 * @ref LeptonLuminosity covers the @f$x \to 1@f$ endpoint, which is where a
 * dressed-lepton density puts its weight, but it is a poor fit to a narrow
 * s-channel resonance. This is the companion channel: @f$\hat s@f$ is drawn
 * with an @ref Invariant, exactly as @ref Luminosity does, and only then split
 * into the two fractions. It is a variance-reduction channel — results are
 * correct without it — and is meant to be mixed with @ref LeptonLuminosity by
 * the usual multichannel weights.
 *
 * Only the *larger* fraction is drawn from the endpoint, over
 * @f$[\sqrt\tau, 1]@f$; the other follows as @f$\tau/x@f$. Which beam gets
 * which is decided per event, so both orderings are covered.
 *
 * **The derived fraction needs its @f$(1-x)^{-p}@f$ put back.** The density is
 * stored reduced by @f$p@f$ and the drawn fraction's Jacobian omits that
 * factor to cancel it (see @ref LeptonLuminosity); the derived one never went
 * through that transformation, so nothing cancels it there and the Jacobian
 * has to supply it. `genps.f` makes the same correction, in the same place.
 *
 * `batch` is the leading batch dimension.
 *
 * **Inputs**
 * - `r_s` – `float`, shape `(batch,)` – random number for @f$\hat s@f$.
 * - `r_x` – `float`, shape `(batch,)` – random number for the drawn fraction.
 * - `r_swap` – `float`, shape `(batch,)` – random number deciding which beam
 *   carries the drawn fraction.
 *
 * **Conditions**
 * - None.
 *
 * **Outputs**
 * - `x1`, `x2`, `omx1`, `omx2`, `s_hat` – as in @ref LeptonLuminosity.
 */
class LeptonLuminosityInvariant : public Mapping {
public:
    /**
     * @param s_lab          Squared collider energy.
     * @param s_hat_min      Lower bound on @f$\hat s@f$.
     * @param s_hat_max      Upper bound; `0` uses @f$s_\mathrm{lab}@f$.
     * @param sampling_power @f$p@f$; must match the density's reduction
     *                       exponent, as for @ref LeptonLuminosity.
     * @param mass           Resonance mass in @f$\hat s@f$.
     * @param width          Resonance width; nonzero selects Breit-Wigner
     *                       sampling. See @ref Invariant.
     * @param invariant_power Exponent of the @f$1/\hat s^{\,n}@f$ sampling.
     */
    LeptonLuminosityInvariant(
        double s_lab,
        double s_hat_min,
        double s_hat_max = 0.,
        double sampling_power = 0.96,
        double mass = 0.,
        double width = 0.,
        double invariant_power = 1.
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

    double _s_lab, _s_hat_min, _s_hat_max, _power;
    Invariant _invariant;
};

} // namespace madspace

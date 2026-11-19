#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace madspace {

/**
 * Shape of the collider-dependent factor of the CIRCE beamstrahlung spectrum.
 *
 * The spectrum is @f$a\,e^{p u}\,Q(u)\,(1-y)^{-2/3}@f$ with @f$u = 1-y@f$, and
 * @f$Q@f$ differs between machines. Getting it wrong is silent -- the
 * normalization of the split drifts and nothing fails -- so it is named
 * explicitly rather than defaulted.
 */
enum class BeamstrahlungQTerm {
    /// @f$e^{q\sqrt u}@f$ (cepc240, fccee240, fccee365, ilc250)
    sqrt_u,
    /// @f$e^{q u^{3/2}}@f$ (ilc500)
    u_three_halves,
    /// @f$e^{q y^{-3/2}}@f$ (clic3000) -- in @f$y@f$, not in @f$u@f$
    y_minus_three_halves,
};

/// CIRCE parameters of one component's beamstrahlung factor.
struct BeamstrahlungParameters {
    double a = 0.;
    double p = 0.;
    double q = 0.;
    /// False for the components that carry no beamstrahlung on this beam --
    /// the delta(1-x_BS) pieces, where the beam simply did not radiate.
    bool has = false;
};

/**
 * Splits a dressed-lepton momentum fraction into its beamstrahlung and ISR
 * parts.
 *
 * A dressed-lepton density is a convolution of a beamstrahlung spectrum with
 * the ISR structure function, so the fraction in an event is a product,
 * @f$x = x_\mathrm{BS}\,x_\mathrm{ISR}@f$. A parton shower must be given
 * @f$x_\mathrm{ISR}@f$ alone: @f$x_\mathrm{BS}@f$ is a beam-energy loss, not
 * radiation off the beam lepton, and a shower handed the product invents
 * collinear radiation that beamstrahlung never produced.
 *
 * This is the C++ port of `eepdf_sample_bstrahl` (`bstrahl.f`): the
 * convolution integrand is built on a uniform grid in the sampling variable,
 * accumulated into a CDF and inverted. It runs once per *written* event, not
 * once per phase-space point, and the fractions do not enter the event weight,
 * so it is deliberately a plain host-side class rather than a compute-graph
 * function.
 *
 * The component is the one chosen by @ref EELuminosity: each carries its own
 * spectrum, and half of them carry none at all.
 */
class BeamstrahlungSampler {
public:
    /**
     * @param beam1        Parameters per component for the first beam.
     * @param beam2        The same for the second beam. The two differ:
     *                     which components radiate depends on the beam.
     * @param q_term       Collider-dependent factor; see @ref
     *                     BeamstrahlungQTerm.
     * @param alpha_em     @f$\alpha@f$ in the scheme the spectrum was fitted
     *                     in.
     * @param lepton_mass  @f$m@f$ of the beam lepton, in GeV.
     * @param grid_points  Points of the CDF grid; the default matches
     *                     `bstrahl.f`.
     */
    BeamstrahlungSampler(
        const std::vector<BeamstrahlungParameters>& beam1,
        const std::vector<BeamstrahlungParameters>& beam2,
        BeamstrahlungQTerm q_term,
        double alpha_em = 0.007562397,
        double lepton_mass = 0.511e-3,
        std::size_t grid_points = 128
    );

    /// True if any component of either beam radiates. False for a pure-ISR
    /// density, where there is nothing to split and no element to write.
    bool has_beamstrahlung() const;

    /**
     * @f$x_\mathrm{BS}@f$ of each beam, `(1, 1)` where the beam did not
     * radiate.
     *
     * @param x1,x2      Momentum fractions.
     * @param q2         Squared factorization scale.
     * @param component  Component index, from @ref EELuminosity.
     * @param u1,u2      Uniform deviates for the first beam: `u1` picks the
     *                   half of the integral, `u2` inverts the CDF in it.
     * @param u3,u4      The same for the second beam.
     */
    std::pair<double, double> sample(
        double x1,
        double x2,
        double q2,
        std::size_t component,
        double u1,
        double u2,
        double u3,
        double u4
    ) const;

    /// @f$x_\mathrm{BS}@f$ for a single beam; `sample` calls this twice.
    double sample_one(
        const BeamstrahlungParameters& pars,
        double x,
        double q2,
        double u1,
        double u2
    ) const;

private:
    double integrand(
        double r, int term, double x, const BeamstrahlungParameters& pars,
        double beta
    ) const;

    std::vector<BeamstrahlungParameters> _beam1, _beam2;
    BeamstrahlungQTerm _q_term;
    double _alpha_em, _lepton_mass;
    std::size_t _grid_points;
};

} // namespace madspace

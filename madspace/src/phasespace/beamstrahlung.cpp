#include "madspace/phasespace/beamstrahlung.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace madspace;

namespace {

// The t integral is split at t = ra exactly as vendor/eepdfgrid does: the
// substitution that removes the endpoint singularities leaves a smooth
// integrand in the sampling variable, which is what makes a uniform-grid CDF
// inversion accurate here.
constexpr double RA = 0.5;
constexpr double PA = -2. / 3.;
// above this x the split is numerically indistinguishable from 1
constexpr double XMAX = 1. - 1e-12;

/// Reduced ISR structure function; a copy of vendor/eepdfgrid/calcisr.f.
double isr_tilde(double x, double beta) {
    double res = 0.;
    if (x < 0.9999999) {
        res = -beta * (1. + x) / 2.;
        res -= beta * beta
            * ((1. + 3. * x * x) / (1. - x) * std::log(x)
               + 4. * (1. + x) * std::log1p(-x) + 5. + x)
            / 8.;
        res *= std::pow(1. - x, 1. - beta);
    }
    constexpr double pi = 3.14159265358979323846;
    res += (1. + 0.75 * beta + (27. - 8. * pi * pi) / 96. * beta * beta) * beta;
    return res;
}

double q_factor(BeamstrahlungQTerm term, double q, double u, double y) {
    switch (term) {
    case BeamstrahlungQTerm::sqrt_u:
        return std::exp(q * std::sqrt(u));
    case BeamstrahlungQTerm::u_three_halves:
        return std::exp(q * std::pow(u, 1.5));
    case BeamstrahlungQTerm::y_minus_three_halves:
        return std::exp(q * std::pow(y, -1.5));
    }
    return 0.;
}

} // namespace

BeamstrahlungSampler::BeamstrahlungSampler(
    const std::vector<BeamstrahlungParameters>& beam1,
    const std::vector<BeamstrahlungParameters>& beam2,
    BeamstrahlungQTerm q_term,
    double alpha_em,
    double lepton_mass,
    std::size_t grid_points
) :
    _beam1(beam1),
    _beam2(beam2),
    _q_term(q_term),
    _alpha_em(alpha_em),
    _lepton_mass(lepton_mass),
    _grid_points(grid_points) {
    if (beam1.size() != beam2.size()) {
        throw std::invalid_argument(
            "BeamstrahlungSampler: the two beams must have the same components"
        );
    }
    if (lepton_mass <= 0.) {
        throw std::invalid_argument(
            "BeamstrahlungSampler: lepton mass must be positive"
        );
    }
    if (grid_points < 2) {
        throw std::invalid_argument("BeamstrahlungSampler: need at least 2 grid points");
    }
}

bool BeamstrahlungSampler::has_beamstrahlung() const {
    auto radiates = [](const BeamstrahlungParameters& p) { return p.has; };
    return std::any_of(_beam1.begin(), _beam1.end(), radiates)
        || std::any_of(_beam2.begin(), _beam2.end(), radiates);
}

double BeamstrahlungSampler::integrand(
    double r, int term, double x, const BeamstrahlungParameters& pars, double beta
) const {
    double pb = beta - 1.;
    double rb = 1. - RA;
    double t = term == 1 ? RA * std::pow(r, 1. / (PA + 1.))
                         : 1. - rb * std::pow(r, 1. / (pb + 1.));
    double z = t * (1. - x) + x;
    if (z <= 0.) {
        return 0.;
    }
    double y = x / z;
    double u = std::max(1. - y, 0.);

    double g = std::exp(pars.p * u) * q_factor(_q_term, pars.q, u, y)
        * isr_tilde(z, beta) * std::pow(z, -1. / 3.);

    return term == 1 ? g * std::pow(1. - t, pb) * std::pow(RA, PA + 1.) / (PA + 1.)
                     : g * std::pow(t, PA) * std::pow(rb, pb + 1.) / (pb + 1.);
}

double BeamstrahlungSampler::sample_one(
    const BeamstrahlungParameters& pars, double x, double q2, double u1, double u2
) const {
    if (!pars.has || x >= XMAX || x <= 0. || q2 <= 0.) {
        return 1.;
    }
    constexpr double pi = 3.14159265358979323846;
    double beta =
        _alpha_em / pi * (std::log(q2 / (_lepton_mass * _lepton_mass)) - 1.);
    double pb = beta - 1.;
    double rb = 1. - RA;
    std::size_t n = _grid_points;
    double h = 1. / static_cast<double>(n);

    // Cumulative (trapezoidal) of both halves over the sampling variable r in
    // [0, 1]. The `a` prefactor is common to the two and drops out of the
    // normalization, so it is left out here.
    std::vector<double> cdf1(n + 1, 0.), cdf2(n + 1, 0.);
    double f0 = integrand(0., 1, x, pars, beta);
    for (std::size_t i = 1; i <= n; ++i) {
        double f1 = integrand(static_cast<double>(i) * h, 1, x, pars, beta);
        cdf1[i] = cdf1[i - 1] + 0.5 * h * (f0 + f1);
        f0 = f1;
    }
    f0 = integrand(0., 2, x, pars, beta);
    for (std::size_t i = 1; i <= n; ++i) {
        double f1 = integrand(static_cast<double>(i) * h, 2, x, pars, beta);
        cdf2[i] = cdf2[i - 1] + 0.5 * h * (f0 + f1);
        f0 = f1;
    }

    double tot = cdf1[n] + cdf2[n];
    if (tot <= 0.) {
        return 1.;
    }

    // pick the half, then invert its CDF by bisection plus a linear
    // interpolation inside the bin
    const std::vector<double>& cdf = (u1 * tot < cdf1[n]) ? cdf1 : cdf2;
    double target = u2 * cdf[n];
    std::size_t lo = 0, hi = n;
    while (hi - lo > 1) {
        std::size_t mid = (lo + hi) / 2;
        (cdf[mid] <= target ? lo : hi) = mid;
    }
    double r = static_cast<double>(lo) * h;
    if (cdf[hi] > cdf[lo]) {
        r += h * (target - cdf[lo]) / (cdf[hi] - cdf[lo]);
    }
    double t = (u1 * tot < cdf1[n]) ? RA * std::pow(r, 1. / (PA + 1.))
                                    : 1. - rb * std::pow(r, 1. / (pb + 1.));

    double z = t * (1. - x) + x;
    if (z <= x) {
        return 1.;
    }
    if (z >= 1.) {
        return x;
    }
    return x / z;
}

std::pair<double, double> BeamstrahlungSampler::sample(
    double x1,
    double x2,
    double q2,
    std::size_t component,
    double u1,
    double u2,
    double u3,
    double u4
) const {
    if (component >= _beam1.size()) {
        throw std::out_of_range("BeamstrahlungSampler: component out of range");
    }
    return {
        sample_one(_beam1[component], x1, q2, u1, u2),
        sample_one(_beam2[component], x2, q2, u3, u4)
    };
}

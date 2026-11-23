#pragma once

#include <array>
#include <vector>

#include "madspace/compgraphs/type.hpp"
#include "madspace/util.hpp"

namespace madspace {

/**
 * Effective Vector boson Approximation (EVA): the W, Z and photon content of
 * electron and muon beams, madevent's `pdlabel = eva`.
 *
 * The density of a vector boson of helicity @f$\lambda@f$ in a lepton with a
 * left-handed fraction @f$f_L@f$ is
 * @f$f_\lambda = f_L f^L_\lambda + (1-f_L) f^R_\lambda@f$ with the closed
 * forms of [1] at leading log (LLA), of [2] at full leading power (LP) and
 * next-to-leading power (NLP), evolved in @f$\mu_F^2@f$ or @f$p_T^2@f$. This is
 * a port of madevent's `ElectroweakFlux.f` and of the checks of
 * `ElectroweakFluxDriver.f`, with the same conventions:
 * - the densities vanish outside of @f$10^{-8} < x < 1 - 10^{-8}@f$, and for a
 *   W or Z below @f$x = M_V / E_\text{beam}@f$ unless `xcut` is zero;
 * - @f$\mu^2@f$ is raised to @f$M_V^2@f$ (a photon: @f$m_\ell^2@f$), times
 *   @f$1-x@f$ with @f$p_T@f$ evolution;
 * - the couplings follow from @f$M_W@f$, @f$M_Z@f$ and @f$e^2@f$ of the model
 *   (@f$s_W^2 = 1 - M_W^2/M_Z^2@f$); a photon is always at leading log;
 * - a W whose charge the beam lepton cannot emit (a W- from a positron, say)
 *   has no density, where madevent stops.
 *
 * The densities depend on the vector boson's helicity, so the cross section is
 * @f$\sum_{\lambda_1\lambda_2} f_{\lambda_1} f_{\lambda_2}
 * |M_{\lambda_1\lambda_2}|^2@f$ with the matrix element inside the sum. The
 * @ref DifferentialCrossSection takes the helicity picked by the matrix element
 * (with a probability proportional to its @f$|M_h|^2@f$) and multiplies the
 * helicity-averaged @f$|M|^2@f$ by @f$n_1 f_{\lambda_1} n_2 f_{\lambda_2}@f$,
 * @f$n_i@f$ the number of helicity states of leg @f$i@f$: an unbiased estimate
 * of the sum, as madevent's `nhel = 1`. The helicity sum
 * @f$\sum_\lambda f_\lambda@f$ serves as the beam density where no helicity is
 * known (flavour sampling, @ref PartonDensity).
 *
 * **References**
 * - [1] R. Ruiz, A. Costantini, F. Maltoni, O. Mattelaer, "The Effective Vector
 *   Boson Approximation in high-energy muon collisions",
 *   https://arxiv.org/abs/2111.02442
 * - [2] I. Bigaran, R. Ruiz, https://arxiv.org/abs/2502.07878
 */
class EvaFlux {
public:
    /// Number of numbers of a parameter row, see @ref parameters.
    static constexpr std::size_t parameter_count = 11;
    /// The helicity argument asking for the sum over the vector's helicities.
    static constexpr int helicity_sum = 2;

    /**
     * @param beam_leptons     PDG ids of the two beam leptons (+-11, +-13).
     * @param beam_energies    Energies of the two beams, in GeV.
     * @param w_mass           @f$M_W@f$, in GeV.
     * @param z_mass           @f$M_Z@f$, in GeV.
     * @param e2               @f$e^2 = 4 \pi \alpha@f$ of the model.
     * @param left_fractions   Left-handed fraction @f$f_L@f$ of each beam lepton,
     *                         0.5 unpolarised (madevent: 0.5 - polbeam / 200).
     * @param order            0: LLA, 1: full LP, 2: NLP (madevent's evaorder).
     * @param pt_evolution     Evolution in @f$p_T^2@f$ rather than @f$\mu^2@f$
     *                         (madevent's ievo_eva = 1).
     * @param xcut             Whether a W or Z needs @f$x > M_V/E_\text{beam}@f$
     *                         (madevent's eva_xcut).
     */
    EvaFlux(
        const std::array<int, 2>& beam_leptons,
        const std::array<double, 2>& beam_energies,
        double w_mass,
        double z_mass,
        double e2,
        const std::array<double, 2>& left_fractions = {0.5, 0.5},
        int order = 0,
        bool pt_evolution = false,
        bool xcut = true
    );

    /// Whether a vector boson @p vector_pid (+-24, 23, 22 or 7) of beam
    /// @p beam (0 or 1) has a density (it is not charge violating).
    bool has_density(int vector_pid, std::size_t beam) const;
    /// The parameter row of vector boson @p vector_pid from beam @p beam, as
    /// the `eva_density` kernels read it: kind (0 none, 1 W/Z, 2 photon),
    /// @f$g^2@f$, @f$g_L^2@f$, @f$g_R^2@f$, @f$f_L@f$, @f$M_V^2@f$,
    /// @f$\mu^2_\text{min}@f$, @f$x_\text{min}@f$, @f$E_\text{beam}@f$, order,
    /// evolution. A vector boson that is not one of the above throws.
    std::vector<double> parameters(int vector_pid, std::size_t beam) const;
    /**
     * The density @f$f_\lambda(x, \mu)@f$ (not @f$x f@f$), on the host: the
     * reference the compute-graph kernels are checked against.
     *
     * @param vector_pid  PDG id of the vector boson.
     * @param beam        The beam (0 or 1).
     * @param helicity    -1, 0, 1, or @ref helicity_sum.
     * @param x           Momentum fraction.
     * @param q           Factorisation scale @f$\mu@f$, in GeV.
     */
    double density(
        int vector_pid, std::size_t beam, int helicity, double x, double q
    ) const;

    /// PDG ids of the beam leptons.
    const std::array<int, 2>& beam_leptons() const { return _beam_leptons; }
    /// Beam energies, in GeV.
    const std::array<double, 2>& beam_energies() const { return _beam_energies; }
    /// Left-handed fractions of the beam leptons.
    const std::array<double, 2>& left_fractions() const { return _left_fractions; }
    /// 0: LLA, 1: LP, 2: NLP.
    int order() const { return _order; }
    /// Whether the evolution is in @f$p_T^2@f$.
    bool pt_evolution() const { return _pt_evolution; }
    /// Whether W and Z need @f$x > M_V/E_\text{beam}@f$.
    bool xcut() const { return _xcut; }
    /// @f$M_W@f$, @f$M_Z@f$ and @f$e^2@f$.
    double w_mass() const { return _w_mass; }
    double z_mass() const { return _z_mass; }
    double e2() const { return _e2; }

private:
    std::array<int, 2> _beam_leptons;
    std::array<double, 2> _beam_energies;
    double _w_mass, _z_mass, _e2;
    std::array<double, 2> _left_fractions;
    int _order;
    bool _pt_evolution;
    bool _xcut;
};

/**
 * The incoming-leg tables a @ref DifferentialCrossSection needs to apply the
 * EVA densities after its matrix element picked a helicity.
 *
 * Built once per differential cross section from the beams (@ref EvaFlux),
 * the flavour options and, per flavour, the helicity table of its matrix
 * element (rows of the incoming legs' helicities, in the order of the matrix
 * element's helicity index). Leg @f$i@f$ comes from beam @f$i@f$, or from the
 * other beam when the event is mirrored.
 */
struct EvaLegTables {
    /// Per leg: per flavour, its first row in leg_helicities.
    std::array<std::vector<me_int_t>, 2> row_offsets;
    /// Per leg: per flavour, its number of rows.
    std::array<std::vector<me_int_t>, 2> row_counts;
    /// Per leg: per flavour and row, the leg's helicity.
    std::array<std::vector<me_int_t>, 2> leg_helicities;
    /// Per leg: per flavour and orientation (2 f + mirror), the row of params.
    std::array<std::vector<me_int_t>, 2> configs;
    /// Per leg: per flavour, the number of helicity states of the leg.
    std::array<std::vector<double>, 2> helicity_counts;
    /// The parameter rows.
    nested_vector2<double> params;

    /**
     * @param flux          The beams.
     * @param pid_options   Per flavour, the PDG ids of the particles, the two
     *                      incoming ones first.
     * @param helicities    Per flavour, the helicity table of its matrix
     *                      element: rows of the helicities of the particles,
     *                      the two incoming ones first.
     */
    EvaLegTables(
        const EvaFlux& flux,
        const nested_vector2<me_int_t>& pid_options,
        const nested_vector3<me_int_t>& helicities
    );
};

} // namespace madspace

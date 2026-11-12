#pragma once

#include <array>
#include <optional>

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/pdf.hpp"

namespace madspace {

/**
 * The renormalization and factorization scales of one kinematics of the FKS
 * integrand, those of setscales.f of MadFKS [2] with the parameters of its run
 * card. The reference scale of scale_global_reference is, for
 * `dynamical_scale_choice`
 *
 * @f[
 *   1:\ \sum_i \frac{E_i\,p_{\mathrm{T},i}}{\sqrt{p_{\mathrm{T},i}^2 + p_{z,i}^2}},
 *   \qquad
 *   2:\ \sum_i \sqrt{(E_i + p_{z,i})(E_i - p_{z,i})},
 *   \qquad
 *   3 \text{ or } -1:\ \frac{1}{2} \sum_i \sqrt{(E_i + p_{z,i})(E_i - p_{z,i})},
 * @f]
 *
 * over the outgoing particles, the first in the partonic center-of-mass frame
 * (a particle without transverse momentum adds nothing), or `mur_ref_fixed`
 * for -2. For a decay it is the mass @f$M@f$ of the decaying particle for
 * every choice, as setscales.f takes it for @f$\mu_R@f$ and the Ellis-Sexton
 * scale. It is raised to @ref min_scale, as minscaleR and minscaleF of
 * setscales.f. A fixed scale (`fixed_ren_scale`, `fixed_fac_scale`) replaces
 * it, and each scale, fixed or not, is multiplied by its factor:
 *
 * @f[
 *   \mu_R = \text{mur\_over\_ref}\ \mu_\mathrm{ref},
 *   \qquad
 *   \mu_{F,k} = \text{muf}k\text{\_over\_ref}\ \mu_\mathrm{ref}.
 * @f]
 *
 * The integrand takes the scales of the event, of the Born (for the soft and
 * soft-collinear counterterms and the n-body part, as MadFKS at the soft
 * counter-event) and of the collinear counter-event. The Ellis-Sexton scale
 * of MadFKS has no counterpart: the integrated counterterms are written at
 * @f$\mu_R@f$ and @f$\mu_F@f$, and the virtual comes at @f$\mu_R@f$. The
 * MadFKS run card has one factorization scale (`muf_ref_fixed`,
 * `muf_over_ref`), and sigint of MadFKS takes that of the first beam for both.
 * A decay has no densities, and its @f$\mu_F@f$ enters nothing. MadFKS takes
 * @f$H_T/2@f$ without the first outgoing particle there (scale_global_reference
 * starts at the third particle), madspace the mass.
 *
 * `batch` is the leading batch dimension and `n` the number of particles.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, n, 4)` – the kinematics, the incoming
 *   first, in any frame boosted along the beams.
 *
 * **Returns**
 * - `ren_scale` – `float`, shape `(batch,)` – renormalization scale.
 * - `fact_scale1`, `fact_scale2` – `float`, shape `(batch,)` – factorization
 *   scales of the two beams.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 * - [2] R. Frederix, S. Frixione, F. Maltoni, T. Stelzer, "Automation of
 *   next-to-leading order computations in QCD: the FKS subtraction",
 *   https://arxiv.org/abs/0908.4272
 */
class FKSScale : public FunctionGenerator {
public:
    /**
     * @param particle_count          Number of particles, incoming included.
     * @param dynamical_scale_choice  -1 or 3 for @f$H_T/2@f$, 2 for @f$H_T@f$,
     *                                1 for @f$E_T@f$, -2 for `mur_ref_fixed`.
     * @param fixed_ren_scale         Whether @f$\mu_R@f$ is fixed.
     * @param fixed_fac_scale         Whether the @f$\mu_F@f$ are fixed.
     * @param mur_ref_fixed           The fixed @f$\mu_R@f$.
     * @param muf1_ref_fixed          The fixed @f$\mu_F@f$ of the first beam.
     * @param muf2_ref_fixed          The fixed @f$\mu_F@f$ of the second beam.
     * @param mur_over_ref            Factor on @f$\mu_R@f$.
     * @param muf1_over_ref           Factor on @f$\mu_F@f$ of the first beam.
     * @param muf2_over_ref           Factor on @f$\mu_F@f$ of the second beam.
     * @param incoming                Number of incoming particles, two or one
     *                                for a decay.
     * @throws std::invalid_argument without an outgoing particle, for another
     *         number of incoming particles or another dynamical_scale_choice
     *         (0 and 10, the scale of the user, are not part of it).
     */
    FKSScale(
        std::size_t particle_count,
        int dynamical_scale_choice = -1,
        bool fixed_ren_scale = false,
        bool fixed_fac_scale = false,
        double mur_ref_fixed = 91.118,
        double muf1_ref_fixed = 91.118,
        double muf2_ref_fixed = 91.118,
        double mur_over_ref = 1.,
        double muf1_over_ref = 1.,
        double muf2_over_ref = 1.,
        std::size_t incoming = 2
    );
    /// The floor on the dynamical reference scale, minscaleR and minscaleF of
    /// setscales.f, in GeV.
    static constexpr double min_scale = 2.;
    /// Number of particles, incoming included.
    std::size_t particle_count() const { return _particle_count; }
    /// Number of incoming particles, one for a decay.
    std::size_t incoming() const { return _incoming; }
    /// The options of the kernel: dynamical_scale_choice, fixed_ren_scale,
    /// fixed_fac_scale, mur_ref_fixed, muf1_ref_fixed, muf2_ref_fixed,
    /// mur_over_ref, muf1_over_ref, muf2_over_ref and @ref min_scale.
    const std::vector<double>& options() const { return _options; }
    /// The same scales for kinematics of another number of particles.
    FKSScale with_particle_count(std::size_t particle_count) const;

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::size_t _particle_count;
    std::size_t _incoming;
    std::vector<double> _options;
};

/**
 * The parton luminosity of one kinematics of the FKS integrand, as the
 * parton_lum_*.f files of MadFKS [2]:
 *
 * @f[
 *   c_r\, f_1^{(r)}(x_1, \mu_{F,1})\, f_2^{(r)}(x_2, \mu_{F,2})
 * @f]
 *
 * for the flavor @f$r@f$ of the event among those of its FKS configuration,
 * for event generation, or the sum over them for fixed order, with the
 * densities @f$f = (x f)/x@f$ of a @ref PdfGrid and a factor @f$c_r@f$ (one
 * by default), without the conversion to pb (which @ref FKSWeights applies).
 * A flavor the grid does not have (a lepton from a proton) has no density, as
 * pdg2pdf.f, and a beam without densities (a lepton, the decaying particle)
 * counts one. The scales are kept within the grid, where the densities freeze,
 * as the built-in PDFs of MadFKS. The flavors and factors may differ per
 * configuration, so that a batch can mix configurations and flavors.
 *
 * The real emission and its counter-events take the flavors of the real
 * matrix element of their configuration. With a flavor index, the n-body
 * part takes the underlying Born of the real flavor of the event (see
 * fks_luminosities of madspace.fks), with @f$c_r@f$ one over the number of
 * real flavors of the configuration with that Born, so that the sum over the
 * real flavors gives the sum over the flavors of the Born.
 *
 * `batch` is the leading batch dimension.
 *
 * **Arguments**
 * - `x1`, `x2` – `float`, shape `(batch,)` – the momentum fractions of the
 *   kinematics.
 * - `fact_scale1`, `fact_scale2` – `float`, shape `(batch,)` – the
 *   factorization scales of the two beams at the kinematics.
 * - `flavor_index` – `int`, shape `(batch,)` – the flavor of the event among
 *   those of its configuration (none with `sum_flavors`). An index beyond
 *   them gives zero.
 * - `fks_index` – `int`, shape `(batch,)` – the FKS configuration of the
 *   event.
 *
 * **Returns**
 * - `luminosity` – `float`, shape `(batch,)` – the luminosity.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 * - [2] R. Frederix, S. Frixione, F. Maltoni, T. Stelzer, "Automation of
 *   next-to-leading order computations in QCD: the FKS subtraction",
 *   https://arxiv.org/abs/0908.4272
 */
class FKSLuminosity : public FunctionGenerator {
public:
    /**
     * @param flavors      For each FKS configuration, the PDG codes of the two
     *                     incoming partons of each of its flavors (for a
     *                     decay, the decaying particle and 0).
     * @param pdf_grid     The densities of the beams, none for beams without.
     * @param pdf_grid2    The densities of the second beam where they differ.
     * @param pdf2_prefix  Prefix of the global names of @p pdf_grid2.
     * @param densities    Which beams take densities from the grids, the
     *                     others count one.
     * @param sum_flavors  Whether the luminosity sums over the flavors of the
     *                     configuration, without a flavor index.
     * @param factors      For each configuration, the factor of each flavor,
     *                     one if empty.
     * @throws std::invalid_argument without configurations, with a
     *         configuration without flavors or a flavor that is not a pair,
     *         with factors that do not match the flavors, with @p pdf_grid2
     *         but no @p pdf_grid, or with a grid and no beam with densities.
     */
    FKSLuminosity(
        const nested_vector3<me_int_t>& flavors,
        const std::optional<PdfGrid>& pdf_grid = std::nullopt,
        const std::optional<PdfGrid>& pdf_grid2 = std::nullopt,
        const std::string& pdf2_prefix = "beam2",
        const std::array<bool, 2>& densities = {true, true},
        bool sum_flavors = false,
        const nested_vector2<double>& factors = {}
    );
    /// Number of FKS configurations.
    std::size_t configuration_count() const { return _positions.size(); }
    /// Whether a beam has densities.
    bool has_pdf() const { return _pdfs.at(0).has_value() || _pdfs.at(1).has_value(); }
    /// Whether the luminosity sums over the flavors.
    bool sum_flavors() const { return _sum_flavors; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    bool _sum_flavors;
    nested_vector2<me_int_t> _positions;
    nested_vector2<double> _factors;
    std::array<std::optional<PartonDensity>, 2> _pdfs;
    std::array<double, 2> _q_min, _q_max;
};

} // namespace madspace

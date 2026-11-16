#pragma once

#include <array>
#include <optional>

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/pdf.hpp"

namespace madspace {

/**
 * The parton luminosity of one kinematics of an NLO integrand, as the
 * parton_lum_*.f files of MadFKS [2]:
 *
 * @f[
 *   c_r\, f_1^{(r)}(x_1, \mu_{F,1})\, f_2^{(r)}(x_2, \mu_{F,2})
 * @f]
 *
 * for the flavor @f$r@f$ of the event among those of its group, for event
 * generation, or the sum over them for fixed order, with the densities
 * @f$f = (x f)/x@f$ of a @ref PdfGrid and a factor @f$c_r@f$ (one by default),
 * without the conversion to pb. A group is a list of flavors, the pairs of
 * PDG codes of the two incoming partons: the flavors of the real matrix
 * element of an FKS configuration, those of the Born or the real process of
 * a Catani-Seymour subprocess, or any other set with factors, such as the
 * partons whose densities a convolution of the K and P operators takes.
 * A flavor the grid does not have (a lepton from a proton) has no density, as
 * pdg2pdf.f, and a beam without densities (a lepton, the decaying particle)
 * counts one. Beams of dressed leptons take in place of the two densities the
 * luminosity @f$\sum_n f^{(1)}_n(x_1, \mu_{F,1}) f^{(2)}_n(x_2, \mu_{F,2})@f$
 * of an @ref EELuminosity (ee_comp_prod of pdg2pdf.f) for the flavor of the
 * two beam leptons and none for the others (compute_eepdf has no photon). A
 * beam of elastic photons has the @ref ElasticPhotonDensity of its photon at
 * the factorization scale of the beam (epa_proton of pdg2pdf.f) and no other
 * parton. The scales are kept within the grid, where the densities freeze, as
 * the built-in PDFs of MadFKS. The flavors and factors may differ per group,
 * so that a batch can mix groups and flavors.
 *
 * In the FKS integrand the real emission and its counter-events take the
 * flavors of the real matrix element of their configuration. With a flavor
 * index, the n-body part takes the underlying Born of the real flavor of the
 * event (see fks_luminosities of madspace.nlo.fks), with @f$c_r@f$ one over
 * the number of real flavors of the configuration with that Born, so that the
 * sum over the real flavors gives the sum over the flavors of the Born.
 *
 * With `all_groups` the luminosity sums over the flavors of every group at
 * once, from one evaluation of the densities, for the convolutions that need
 * several sums at the same momentum fractions.
 *
 * `batch` is the leading batch dimension and `g` the number of groups.
 *
 * **Arguments**
 * - `x1`, `x2` – `float`, shape `(batch,)` – the momentum fractions of the
 *   kinematics.
 * - `one_minus_x1`, `one_minus_x2` – `float`, shape `(batch,)` – 1 - x1 and
 *   1 - x2, with dressed leptons only.
 * - `fact_scale1`, `fact_scale2` – `float`, shape `(batch,)` – the
 *   factorization scales of the two beams at the kinematics.
 * - `flavor_index` – `int`, shape `(batch,)` – the flavor of the event among
 *   those of its group (none with `sum_flavors`). An index beyond them gives
 *   zero.
 * - `group_index` – `int`, shape `(batch,)` – the group of the event (none
 *   with `all_groups`).
 *
 * **Returns**
 * - `luminosity` – `float`, shape `(batch,)`, or `(batch, g)` with
 *   `all_groups` – the luminosity.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 * - [2] R. Frederix, S. Frixione, F. Maltoni, T. Stelzer, "Automation of
 *   next-to-leading order computations in QCD: the FKS subtraction",
 *   https://arxiv.org/abs/0908.4272
 */
class NLOLuminosity : public FunctionGenerator {
public:
    /**
     * @param flavors      For each group, the PDG codes of the two incoming
     *                     partons of each of its flavors (for a decay, the
     *                     decaying particle and 0).
     * @param pdf_grid     The densities of the beams, none for beams without.
     * @param pdf_grid2    The densities of the second beam where they differ.
     * @param pdf2_prefix  Prefix of the global names of @p pdf_grid2.
     * @param densities    Which beams take densities from the grids, the
     *                     others count one.
     * @param sum_flavors  Whether the luminosity sums over the flavors of the
     *                     group, without a flavor index.
     * @param factors      For each group, the factor of each flavor, one if
     *                     empty.
     * @param lepton_luminosity The luminosity of beams of dressed leptons, with
     *                     the scales of the beams (`beam_scales`), with the
     *                     full densities where the Jacobian of the
     *                     @ref LeptonLuminosity takes them (FKS) and the
     *                     reduced ones where it does not (Catani-Seymour).
     *                     Not with a grid.
     * @param lepton_pdgs  The PDG codes of the leptons of the two beams, the
     *                     flavor of @p lepton_luminosity.
     * @param pdf_prefix   Prefix of the global names of @p pdf_grid, for the
     *                     members of a set of PDFs beside its central one.
     * @param photon_beams Which beams are elastic photons, not with
     *                     densities of the grids or dressed leptons.
     * @param all_groups   Whether the luminosity is that of every group, as
     *                     the last dimension of the output, without a group
     *                     index. Only with @p sum_flavors.
     * @throws std::invalid_argument without groups, with a group without
     *         flavors or a flavor that is not a pair, with factors that do
     *         not match the flavors, with @p pdf_grid2 but no @p pdf_grid,
     *         with a grid and no beam with densities, with
     *         @p lepton_luminosity and a grid or without the scales of the
     *         beams, with a beam of elastic photons that takes the densities
     *         of the grids or with dressed leptons, or with @p all_groups
     *         and a flavor index.
     */
    NLOLuminosity(
        const nested_vector3<me_int_t>& flavors,
        const std::optional<PdfGrid>& pdf_grid = std::nullopt,
        const std::optional<PdfGrid>& pdf_grid2 = std::nullopt,
        const std::string& pdf2_prefix = "beam2",
        const std::array<bool, 2>& densities = {true, true},
        bool sum_flavors = false,
        const nested_vector2<double>& factors = {},
        const std::optional<EELuminosity>& lepton_luminosity = std::nullopt,
        const std::array<int, 2>& lepton_pdgs = {0, 0},
        const std::string& pdf_prefix = "",
        const std::array<bool, 2>& photon_beams = {false, false},
        bool all_groups = false
    );
    /// Number of groups of flavors.
    std::size_t group_count() const { return _positions.size(); }
    /// Whether a beam has densities of a grid.
    bool has_pdf() const { return _pdfs.at(0).has_value() || _pdfs.at(1).has_value(); }
    /// Whether the luminosity sums over the flavors.
    bool sum_flavors() const { return _sum_flavors; }
    /// Whether the luminosity is that of every group at once.
    bool all_groups() const { return _all_groups; }
    /// Whether the beams are dressed leptons.
    bool lepton() const { return _lepton_luminosity.has_value(); }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    bool _sum_flavors, _all_groups;
    nested_vector2<me_int_t> _positions;
    nested_vector2<double> _factors;
    std::array<std::optional<PartonDensity>, 2> _pdfs;
    std::array<double, 2> _q_min, _q_max;
    std::optional<EELuminosity> _lepton_luminosity;
    std::array<bool, 2> _photon_beams;
};

} // namespace madspace

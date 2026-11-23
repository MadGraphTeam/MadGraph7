#pragma once

#include "madspace/phasespace/eva.hpp"
#include "madspace/phasespace/matrix_element.hpp"
#include "madspace/phasespace/pdf.hpp"
#include "madspace/phasespace/scale.hpp"

namespace madspace {

/**
 * The fully differential cross section @f$f(x)@f$.
 *
 * Assembles the integrand of the phase-space integral: a @ref MatrixElement
 * evaluation, with @f$\alpha_s@f$ from a @ref RunningCoupling, multiplied by
 * the flux factor and the @ref PartonDensity values for the two beams, all at
 * the @ref EnergyScale of the event [1]. The `Cached*` variants of the PDF and
 * scale arguments reuse values already computed elsewhere in the graph.
 *
 * With @p decay the generator instead describes the decay of a single incoming
 * particle. The differential rate is then @f$|M|^2 / (2 M)@f$ with
 * @f$M@f$ = @p cm_energy, in GeV, rather than a hadronic cross section in pb.
 * There are no beams, hence no PDFs; declared `x1`, `x2` inputs are ignored.
 *
 * With @p eva the beams are effective vector bosons (@ref EvaFlux), whose
 * densities depend on their helicity. The cached `pdf1`, `pdf2` (helicity
 * sums) are then replaced, after the matrix element picked the helicity row
 * of the event with a probability proportional to its @f$|M_h|^2@f$, by
 * @f$x_i n_i f_{\lambda_i}(x_i, \mu_i)@f$ with @f$\lambda_i@f$ the
 * helicity of incoming leg @f$i@f$ in that row of @p eva_helicities and
 * @f$n_i@f$ its number of helicity states: the helicity-averaged
 * @f$|M|^2@f$ times these is an unbiased estimate of
 * @f$\sum_h f_{\lambda_1(h)} f_{\lambda_2(h)} |M_h|^2@f$. Leg @f$i@f$
 * comes from beam @f$i@f$, or from the other one for `mirror_index` 1.
 *
 * With massive @p incoming_masses (those of the @ref PhaseSpaceMapping, which
 * puts the incoming particles on their mass shells) the flux is
 * @f$1 / (2 \sqrt{\lambda(\hat s, m_1^2, m_2^2)})@f$ with
 * @f$\hat s = x_1 x_2 s@f$ instead of @f$1 / (2 \hat s)@f$, as in madevent,
 * and zero at and below the threshold @f$(m_1 + m_2)^2@f$.
 *
 * `batch` is the leading batch dimension.
 *
 * **Arguments**
 * - the arguments of @p matrix_element except `alpha_s`, plus
 * - `x1`, `x2` – `float`, shape `(batch,)` – parton momentum fractions (when
 *   @p input_momentum_fraction is true).
 * - `pdf_id` – `int`, shape `(batch,)` – selected flavour combination.
 * - `pdf1`, `pdf2` – `float`, shape `(batch,)` – cached PDF values (with a
 *   `CachedPdf` argument).
 * - `alpha_s` – `float`, shape `(batch,)` – cached coupling (with a
 *   `CachedScale` argument).
 * - `fact_scale1`, `fact_scale2` – `float`, shape `(batch,)` – factorisation
 *   scales of the two legs (with @p eva).
 * - `mirror_index` – `int`, shape `(batch,)` – 1 for an event whose legs come
 *   from the other beams (with @p eva).
 *
 * **Returns**
 * - the return values of @p matrix_element, scaled to the differential cross
 *   section.
 * - `raw_matrix_element` – `float`, shape `(batch,)` – the squared matrix
 *   element without PDFs and flux (with @p return_matrix_element).
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 3.2)
 */
class DifferentialCrossSection : public FunctionGenerator {
public:
    /// Argument tag: take the PDF value from a value already in the graph.
    struct CachedPdf {};
    /// Argument tag: take the scale / coupling from a value already in the graph.
    struct CachedScale {};

    /**
     * @param matrix_element          The squared matrix element.
     * @param cm_energy                Total collision energy (the mass of the
     *                                 decaying particle with @p decay).
     * @param running_coupling         Optional @f$\alpha_s@f$ source.
     * @param energy_scale             An @ref EnergyScale, a `CachedScale` tag,
     *                                 or nothing.
     * @param pid_options              Allowed flavour combinations of the two
     *                                 initial partons.
     * @param pdf1                     A @ref PdfGrid, a `CachedPdf` tag, or
     *                                 nothing, for the first beam.
     * @param pdf2                     Likewise for the second beam.
     * @param input_momentum_fraction  If true, `x1` / `x2` are taken as inputs
     *                                 rather than reconstructed.
     * @param decay                    If true, describe the decay of a single
     *                                 incoming particle instead of a collision.
     * @param return_matrix_element    If true, also return the squared matrix
     *                                 element itself as `raw_matrix_element`.
     * @param eva                      Effective-vector-boson beams; needs
     *                                 `CachedPdf` on both beams, a
     *                                 `CachedScale`, @p pid_options and a
     *                                 matrix element returning its helicity.
     * @param eva_helicities           With @p eva, per flavour option, the
     *                                 helicity table of the matrix element
     *                                 (rows of the particles' helicities,
     *                                 indexed by its helicity index).
     * @param incoming_masses          Masses of the two incoming particles of a
     *                                 collision, for the flux, empty or zero
     *                                 for massless ones. Needs
     *                                 @p input_momentum_fraction.
     */
    DifferentialCrossSection(
        const MatrixElement& matrix_element,
        double cm_energy,
        const std::optional<RunningCoupling>& running_coupling,
        const std::variant<std::monostate, EnergyScale, CachedScale>& energy_scale =
            std::monostate{},
        const nested_vector2<me_int_t>& pid_options = {},
        const std::variant<std::monostate, PdfGrid, CachedPdf>& pdf1 = std::monostate{},
        const std::variant<std::monostate, PdfGrid, CachedPdf>& pdf2 = std::monostate{},
        bool input_momentum_fraction = true,
        bool decay = false,
        bool return_matrix_element = false,
        const std::optional<EvaFlux>& eva = std::nullopt,
        const nested_vector3<me_int_t>& eva_helicities = {},
        const std::vector<double>& incoming_masses = {}
    );

    /// The allowed initial-parton flavour combinations.
    const nested_vector2<me_int_t>& pid_options() const { return _pid_options; }
    /// Whether beam @p pdf_index (0 or 1) has a PDF attached.
    bool has_pdf(std::size_t pdf_index) const { return _has_pdf.at(pdf_index); }
    /// The matrix element.
    const MatrixElement& matrix_element() const { return _matrix_element; }
    /// The strong-coupling source, if any.
    const std::optional<RunningCoupling>& running_coupling() const {
        return _running_coupling;
    }
    /// Whether the squared matrix element is returned as `raw_matrix_element`.
    bool returns_matrix_element() const { return _return_matrix_element; }
    /// Whether the beams are effective vector bosons (then the arguments end
    /// with `fact_scale1`, `fact_scale2` and `mirror_index`).
    bool eva() const { return _eva_tables.has_value(); }
    /// Masses of the incoming particles the flux is computed with (absolute
    /// values): two for a collision with a massive one, empty otherwise.
    const std::vector<double>& incoming_masses() const { return _incoming_masses; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    nested_vector2<me_int_t> _pid_options;
    MatrixElement _matrix_element;
    std::array<std::optional<PartonDensity>, 2> _pdfs;
    std::array<bool, 2> _has_pdf;
    std::optional<RunningCoupling> _running_coupling;
    double _e_cm;
    std::variant<std::monostate, EnergyScale, CachedScale> _energy_scale;
    bool _input_momentum_fraction;
    bool _decay;
    bool _return_matrix_element;
    std::optional<EvaLegTables> _eva_tables;
    std::vector<double> _incoming_masses;
};

} // namespace madspace

#pragma once

#include "madspace/driver/context.hpp"
#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/custom_function.hpp"

namespace madspace {

/**
 * A parton-density grid in the LHAPDF format, loaded from a file.
 *
 * `madspace` carries a built-in PDF interpolator compatible with the LHAPDF
 * grid format so it needs no external PDF library [1, 2]. This struct holds the
 * loaded grid; @ref PartonDensity does the interpolation. `initialize_globals`
 * uploads the grid to a @ref Context.
 */
struct PdfGrid {
    /// Momentum-fraction grid knots.
    std::vector<double> x;
    /// `log(x)` grid knots.
    std::vector<double> logx;
    /// Scale grid knots.
    std::vector<double> q;
    /// `log(q^2)` grid knots.
    std::vector<double> logq2;
    /// PDG ids of the flavours in the grid.
    std::vector<int> pids;
    /// Grid values, indexed by `(q, x)` grid point (`q_index * x.size() + x_index`),
    /// then by flavour.
    std::vector<std::vector<double>> values;
    /// Number of `q` intervals in each sub-grid region.
    std::vector<std::size_t> region_sizes;

    /// Load an LHAPDF-format grid.
    /// @param file  Path to the grid file.
    PdfGrid(const std::string& file);
    /// Number of `(x, q)` grid points, including padding, as stored in the
    /// coefficient tensor.
    std::size_t grid_point_count() const;
    /// Number of scale knots, counting the knots shared by two regions once.
    std::size_t q_count() const;
    /// Fill @p tensor with the interpolation coefficients.
    void initialize_coefficients(Tensor tensor) const;
    /// Fill @p tensor with the `log(x)` knots.
    void initialize_logx(Tensor tensor) const;
    /// Fill @p tensor with the `log(q^2)` knots.
    void initialize_logq2(Tensor tensor) const;
    /// Shape of the coefficient tensor, optionally with a leading batch axis.
    std::vector<std::size_t> coefficients_shape(bool batch_dim = false) const;
    /// Shape of the `log(x)` tensor.
    std::vector<std::size_t> logx_shape(bool batch_dim = false) const;
    /// Shape of the `log(q^2)` tensor.
    std::vector<std::size_t> logq2_shape(bool batch_dim = false) const;
    /// Upload the grid to @p context as compute-graph globals under @p prefix.
    void initialize_globals(ContextPtr context, const std::string& prefix = "") const;
};

/**
 * Parton-density interpolation as a compute-graph function.
 *
 * Interpolates a @ref PdfGrid at the sampled @f$(x, q)@f$, returning
 * @f$x f(x, q)@f$ for the requested flavours; the numerical output agrees with
 * LHAPDF to double precision [1] (Sec. 3.2.3 of [2]). With a @ref CustomPdf,
 * the user-supplied library is called for each requested flavour instead.
 *
 * `batch` is the leading batch dimension.
 *
 * **Arguments**
 * - `x` – `float`, shape `(batch,)` – momentum fraction.
 * - `q` – `float`, shape `(batch,)` – factorization scale.
 * - `flavor_index` – `int`, shape `(batch,)` – index into @p pids selecting
 *   the flavour to return. Present only when @p dynamic_pid is true.
 *
 * **Returns**
 * - `pdf` – `float`, shape `(batch,)` when @p dynamic_pid is true, otherwise
 *   `(batch, len(pids))` – the interpolated `x f(x, q)`.
 *
 * **References**
 * - [1] A. Buckley et al., "LHAPDF6: parton density access in the LHC
 *   precision era", https://arxiv.org/abs/1412.7420
 * - [2] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 3.2.3)
 */
class PartonDensity : public FunctionGenerator {
public:
    /**
     * @param grid         The parton-density grid to interpolate.
     * @param pids          PDG ids of the flavours to return.
     * @param dynamic_pid   If true, the single flavour of @p pids selected per
     *                      event by `flavor_index` is returned.
     * @param prefix        Prefix for the grid global names.
     */
    PartonDensity(
        const PdfGrid& grid,
        const std::vector<int>& pids,
        bool dynamic_pid = false,
        const std::string& prefix = ""
    );
    /**
     * @param custom_pdf    The user-supplied parton density.
     * @param pids          PDG ids of the flavours to return.
     * @param beam_index    Index of the beam (0 or 1), passed to the library.
     * @param dynamic_pid   If true, the single flavour of @p pids selected per
     *                      event by `flavor_index` is returned.
     */
    PartonDensity(
        const CustomPdf& custom_pdf,
        const std::vector<int>& pids,
        int beam_index,
        bool dynamic_pid = false
    );

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<me_int_t> _pid_indices;
    std::vector<me_int_t> _pids;
    std::optional<CustomPdf> _custom_pdf;
    me_int_t _beam_index = 0;
    bool _dynamic_pid;
    std::string _prefix;
    std::vector<std::size_t> _logx_shape;
    std::vector<std::size_t> _logq2_shape;
    std::vector<std::size_t> _coeffs_shape;
};

/**
 * Dressed-lepton density: an interpolated grid times its analytic reduction.
 *
 * A dressed-lepton density is singular as @f$x \to 1@f$, so it is tabulated in
 * reduced form and the endpoint is restored analytically:
 *
 * @f[
 *   f_n(x, Q) = \bar\phi_n(x, Q) \, (1-x)^{-\kappa_n(Q) + p},
 *   \qquad
 *   \kappa_n(Q) = -\beta(Q) + \kappa^{(0)}_n,
 *   \qquad
 *   \beta(Q) = \frac{\alpha}{\pi}\Big(\log\frac{Q^2}{m^2} - 1\Big),
 * @f]
 *
 * with @f$p@f$ the exponent the phase-space mapping already absorbs. Only
 * @f$\bar\phi_n@f$ is in the grid; baking the @f$(1-x)^{-\kappa}@f$ factor in
 * would destroy exactly the endpoint resolution the reduction exists to keep.
 *
 * @f$1-x@f$ is taken as its own argument rather than recomputed from @p x:
 * near the endpoint the density lives in @f$1-x@f$, and a caller that sampled
 * it there must be able to pass it through at full precision.
 *
 * The component is chosen per event, which is what lets the caller sample the
 * four components discretely instead of evaluating all of them.
 *
 * `batch` is the leading batch dimension.
 *
 * **Arguments**
 * - `x` – `float`, shape `(batch,)` – momentum fraction.
 * - `omx` – `float`, shape `(batch,)` – @f$1-x@f$.
 * - `q` – `float`, shape `(batch,)` – factorization scale.
 * - `component_index` – `int`, shape `(batch,)` – which component to return,
 *   indexing the `pids` given to the constructor.
 *
 * **Returns**
 * - `pdf` – `float`, shape `(batch,)` – the unreduced density.
 */
class EEPartonDensity : public FunctionGenerator {
public:
    /**
     * @param grid              Grid of the reduced density, one flavour per
     *                          component (see
     *                          `vendor/eepdfgrid/gen_eepdf_lhagrid.py`).
     * @param pids              Component ids to select between.
     * @param exponent_offsets  Per component, @f$p - \kappa^{(0)}_n@f$; the
     *                          @f$Q@f$-dependent part of @f$\kappa@f$ is
     *                          applied here. One entry per entry of @p pids.
     * @param alpha_em          @f$\alpha@f$ in the scheme the grid was fitted
     *                          in.
     * @param lepton_mass       @f$m@f$ of the beam lepton, in GeV.
     * @param prefix            Prefix for the grid global names.
     */
    EEPartonDensity(
        const PdfGrid& grid,
        const std::vector<int>& pids,
        const std::vector<double>& exponent_offsets,
        double alpha_em = 0.007562397,
        double lepton_mass = 0.511e-3,
        const std::string& prefix = ""
    );

    /// Upload the per-component exponents to @p context. The grid's own
    /// globals are uploaded by @ref PdfGrid::initialize_globals.
    void initialize_globals(ContextPtr context) const;

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    PartonDensity _density;
    std::vector<double> _offsets;
    double _alpha_em;
    double _lepton_mass;
    std::string _prefix;
};

/**
 * Two-beam luminosity of a dressed-lepton density, and its component choice.
 *
 * The beam densities do not simply multiply. A dressed-lepton density is a
 * sum of factorized terms, and the two-beam luminosity pairs them off term by
 * term,
 *
 * @f[
 *   L(x_1, x_2, Q) = \sum_{n} f^{(1)}_n(x_1, Q)\, f^{(2)}_n(x_2, Q),
 * @f]
 *
 * which is `ee_comp_prod` of `pdg2pdf.f`. Pairing the *sums* instead would be
 * wrong, so the two pid lists must be given in the same component order.
 *
 * One component is also sampled, with probability proportional to
 * @f$|f^{(1)}_n f^{(2)}_n|@f$. That choice does not enter @p luminosity and so
 * does not affect the cross section — it is what the beamstrahlung split needs
 * downstream, where each component carries its own spectrum (the same choice
 * `sample_bstrahl.f` makes).
 *
 * `batch` is the leading batch dimension.
 *
 * **Arguments**
 * - `x1`, `omx1` – `float`, shape `(batch,)` – first beam's fraction and
 *   @f$1-x@f$.
 * - `x2`, `omx2` – `float`, shape `(batch,)` – second beam's.
 * - `q` – `float`, shape `(batch,)` – factorization scale.
 * - `r` – `float`, shape `(batch,)` – random number for the component choice.
 *
 * **Returns**
 * - `luminosity` – `float`, shape `(batch,)` – @f$L@f$, signed.
 * - `component_index` – `int`, shape `(batch,)` – the sampled component,
 *   indexing the pid lists.
 */
class EELuminosity : public FunctionGenerator {
public:
    /**
     * @param grid       Grid of the reduced densities.
     * @param pids1      Components of the first beam, in component order.
     * @param pids2      Components of the second beam, same order.
     * @param offsets1   Per component, @f$p - \kappa^{(0)}_n@f$ for beam 1.
     * @param offsets2   The same for beam 2. The two differ: which components
     *                   are the sharply peaked ones depends on the beam.
     * @param alpha_em   @f$\alpha@f$ in the scheme the grid was fitted in.
     * @param lepton_mass @f$m@f$ of the beam leptons, in GeV.
     * @param prefix     Prefix for the grid global names.
     */
    EELuminosity(
        const PdfGrid& grid,
        const std::vector<int>& pids1,
        const std::vector<int>& pids2,
        const std::vector<double>& offsets1,
        const std::vector<double>& offsets2,
        double alpha_em = 0.007562397,
        double lepton_mass = 0.511e-3,
        const std::string& prefix = ""
    );

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    PartonDensity _density1, _density2;
    std::vector<double> _offsets1, _offsets2;
    double _alpha_em;
    double _lepton_mass;
};

/**
 * A strong-coupling grid in the LHAPDF format, loaded from a file.
 *
 * Holds the @f$\alpha_s(q)@f$ knots of an LHAPDF `.info` set;
 * @ref RunningCoupling does the interpolation.
 */
struct AlphaSGrid {
    /// Scale grid knots.
    std::vector<double> q;
    /// `log(q^2)` grid knots.
    std::vector<double> logq2;
    /// `alpha_s` values at the knots.
    std::vector<double> values;
    /// Number of `q` intervals in each sub-grid region.
    std::vector<std::size_t> region_sizes;

    /// Load an LHAPDF-format `alpha_s` grid.
    /// @param file  Path to the grid file.
    AlphaSGrid(const std::string& file);
    /// Number of scale knots, counting the knots shared by two regions once.
    std::size_t q_count() const;
    /// Fill @p tensor with the interpolation coefficients.
    void initialize_coefficients(Tensor tensor) const;
    /// Fill @p tensor with the `log(q^2)` knots.
    void initialize_logq2(Tensor tensor) const;
    /// Shape of the coefficient tensor, optionally with a leading batch axis.
    std::vector<std::size_t> coefficients_shape(bool batch_dim = false) const;
    /// Shape of the `log(q^2)` tensor.
    std::vector<std::size_t> logq2_shape(bool batch_dim = false) const;
    /// Upload the grid to @p context as compute-graph globals under @p prefix.
    void initialize_globals(ContextPtr context, const std::string& prefix = "") const;
};

/**
 * Strong-coupling interpolation as a compute-graph function.
 *
 * Interpolates an @ref AlphaSGrid to return @f$\alpha_s(q)@f$ at the sampled
 * scale (Sec. 3.2.3 of [1]).
 *
 * `batch` is the leading batch dimension.
 *
 * **Arguments**
 * - `q` – `float`, shape `(batch,)` – renormalization scale.
 *
 * **Returns**
 * - `alpha_s` – `float`, shape `(batch,)` – the interpolated @f$\alpha_s(q)@f$.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 3.2.3)
 */
class RunningCoupling : public FunctionGenerator {
public:
    /// @param grid    The `alpha_s` grid to interpolate.
    /// @param prefix  Prefix for the grid global names.
    RunningCoupling(const AlphaSGrid& grid, const std::string& prefix = "");

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::string _prefix;
    std::vector<std::size_t> _logq2_shape;
    std::vector<std::size_t> _coeffs_shape;
};

} // namespace madspace

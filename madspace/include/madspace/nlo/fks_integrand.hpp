#pragma once

#include <optional>
#include <vector>

#include "madspace/nlo/cuts.hpp"
#include "madspace/nlo/fks_limits.hpp"
#include "madspace/nlo/fks_mapping.hpp"
#include "madspace/nlo/fks_subtraction.hpp"
#include "madspace/nlo/scale.hpp"
#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/matrix_element.hpp"

namespace madspace {

/**
 * The FKS integrand of one Born channel at given random numbers and FKS
 * configurations, with its matrix elements from UMAMI libraries, as the
 * fixed-order driver of MadGraph7 evaluates a channel of a subprocess
 * (SubprocessIntegrand.evaluate of madgraph/mg7/nlo/fks.py).
 *
 * @ref FKSKinematics gives the event and its counter-events, @ref
 * FKSSplittingKernels the transverse direction @f$k_\perp@f$ of the
 * collinear counter-event and @ref NLOCuts which kinematics pass the cuts.
 * The matrix elements come through @ref MatrixElement with the provisional
 * NLO keys of UMAMI, at @f$\alpha_s = 1 / (4 \pi)@f$, that is per squared
 * coupling order without the strong coupling, and at the electroweak coupling
 * @f$e@f$ of the model, where the cuts let a contribution through:
 * - the real of the real matrix element of the configuration of the event
 *   (`squared_orders`), where the event has a weight and passes the cuts;
 * - the Born (`squared_orders`), its amplitudes per Born diagram of the
 *   channels (`diagram_amp2`), its colour correlations over the pairs of
 *   coloured legs (`color_correlated`) and its spin correlation
 *   (`spin_correlated`) at the mother of the configuration with
 *   @f$k_\perp@f$ for a gluon or photon mother, where the Born has a weight
 *   and one of the event and the soft and collinear counter-events passes
 *   the cuts;
 * - the virtual (`loop`, its finite part) at the renormalization scale of
 *   the Born, where the soft counter-event passes the cuts and `virtual_mask`
 *   is not zero.
 *
 * The colour-correlated Born that @ref FKSCrossSection takes is the matrix
 * of @f$\langle T_m \cdot T_n \rangle B@f$ over all Born legs: the pairs of
 * the library off the diagonal, @f$C_m B@f$ with the Casimir of the
 * representation of the leg on it and zero for the colourless legs. The
 * spin-correlated Born that @ref FKSCrossSection takes is @f$B / 2@f$
 * for the other mothers, as the azimuthal terms of their splittings vanish.
 * The channel weight is that of MadFKS, the amplitude of the Born diagram of
 * the channel over the sum of those of all Born diagrams, zero where the Born
 * is not evaluated. The virtual takes the virtual trick of MadFKS: with the
 * average @f$r_o@f$ of @f$V_o / B@f$ per NLO order and the fraction @f$f@f$
 * of the points with the virtual, @ref FKSCrossSection takes @f$r_o B + (V_o
 * - r_o B) / f@f$ where the virtual is evaluated and @f$r_o B@f$ elsewhere,
 * with @f$B@f$ the sum of the Born orders. `virtual_residual` is the cross
 * section less that with @f$r_o B@f$ everywhere, the part of the evaluated
 * virtual that adapts the fraction.
 *
 * With tagged photons and QED corrections, the factors of the couplings of
 * the tagged photons multiply the real and the virtual per NLO order, the
 * Born and its correlated versions per Born order and the QED coupling of the
 * counterterms (tagged_factors of madgraph/mg7/nlo/provider.py).
 *
 * The scale of the user, the Borns of extra counterterms and the Sudakov
 * approximation of the EW corrections are not covered yet.
 *
 * `batch` is the leading batch dimension, `b` the number of Born orders and
 * `o` the number of NLO orders.
 *
 * **Arguments**
 * - `random` – `float`, shape `(batch, d)` – the random numbers of
 *   @ref FKSKinematics.
 * - `fks_index` – `int`, shape `(batch,)` – the FKS configuration.
 * - `configuration_weight` – `float`, shape `(batch,)` – the weight of the
 *   choice of the configuration.
 * - `virtual_mask` – `float`, shape `(batch,)` – nonzero for the points that
 *   take the virtual.
 * - `virtual_ratio` – `float`, shape `(batch, o)` – @f$r_o@f$.
 * - `virtual_fraction` – `float`, shape `(batch,)` – @f$f@f$.
 *
 * **Returns**
 * - those of @ref FKSCrossSection, `cross_section` with the virtual trick.
 * - `virtual_residual` – `float`, shape `(batch,)` – the part of the
 *   cross section from the evaluated virtual.
 * - `virtual` – `float`, shape `(batch, o)` – the virtual where it is
 *   evaluated, zero elsewhere.
 * - `born` – `float`, shape `(batch, b)` – the Born where it is evaluated,
 *   zero elsewhere.
 * - `evaluated` – `float`, shape `(batch,)` – one where the virtual is
 *   evaluated, zero elsewhere.
 * - `momenta`, `soft_momenta`, `collinear_momenta`, `soft_collinear_momenta`
 *   – `float`, shape `(batch, n + 1, 4)`, and `born_momenta` – `float`,
 *   shape `(batch, n, 4)` – the kinematics of @ref FKSKinematics.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 * - [2] R. Frederix, S. Frixione, F. Maltoni, T. Stelzer, "Automation of
 *   next-to-leading order computations in QCD: the FKS subtraction",
 *   https://arxiv.org/abs/0908.4272
 */
class FKSChannelIntegrand : public FunctionGenerator {
public:
    /**
     * @param kinematics      The kinematics of the channel.
     * @param kernels         The splitting kernels, for @f$k_\perp@f$.
     * @param cuts            The generation cuts of the real processes, with
     *                        `fks_index` for the layout.
     * @param born_scale      The scales of the Born, for the renormalization
     *                        scale of the virtual.
     * @param cross_section   The integrand, with the scales, the couplings and
     *                        the luminosities inside.
     * @param born            The Born library: inputs `momenta_in`,
     *                        `alpha_s_in`, `g_ew_in`, `spin_legs_in`,
     *                        `spin_vectors_in`, outputs `squared_orders_out`,
     *                        `diagram_amp2_out`, `spin_correlated_out` and
     *                        `color_correlated_out` with at least two
     *                        coloured legs.
     * @param reals           The real libraries, one per real matrix element:
     *                        inputs `momenta_in`, `alpha_s_in`, `g_ew_in`,
     *                        output `squared_orders_out`.
     * @param loop            The loop library: inputs `momenta_in`,
     *                        `alpha_s_in`, `g_ew_in`, `scale_in`, output
     *                        `loop_out`. Without it the virtual is zero.
     * @param real_indices    Per FKS configuration its real matrix element,
     *                        an index of @p reals.
     * @param spin_legs       Per FKS configuration the Born leg of its mother,
     *                        from 0, for a gluon or photon mother, -1 for the
     *                        others.
     * @param color_representations Per Born leg its colour representation
     *                        as UMAMI_META_COLOR_REPRESENTATIONS gives it, 1
     *                        for a colourless leg.
     * @param channel         The Born diagram of the channel, an index of
     *                        `diagram_amp2`.
     * @param g_ew            The electroweak coupling of the libraries, the
     *                        @f$e@f$ of the model, as the cross section takes
     *                        the blocks with their powers of @f$e@f$.
     * @param alpha           The QED coupling of the counterterms.
     * @param tagged_nlo      The factors of tagged photons per NLO order, none
     *                        if empty.
     * @param tagged_born     Those per Born order, none if empty.
     * @param tagged_alpha    That of the QED coupling.
     * @throws std::invalid_argument if the pieces do not fit together, the
     *         cross section takes the scale of the user, the Sudakov
     *         approximation or Borns of extra counterterms, a table has
     *         another length than the configurations or the Born legs, or a
     *         leg has a colour representation without a Casimir here.
     */
    FKSChannelIntegrand(
        const FKSKinematics& kinematics,
        const FKSSplittingKernels& kernels,
        const NLOCuts& cuts,
        const NLOScale& born_scale,
        const FKSCrossSection& cross_section,
        const MatrixElement& born,
        const std::vector<MatrixElement>& reals,
        const std::optional<MatrixElement>& loop,
        const std::vector<me_int_t>& real_indices,
        const std::vector<me_int_t>& spin_legs,
        const std::vector<int>& color_representations,
        std::size_t channel,
        double g_ew,
        double alpha,
        const std::vector<double>& tagged_nlo = {},
        const std::vector<double>& tagged_born = {},
        double tagged_alpha = 1.
    );
    /// Number of FKS configurations.
    std::size_t configuration_count() const { return _real_indices.size(); }
    /// Number of Born orders.
    std::size_t born_order_count() const { return _born_orders; }
    /// Number of NLO orders.
    std::size_t nlo_order_count() const { return _nlo_orders; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    FKSKinematics _kinematics;
    FKSSplittingKernels _kernels;
    NLOCuts _cuts;
    NLOScale _born_scale;
    FKSCrossSection _cross_section;
    MatrixElement _born;
    std::vector<MatrixElement> _reals;
    std::optional<MatrixElement> _loop;
    std::vector<me_int_t> _real_indices;
    std::vector<me_int_t> _spin_legs;
    std::vector<double> _has_spin_leg;
    // per Born leg the indices of its row of the colour matrix in the pairs
    // of one order followed by the Casimirs times the Born and a zero
    nested_vector2<me_int_t> _color_rows;
    std::vector<double> _casimirs;
    bool _has_color;
    std::size_t _channel;
    double _g_ew;
    double _alpha;
    std::vector<double> _tagged_nlo, _tagged_born;
    double _tagged_alpha;
    std::size_t _born_orders, _nlo_orders, _born_particles;
};

} // namespace madspace

#pragma once

#include <optional>
#include <vector>

#include "madspace/nlo/common.hpp"
#include "madspace/nlo/cs_integrated.hpp"
#include "madspace/nlo/cuts.hpp"
#include "madspace/nlo/luminosity.hpp"
#include "madspace/nlo/scale.hpp"
#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/custom_function.hpp"
#include "madspace/phasespace/matrix_element.hpp"
#include "madspace/phasespace/pdf.hpp"
#include "madspace/phasespace/phasespace.hpp"

namespace madspace {

/**
 * The n-body stream of the Catani-Seymour integrand of one Born channel at
 * given random numbers, with its matrix elements from UMAMI libraries, as the
 * fixed-order driver of MadGraph7 evaluates a channel of the n-body stream of
 * a subprocess (CSBornIntegrand.evaluate of madgraph/mg7/nlo/cs.py).
 *
 * The @ref PhaseSpaceMapping of the channel gives the Born, @ref NLOCuts
 * whether it passes the cuts, @ref NLOScale its scales and
 * @ref RunningCoupling @f$\alpha_s@f$ at the renormalization scale. Where the
 * Born has a weight and passes the cuts, the libraries give at
 * @f$\alpha_s = 1 / (4 \pi)@f$ and the electroweak coupling @f$e@f$ of the
 * model the Born (`squared_orders`), its amplitudes per Born diagram
 * (`diagram_amp2`) and its colour correlations over the pairs of coloured legs
 * (`color_correlated`), and where `virtual_mask` is not zero as well the
 * virtual (`loop`, its finite part) at the renormalization scale. The colour
 * matrix of the operators is that of @ref ColorMatrix.
 *
 * Per Born order @f$k@f$ with @f$g_s^{p_k}@f$ the weight is
 *
 * @f[ w = J F c \Big[ \sum_k g_s^{p_k} \Big( L B_k + L \sum_G I^G_k +
 *     \sum_G \sum_{i = 1, 2} \big(W^G_{k i 1} L^G_{i 1}
 *     + W^G_{k i 2} L^G_{i 2} + W^G_{k i 3} L\big) \Big)
 *     + \sum_o g_s^{q_o} L V_o \Big], @f]
 *
 * with the Jacobian @f$J@f$ of the mapping, the flux @f$F@f$ (with the
 * conversion to pb, @f$1 / (2 M)@f$ for a decay), the channel weight
 * @f$c@f$ of MadFKS (the amplitude of the Born diagram of the channel over the
 * sum of those of all Born diagrams), the luminosity @f$L@f$ of the Born, the
 * insertion operators @f$I^G@f$ of the groups of corrections @f$G@f$ (QCD and
 * QED, @ref CSInsertion) of the Born orders they correct, and for hadron beams
 * the weights @f$W^G@f$ of the K and P operators (@ref CSCollinear) with the
 * luminosities of the same parton and of the partons that split into it at
 * @f$x_i / z_i@f$ over @f$z_i@f$ of each beam (zero for @f$x_i > z_i@f$).
 * The virtual of NLO order @f$o@f$ with @f$g_s^{q_o}@f$ takes the virtual
 * trick of @ref virtual_trick.
 *
 * `batch` is the leading batch dimension, `d` the number of random numbers
 * of the mapping, `b` the number of Born orders, `o` the number of NLO orders
 * and `n` the number of Born particles.
 *
 * **Arguments**
 * - `random` – `float`, shape `(batch, d)` or `(batch, d + 1)` – the random
 *   numbers of the mapping, for hadron beams followed by that of the
 *   convolution of K and P.
 * - `virtual_mask` – `float`, shape `(batch,)` – nonzero for the points that
 *   take the virtual.
 * - `virtual_ratio` – `float`, shape `(batch, o)` – @f$r_o@f$ of the virtual
 *   trick.
 * - `virtual_fraction` – `float`, shape `(batch,)` – @f$f@f$ of the virtual
 *   trick.
 *
 * **Returns**
 * - `cross_section` – `float`, shape `(batch,)` – the weight, with the
 *   virtual trick.
 * - `lo_weight` – `float`, shape `(batch,)` – its LO part.
 * - `virtual_residual` – `float`, shape `(batch,)` – the part of the weight
 *   from the evaluated virtual.
 * - `virtual` – `float`, shape `(batch, o)` – the virtual where it is
 *   evaluated, zero elsewhere.
 * - `born` – `float`, shape `(batch, b)` – the Born where it is evaluated,
 *   zero elsewhere.
 * - `evaluated` – `float`, shape `(batch,)` – one where the virtual is
 *   evaluated, zero elsewhere.
 * - `momenta` – `float`, shape `(batch, n, 4)` – the Born.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 * - [2] S. Catani, M. H. Seymour, https://arxiv.org/abs/hep-ph/9605323
 * - [3] S. Catani, S. Dittmaier, M. H. Seymour, Z. Trocsanyi,
 *   https://arxiv.org/abs/hep-ph/0201036
 */
class CSBornChannelIntegrand : public FunctionGenerator {
public:
    /**
     * @param mapping       The mapping of the Born channel.
     * @param cuts          The cuts of the Born, without unresolved partons.
     * @param scale         The scales of the Born.
     * @param running_coupling The running of @f$\alpha_s@f$.
     * @param luminosity    The luminosity of the Born, one group with all its
     *                      flavors (`all_groups`).
     * @param born          The Born library: inputs `momenta_in`,
     *                      `alpha_s_in`, `g_ew_in`, outputs
     *                      `squared_orders_out`, `diagram_amp2_out` and
     *                      `color_correlated_out` with at least two coloured
     *                      legs.
     * @param loop          The loop library: inputs `momenta_in`,
     *                      `alpha_s_in`, `g_ew_in`, `scale_in`, output
     *                      `loop_out`. Without it the virtual is zero.
     * @param color_representations Per Born leg its colour representation,
     *                      as for @ref ColorMatrix.
     * @param channel       The Born diagram of the channel, an index of
     *                      `diagram_amp2`.
     * @param g_ew          The electroweak coupling of the libraries, the
     *                      @f$e@f$ of the model.
     * @param alpha         The QED coupling of the QED operators.
     * @param born_powers   Per Born order its power of @f$g_s@f$.
     * @param nlo_powers    Per NLO order its power of @f$g_s@f$.
     * @param insertion     The insertion operator of the QCD corrections.
     * @param insertion_qed The insertion operator of the QED corrections.
     * @param qcd_corrected Per Born order whether the QCD operators correct
     *                      it.
     * @param qed_corrected Per Born order whether the QED operators correct
     *                      it.
     * @param collinear     The K and P operators of the QCD corrections, with
     *                      hadron beams.
     * @param collinear_qed The K and P operators of the QED corrections, with
     *                      hadron beams.
     * @param convolution   With hadron beams, per beam the luminosity of the
     *                      flavors of the Born and of each group of
     *                      corrections in the order QCD, QED with their
     *                      partons of that beam split (`all_groups`).
     * @param energy        The energy of the beams, the mass of the decaying
     *                      particle for a decay.
     * @param reference_scale The reference scale of the user at the Born, a
     *                      @ref CustomFunction of type scale, needed when the
     *                      scales take the scale of the user.
     * @throws std::invalid_argument if the pieces do not fit together: the
     *         libraries and operators with other particles or orders than the
     *         Born, a table with another length than the orders, K and P
     *         without the convolutions or with lepton beams, or the scales of
     *         the user without @p reference_scale.
     */
    CSBornChannelIntegrand(
        const PhaseSpaceMapping& mapping,
        const NLOCuts& cuts,
        const NLOScale& scale,
        const RunningCoupling& running_coupling,
        const NLOLuminosity& luminosity,
        const MatrixElement& born,
        const std::optional<MatrixElement>& loop,
        const std::vector<int>& color_representations,
        std::size_t channel,
        double g_ew,
        double alpha,
        const std::vector<double>& born_powers,
        const std::vector<double>& nlo_powers,
        const std::optional<CSInsertion>& insertion,
        const std::optional<CSInsertion>& insertion_qed,
        const std::vector<bool>& qcd_corrected,
        const std::vector<bool>& qed_corrected,
        const std::optional<CSCollinear>& collinear,
        const std::optional<CSCollinear>& collinear_qed,
        const std::vector<NLOLuminosity>& convolution,
        double energy,
        const std::optional<CustomFunction>& reference_scale = std::nullopt
    );
    /// Number of random numbers.
    std::size_t random_dim() const { return _random_dim; }
    /// Number of Born orders.
    std::size_t born_order_count() const { return _born_powers.size(); }
    /// Number of NLO orders.
    std::size_t nlo_order_count() const { return _nlo_powers.size(); }
    /// Whether the integrand convolves K and P over the beams, with one
    /// random number more than the mapping.
    bool convolves() const { return !_convolution.empty(); }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    PhaseSpaceMapping _mapping;
    NLOCuts _cuts;
    NLOScale _scale;
    RunningCoupling _running_coupling;
    NLOLuminosity _luminosity;
    MatrixElement _born;
    std::optional<MatrixElement> _loop;
    ColorMatrix _color;
    std::size_t _channel;
    double _g_ew, _alpha;
    std::vector<double> _born_powers, _nlo_powers;
    std::optional<CSInsertion> _insertion, _insertion_qed;
    std::vector<bool> _qcd_corrected, _qed_corrected;
    std::optional<CSCollinear> _collinear, _collinear_qed;
    std::vector<NLOLuminosity> _convolution;
    double _energy;
    std::optional<CustomFunction> _reference_scale;
    std::size_t _random_dim, _particles;
};

} // namespace madspace

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "madspace/nlo/common.hpp"
#include "madspace/nlo/cs_histograms.hpp"
#include "madspace/nlo/cs_integrated.hpp"
#include "madspace/nlo/cs_mapping.hpp"
#include "madspace/nlo/cuts.hpp"
#include "madspace/nlo/iteration_state.hpp"
#include "madspace/nlo/luminosity.hpp"
#include "madspace/nlo/scale.hpp"
#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/custom_function.hpp"
#include "madspace/phasespace/flow.hpp"
#include "madspace/phasespace/integrand.hpp"
#include "madspace/phasespace/matrix_element.hpp"
#include "madspace/phasespace/pdf.hpp"
#include "madspace/phasespace/phasespace.hpp"
#include "madspace/phasespace/vegas.hpp"

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
 * (`diagram_amp2`) and its color correlations over the pairs of colored legs
 * (`color_correlated`), and where `virtual_mask` is not zero as well the
 * virtual (`loop`, its finite part) at the renormalization scale. The color
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
 * - [2] S. Catani, M. H. Seymour, "A general algorithm for calculating jet
 *   cross sections in NLO QCD", https://arxiv.org/abs/hep-ph/9605323
 * - [3] S. Catani, S. Dittmaier, M. H. Seymour, Z. Trocsanyi, "The dipole
 *   formalism for next-to-leading order QCD calculations with massive
 *   partons", https://arxiv.org/abs/hep-ph/0201036
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
     *                      `color_correlated_out` with at least two colored
     *                      legs.
     * @param loop          The loop library: inputs `momenta_in`,
     *                      `alpha_s_in`, `g_ew_in`, `scale_in`, output
     *                      `loop_out`. Without it the virtual is zero.
     * @param color_representations Per Born leg its color representation,
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

/**
 * The Born of one dipole of a real process, as @ref CSRealChannelIntegrand
 * takes it: its library, cuts and scales, and how the dipole takes its
 * blocks.
 */
struct CSRealDipole {
    /**
     * @param born          The Born library: inputs `momenta_in`,
     *                      `alpha_s_in`, `g_ew_in`, `spin_legs_in`,
     *                      `spin_vectors_in`, outputs `squared_orders_out`,
     *                      `diagram_amp2_out`, `color_correlated_out` with at
     *                      least two colored legs, `spin_correlated_out`.
     * @param cuts          The cuts of the Born, without unresolved partons.
     * @param scale         The scales of the Born.
     * @param order         Per particle of the library the particle of the
     *                      Born of the mapping of the dipole (the real
     *                      process without the emitted particle).
     * @param emitter       The emitter, a particle of the library.
     * @param spectator     The spectator, a particle of the library.
     * @param color_representations Per particle of the library its color
     *                      representation, as for @ref ColorMatrix.
     * @param born_powers   Per Born order its power of @f$g_s@f$.
     * @param corrected     Per Born order whether the dipole corrects it.
     * @param qed           Whether the dipole is one of QED, with the charges
     *                      of @p charge in place of the color correlation
     *                      and @f$e^2 = 4 \pi \alpha@f$ in place of
     *                      @f$g_s^2@f$.
     * @param charge        For a QED dipole the factor of the Born in place
     *                      of the color correlation.
     * @param alpha         The QED coupling of a QED dipole.
     * @param factor        The factor of the weights of the dipole.
     * @param spin_leg      The particle of the library whose polarization the
     *                      spin correlation replaces, -1 for none.
     * @param swap          A particle of the library exchanged with the
     *                      emitter for the spin correlation (an identical
     *                      boson with the spin correlation the emitter has
     *                      not), -1 for none.
     * @param reference_scale The reference scale of the user at the Born.
     */
    CSRealDipole(
        const MatrixElement& born,
        const NLOCuts& cuts,
        const NLOScale& scale,
        const std::vector<std::size_t>& order,
        std::size_t emitter,
        std::size_t spectator,
        const std::vector<int>& color_representations,
        const std::vector<double>& born_powers,
        const std::vector<bool>& corrected,
        bool qed = false,
        double charge = 0.,
        double alpha = 0.,
        double factor = 1.,
        int spin_leg = -1,
        int swap = -1,
        const std::optional<CustomFunction>& reference_scale = std::nullopt
    );

    MatrixElement born;
    NLOCuts cuts;
    NLOScale scale;
    std::vector<std::size_t> order;
    std::size_t emitter, spectator;
    std::vector<int> color_representations;
    std::vector<double> born_powers;
    std::vector<bool> corrected;
    bool qed;
    double charge, alpha, factor;
    int spin_leg, swap;
    std::optional<CustomFunction> reference_scale;
    /// The index of the color-correlated pair of the emitter and the
    /// spectator among those of the library, -1 for a QED dipole.
    int color_pair;
};

/**
 * The real stream of the Catani-Seymour integrand of one real process and
 * channel at given random numbers, with its matrix elements from UMAMI
 * libraries, as the fixed-order driver of MadGraph7 evaluates a channel of
 * the real stream of a real process (CSRealIntegrand.evaluate of
 * madgraph/mg7/nlo/cs.py with the combined channels).
 *
 * The channels are those of the dipoles, a @ref CSMapping per dipole and
 * Born diagram of its Born, and those of the diagrams of the real process,
 * a @ref PhaseSpaceMapping each. The mapping of the channel gives the real
 * emission, the weight of the channel is the density of its mapping over the
 * sum of those of all channels at the event (as @ref MappingChannelWeights,
 * zero for a channel that cannot produce the event), so that the weights add
 * up to one at every event [4]. The technical cut is the cutoff of the
 * mappings of the dipoles: a channel of a dipole covers the events where the
 * two variables of its dipole are at or above it, one of a diagram those
 * where the variables of all dipoles are.
 *
 * The real emission takes its cuts (@ref NLOCuts), scales (@ref NLOScale),
 * @f$\alpha_s@f$ (@ref RunningCoupling) and luminosity (@ref NLOLuminosity),
 * and the real library at @f$\alpha_s = 1 / (4 \pi)@f$ and the electroweak
 * coupling @f$e@f$ of the model its matrix element per NLO order. Each
 * dipole projects the event onto its Born (@ref CSDipoles), which takes the
 * cuts, scales and @f$\alpha_s@f$ of its @ref CSRealDipole and the
 * luminosity of the real emission at its momentum fractions and the
 * factorization scales of the Born. Where the Born passes its cuts and the
 * weights of the dipole are not zero, the dipole is
 *
 * @f[ -D = c^2 \sum_k g_s^{p_k} \Big( w_c C_k + w_s
 *     \frac{C_k}{B_k} B_k(k_t) \Big), @f]
 *
 * over the Born orders @f$k@f$ it corrects, with @f$c^2 = g_s^2@f$ (QCD) or
 * @f$4 \pi \alpha@f$ (QED), the color weight @f$w_c@f$ and spin weight
 * @f$w_s@f$ of @ref CSDipoles times the factor of the dipole, the color
 * correlation @f$C_k = \langle T_j \cdot T_k \rangle B_k@f$ of the emitter
 * and the spectator (the charge times the Born for QED) and the Born with
 * the polarization of a gluon or photon emitter replaced by the vector
 * @f$k_t@f$ of the dipole (`spin_correlated` of the library, @f$-k_t^2@f$
 * times its spin-correlated Born), the factorized form of the spin
 * correlation of the driver.
 *
 * The weight is that of the real emission minus the dipoles times the
 * Jacobian of the mapping, the weight of the channel and the flux (with the
 * conversion to pb, @f$1 / (2 M)@f$ for a decay).
 *
 * `batch` is the leading batch dimension, `r` the number of random numbers
 * of the mapping of the channel, `d` the number of dipoles and `n` the number
 * of particles of the real emission.
 *
 * **Arguments**
 * - `random` – `float`, shape `(batch, r)` – the random numbers of the
 *   mapping of the channel.
 *
 * **Returns**
 * - `cross_section` – `float`, shape `(batch,)` – the weight.
 * - `real_weight` – `float`, shape `(batch,)` – its part of the real
 *   emission.
 * - `dipole_weights` – `float`, shape `(batch, d)` – its parts of the
 *   dipoles, @f$-D@f$.
 * - `channel_weight` – `float`, shape `(batch,)` – the weight of the
 *   channel.
 * - `momenta` – `float`, shape `(batch, n, 4)` – the real emission.
 * - `born_momenta` – `float`, shape `(batch, d, n - 1, 4)` – the Borns of the
 *   dipoles, in the order of the particles of their libraries.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 * - [2] S. Catani, M. H. Seymour, "A general algorithm for calculating jet
 *   cross sections in NLO QCD", https://arxiv.org/abs/hep-ph/9605323
 * - [3] S. Catani, S. Dittmaier, M. H. Seymour, Z. Trocsanyi, "The dipole
 *   formalism for next-to-leading order QCD calculations with massive
 *   partons", https://arxiv.org/abs/hep-ph/0201036
 * - [4] R. Kleiss, R. Pittau, "Weight optimization in multichannel Monte
 *   Carlo", https://arxiv.org/abs/hep-ph/9405257
 */
class CSRealChannelIntegrand : public FunctionGenerator {
public:
    /**
     * @param dipoles       The dipoles of the real process.
     * @param borns         Per dipole its Born.
     * @param dipole_channels The channels of the dipoles, a @ref CSMapping
     *                      per dipole and Born diagram.
     * @param real_channels The channels of the diagrams of the real process.
     * @param channel       The channel, an index of the channels of the
     *                      dipoles followed by those of the diagrams.
     * @param cuts          The cuts of the real emission.
     * @param scale         The scales of the real emission.
     * @param running_coupling The running of @f$\alpha_s@f$.
     * @param luminosity    The luminosity of the real emission, one group
     *                      with all its flavors (`all_groups`).
     * @param real          The real library: inputs `momenta_in`,
     *                      `alpha_s_in`, `g_ew_in`, output
     *                      `squared_orders_out`.
     * @param real_powers   Per NLO order its power of @f$g_s@f$.
     * @param g_ew          The electroweak coupling of the libraries, the
     *                      @f$e@f$ of the model.
     * @param energy        The energy of the beams, the mass of the decaying
     *                      particle for a decay.
     * @param reference_scale The reference scale of the user at the real
     *                      emission.
     * @throws std::invalid_argument if the pieces do not fit together: the
     *         mappings, dipoles, libraries or Borns with other particles,
     *         orders or keys, a channel outside the channels, or the scales
     *         of the user without a reference scale.
     */
    CSRealChannelIntegrand(
        const CSDipoles& dipoles,
        const std::vector<CSRealDipole>& borns,
        const std::vector<CSMapping>& dipole_channels,
        const std::vector<PhaseSpaceMapping>& real_channels,
        std::size_t channel,
        const NLOCuts& cuts,
        const NLOScale& scale,
        const RunningCoupling& running_coupling,
        const NLOLuminosity& luminosity,
        const MatrixElement& real,
        const std::vector<double>& real_powers,
        double g_ew,
        double energy,
        const std::optional<CustomFunction>& reference_scale = std::nullopt
    );
    /// Number of random numbers.
    std::size_t random_dim() const { return _random_dim; }
    /// Number of channels, those of the dipoles and of the diagrams.
    std::size_t channel_count() const {
        return _dipole_channels.size() + _real_channels.size();
    }
    /// Number of dipoles.
    std::size_t dipole_count() const { return _borns.size(); }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    CSDipoles _dipoles;
    std::vector<CSRealDipole> _borns;
    std::vector<CSMapping> _dipole_channels;
    std::vector<PhaseSpaceMapping> _real_channels;
    std::size_t _channel;
    NLOCuts _cuts;
    NLOScale _scale;
    RunningCoupling _running_coupling;
    NLOLuminosity _luminosity;
    MatrixElement _real;
    std::vector<double> _real_powers;
    double _g_ew, _energy;
    std::optional<CustomFunction> _reference_scale;
    std::size_t _random_dim, _particles;
    bool _decay;
};

/**
 * The Catani-Seymour integrand of a channel as a @ref ChannelEventGenerator
 * runs it at fixed order (see @ref EventGenerator::integrate): a
 * @ref CSBornChannelIntegrand (the n-body stream) or a
 * @ref CSRealChannelIntegrand (the real stream) at random numbers through an
 * optional @ref VegasMapping or @ref Flow. For the n-body stream the points
 * with the virtual are drawn per point from the fraction and ratios of the
 * virtual trick that its @ref NLOIterationState adapts. The channels of the
 * real stream need no state besides the sums of the results.
 *
 * With @p madnis_training, the integrand gives the quantities of the
 * @ref MadnisLoss instead, without the virtual (the virtual trick then takes
 * @f$r_o@f$ times the Born at all points), whose integrand in the latent space
 * of the adaptive mapping is the cross section.
 *
 * `batch` is the leading batch dimension, `d` the number of random numbers of
 * the channel integrand, `o` the number of NLO orders, `b` the number of Born
 * orders, `m` the number of dipoles, `n` the number of particles of the
 * kinematics of the stream (the Born or the real emission).
 *
 * **Arguments**
 * - `batch_size` – the number of points.
 *
 * **Returns** (without @p madnis_training)
 * - `weight` – `float`, shape `(batch,)` – the cross section, with the
 *   virtual trick for the n-body stream, times the Jacobian of the adaptive
 *   mapping, as all weights here.
 * - n-body stream: `lo_weight`, `virtual_residual` – those of
 *   @ref CSBornChannelIntegrand, `virtual` – `float`, shape `(batch, o)`,
 *   `born` – `float`, shape `(batch, b)` and `evaluated` – `float`, shape
 *   `(batch,)` – those of @ref CSBornChannelIntegrand, unweighted.
 * - real stream: `real_weight` – `float`, shape `(batch,)` and
 *   `dipole_weights` – `float`, shape `(batch, m)` – those of
 *   @ref CSRealChannelIntegrand.
 * - `momenta` – `float`, shape `(batch, n, 4)` – the kinematics.
 * - real stream with histograms: `born_momenta` – `float`, shape
 *   `(batch, m, n - 1, 4)` – the Borns of the dipoles, for the
 *   @ref CSObservableHistograms.
 * - `random` – `float`, shape `(batch, d)` – the inputs of the adaptive
 *   mapping.
 *
 * **Returns** (with @p madnis_training)
 * - `full_weight` – `float`, shape `(batch,)` – the weight of the point, the
 *   cross section times the Jacobian of the adaptive mapping.
 * - `weight` – `float`, shape `(batch,)` – the cross section.
 * - `latent` – `float`, shape `(batch, d)` – the random numbers of the
 *   channel integrand.
 * - `adaptive_prob` – `float`, shape `(batch,)` – the density of the
 *   adaptive mapping at the point.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 * - [2] S. Catani, M. H. Seymour, "A general algorithm for calculating jet
 *   cross sections in NLO QCD", https://arxiv.org/abs/hep-ph/9605323
 */
class CSIntegrand : public FunctionGenerator, public GeneratorIntegrand {
public:
    /// The optional adaptive reparametrization of the random numbers.
    using AdaptiveMapping = std::variant<std::monostate, VegasMapping, Flow>;
    /// The integrand of a channel of the n-body or the real stream.
    using ChannelIntegrand =
        std::variant<CSBornChannelIntegrand, CSRealChannelIntegrand>;

    /**
     * @param integrand             The integrand of the channel.
     * @param adaptive_map          The adaptive mapping of its random numbers,
     *                              none for uniform ones.
     * @param prefix                Prefix of the globals of the virtual trick.
     * @param virtual_fraction      The fraction of the points with the virtual
     *                              in the first iteration.
     * @param min_virtual_fraction  The smallest fraction.
     * @param histograms            The histograms the points fill, of the
     *                              stream of the integrand.
     * @param madnis_training       Whether the integrand gives the quantities
     *                              of the @ref MadnisLoss.
     * @throws std::invalid_argument if the adaptive mapping has another
     *         dimension than the random numbers, the flow takes a condition,
     *         the training has no @ref Flow, the fractions are not in (0, 1],
     *         or the histograms take what the integrand does not give (those
     *         of another stream, or with the training).
     */
    CSIntegrand(
        const ChannelIntegrand& integrand,
        const AdaptiveMapping& adaptive_map,
        const std::string& prefix,
        double virtual_fraction = 1.,
        double min_virtual_fraction = 0.005,
        const std::optional<CSObservableHistograms>& histograms = std::nullopt,
        bool madnis_training = false
    );
    /// Define the globals of the virtual trick on @p context (the fraction of
    /// the first iteration and zero ratios, for the n-body stream) and those
    /// of a VEGAS grid, uniform. The globals of a @ref Flow are left to it.
    void initialize_globals(ContextPtr context) const;
    /// The adaptive mapping.
    const AdaptiveMapping& adaptive_map() const { return _adaptive_map; }
    /// Whether the channel is one of the n-body stream.
    bool nbody() const { return _nbody; }
    /// Number of random numbers of the channel integrand.
    std::size_t random_dim() const { return _random_dim; }
    /// Global of the fraction of the virtual.
    const std::string& fraction_name() const { return _fraction_name; }
    /// Global of the ratios of the virtual.
    const std::string& ratio_name() const { return _ratio_name; }

    std::size_t particle_count() const override { return _particle_count; }
    const NamedVector<Type>& event_types() const override { return return_types(); }
    Function channel_function() const override;
    Function common_function() const override;
    Function concat_function() const override;
    std::optional<std::string> vegas_grid_name() const override;
    std::size_t vegas_dimension() const override;
    std::size_t vegas_bin_count() const override;
    std::vector<std::string> discrete_prob_names() const override { return {}; }
    std::vector<std::size_t> discrete_option_counts() const override { return {}; }
    std::array<bool, 3> provided_indices() const override {
        return {false, false, false};
    }
    bool madnis_training() const override { return _madnis_training; }
    /// The integrand itself.
    const FunctionGenerator& function_generator() const override { return *this; }
    /// A @ref CSIntegrandProbability of the integrand.
    std::shared_ptr<FunctionGenerator> probability() const override;
    /// None, the integrand has no flavors to drop.
    const std::vector<std::size_t>& active_flavors() const override;
    /// None, the integrand has no channel weights.
    const std::vector<me_int_t>& channel_indices() const override;
    bool unweighted() const override { return false; }
    /// The function of the histograms and the histograms it fills, none
    /// without histograms.
    std::optional<std::pair<Function, std::vector<Histogram>>>
    histograms() const override;
    /// The outputs of the integrand the histograms take.
    std::vector<std::string> histogram_inputs() const override;
    std::shared_ptr<IterationState> iteration_state() const override;

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;
    NamedVector<Value>
    build_channel_part(FunctionBuilder& fb, const NamedVector<Value>& args) const;
    NamedVector<Value>
    build_common_part(FunctionBuilder& fb, const NamedVector<Value>& args) const;

    ChannelIntegrand _integrand;
    AdaptiveMapping _adaptive_map;
    std::optional<CSObservableHistograms> _histograms;
    std::string _fraction_name, _ratio_name;
    double _virtual_fraction, _min_virtual_fraction;
    bool _madnis_training, _nbody;
    std::size_t _random_dim, _particle_count, _nlo_orders;
    NamedVector<Type> _channel_types;

    friend class CSIntegrandPart;
    friend class CSIntegrandProbability;
};

/**
 * The density of the adaptive mapping of a @ref CSIntegrand at a point, for
 * the @ref MadnisLoss of its training.
 *
 * `batch` is the leading batch dimension, `d` the number of random numbers of
 * the channel integrand.
 *
 * **Arguments**
 * - `latent` – `float`, shape `(batch, d)` – the random numbers of the
 *   channel integrand.
 *
 * **Returns**
 * - `prob` – `float`, shape `(batch,)` – the density of the adaptive mapping
 *   at `latent`, one without it.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 * - [2] T. Heimel et al., "MadNIS - Neural multi-channel importance sampling",
 *   https://arxiv.org/abs/2212.06172
 */
class CSIntegrandProbability : public FunctionGenerator {
public:
    /// @param integrand  The integrand to take the density of.
    CSIntegrandProbability(const CSIntegrand& integrand);

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    CSIntegrand::AdaptiveMapping _adaptive_map;
};

} // namespace madspace

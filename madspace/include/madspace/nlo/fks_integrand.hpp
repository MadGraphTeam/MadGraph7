#pragma once

#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "madspace/nlo/common.hpp"
#include "madspace/nlo/cuts.hpp"
#include "madspace/nlo/fks_histograms.hpp"
#include "madspace/nlo/fks_limits.hpp"
#include "madspace/nlo/fks_mapping.hpp"
#include "madspace/nlo/fks_subtraction.hpp"
#include "madspace/nlo/scale.hpp"
#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/custom_function.hpp"
#include "madspace/phasespace/discrete_flow.hpp"
#include "madspace/phasespace/flow.hpp"
#include "madspace/phasespace/integrand.hpp"
#include "madspace/phasespace/matrix_element.hpp"
#include "madspace/phasespace/vegas.hpp"

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
 *   is not zero;
 * - the Born of the extra counterterm of the configuration of the point
 *   (`squared_orders` and `spin_correlated` at the mother of the
 *   configuration), one library per extra counterterm, where the Born has a
 *   weight and one of the kinematics passes the cuts;
 * - with the scale of the user, the reference scale (`ren_scale` of a
 *   @ref CustomFunction of type scale) at the event where it has a weight, at
 *   the soft counter-event where it or the Born has one (the Born takes it,
 *   as MadFKS takes the scales of the Born at the soft counter-event) and at
 *   the collinear counter-event where it has one, zero elsewhere.
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
 * The Born inputs of @ref FKSCrossSection are the Born orders followed by
 * the orders of the Born of each extra counterterm, in the order of the
 * libraries, those of the extra counterterm of the configuration of the point
 * and zero for the others. The colour correlations and the virtual trick take
 * the Born orders alone.
 *
 * With tagged photons and QED corrections, the factors of the couplings of
 * the tagged photons multiply the real and the virtual per NLO order, the
 * Born and its correlated versions per Born input and the QED coupling of the
 * counterterms (tagged_factors of madgraph/mg7/nlo/provider.py).
 *
 * The Sudakov approximation of the EW corrections is not covered yet.
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
     * @param tagged_born     Those per Born input, none if empty.
     * @param tagged_alpha    That of the QED coupling.
     * @param reference_scale The reference scale of the user, a
     *                        @ref CustomFunction of type scale at the
     *                        kinematics of the real process, needed when the
     *                        scales take the scale of the user.
     * @param extras          The libraries of the Borns of the extra
     *                        counterterms: inputs and outputs of @p born
     *                        without `diagram_amp2_out` and
     *                        `color_correlated_out`.
     * @param extra_indices   Per FKS configuration its extra counterterm, an
     *                        index of @p extras, -1 for none. Empty for none.
     * @param extra_spin_legs Per FKS configuration with an extra counterterm
     *                        the Born leg of its mother, from 0, -1 for the
     *                        others. Empty for none.
     * @throws std::invalid_argument if the pieces do not fit together, the
     *         cross section takes the Sudakov approximation, the scales take
     *         the scale of the user without @p reference_scale, the extra
     *         counterterms do not give the Born inputs of the cross section, a
     *         table has another length than the configurations or the Born
     *         legs, or a leg has a colour representation without a Casimir
     *         here.
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
        double tagged_alpha = 1.,
        const std::optional<CustomFunction>& reference_scale = std::nullopt,
        const std::vector<MatrixElement>& extras = {},
        const std::vector<me_int_t>& extra_indices = {},
        const std::vector<me_int_t>& extra_spin_legs = {}
    );
    /// Number of FKS configurations.
    std::size_t configuration_count() const { return _real_indices.size(); }
    /// Number of Born orders.
    std::size_t born_order_count() const { return _born_orders; }
    /// Number of Born inputs of the cross section: the Born orders and those
    /// of the Borns of the extra counterterms.
    std::size_t born_input_count() const { return _born_inputs; }
    /// Number of NLO orders.
    std::size_t nlo_order_count() const { return _nlo_orders; }
    /// Number of particles of the Born.
    std::size_t born_particle_count() const { return _born_particles; }

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
    ColorMatrix _color;
    std::size_t _channel;
    double _g_ew;
    double _alpha;
    std::vector<double> _tagged_nlo, _tagged_born;
    double _tagged_alpha;
    std::optional<CustomFunction> _reference_scale;
    std::vector<MatrixElement> _extras;
    // per extra counterterm one where a configuration has it, zero elsewhere
    std::vector<std::vector<double>> _extra_tables;
    std::vector<me_int_t> _extra_spin_legs;
    std::size_t _born_orders, _nlo_orders, _born_particles, _born_inputs;
};

/**
 * What the iterations of an @ref FKSIntegrand adapt besides its VEGAS grid,
 * and the results of the iterations kept, as the fixed-order driver of
 * MadGraph7 keeps them per channel (Channel of madgraph/mg7/nlo/core.py).
 *
 * Per iteration of the channel:
 * - the choice of the FKS configuration takes the probabilities
 *   @f$p_k \propto \langle |w - w_n| \, p_k \rangle_k@f$ over the points of
 *   configuration @f$k@f$, mixed with the uniform choice by @p floor, as
 *   MC_integer of MadFKS, with @f$w_n@f$ the n-body part of the weight, which
 *   does not depend on the configuration (the root mean square of
 *   @f$w p_k@f$ let one large weight decide the probabilities, and its n-body
 *   part @f$w_n p_k@f$ kept them where they were);
 * - the virtual trick takes @f$r_o@f$ as the sum of the virtual of NLO order
 *   @f$o@f$ over the sum of the Born orders where the virtual is evaluated, and
 *   multiplies its fraction @f$f@f$ by @f$2 \sigma_V / \sigma@f$ clamped to
 *   @f$[1/4, 2]@f$, @f$f@f$ in @f$[f_\text{min}, 1]@f$, with @f$\sigma_V@f$
 *   and @f$\sigma@f$ the errors of the virtual residual and of the cross
 *   section in the iteration.
 *
 * The state starts from the probabilities, ratios and fraction in the globals
 * of the first context at its first iteration, so that it continues the
 * adaptation of an earlier state of the channel.
 *
 * The results are the sums of the cross section, its LO part, its virtual
 * residual, the scale and PDF variations and the modes of the Sudakov
 * approximation, in this order, over the points of the iterations since
 * @ref reset, each iteration with its own variance.
 */
class FKSIterationState : public IterationState {
public:
    /**
     * @param event_types           The outputs of the @ref FKSIntegrand.
     * @param probability_name      Global of the probabilities of the
     *                              configurations.
     * @param fraction_name         Global of the fraction of the virtual.
     * @param ratio_name            Global of the ratios of the virtual.
     * @param configuration_count   Number of FKS configurations.
     * @param virtual_fraction      The fraction of the first iteration.
     * @param min_virtual_fraction  The smallest fraction.
     * @param floor                 The weight of the uniform choice.
     * @param adapt_probabilities   Whether the iterations adapt the
     *                              probabilities of the configurations, not
     *                              when a @ref DiscreteFlow chooses them.
     */
    FKSIterationState(
        const NamedVector<Type>& event_types,
        const std::string& probability_name,
        const std::string& fraction_name,
        const std::string& ratio_name,
        std::size_t configuration_count,
        double virtual_fraction,
        double min_virtual_fraction,
        double floor,
        bool adapt_probabilities = true
    );
    /// Add the outputs of the @ref FKSIntegrand for one batch.
    void add(const TensorVec& events) override;
    /// Adapt to the iteration, keep its results and write the globals of
    /// @p contexts.
    void end_iteration(const std::vector<ContextPtr>& contexts) override;
    /// Discard the results of the iterations so far.
    void reset() override;
    /// The outputs of the @ref FKSIntegrand the state takes.
    const NamedVector<Type>& event_types() const { return _event_types; }
    /// Write the globals of the state into @p context.
    void write_globals(ContextPtr context) const;
    /// Take the probabilities, ratios and fraction from the globals of
    /// @p context.
    void read_globals(ContextPtr context);
    /// Number of points of the kept iterations.
    std::size_t count() const { return _count; }
    /// Number of sums: three, the scale and PDF variations and the modes.
    std::size_t value_count() const { return _sums.size(); }
    /// The estimate and its error of the linear combination of the sums with
    /// @p coefficients, missing ones zero.
    std::pair<double, double>
    combination(const std::vector<double>& coefficients) const;
    /// The fraction of the virtual trick.
    double virtual_fraction() const { return _fraction; }
    /// The ratios of the virtual trick, per NLO order.
    const std::vector<double>& virtual_ratios() const { return _ratios; }
    /// The probabilities of the FKS configurations.
    const std::vector<double>& configuration_probabilities() const {
        return _probabilities;
    }
    /// The standard deviation of the cross section in the last iteration.
    double iteration_std() const { return _iteration_std; }

private:
    NamedVector<Type> _event_types;
    std::string _probability_name, _fraction_name, _ratio_name;
    double _fraction, _min_fraction, _floor;
    bool _adapt_probabilities;
    // whether the state has taken the globals of the contexts
    bool _started = false;
    std::vector<double> _ratios, _probabilities;
    // the outputs per point summed for the results, (index, width)
    std::vector<std::pair<std::size_t, std::size_t>> _value_fields;
    std::size_t _virtual, _born, _evaluated, _configuration_weight, _nbody_weight;
    std::size_t _fks_index;
    // the kept iterations
    std::size_t _count = 0;
    std::vector<double> _sums;
    std::vector<double> _covariance;
    // the current iteration
    std::size_t _iteration_count = 0;
    std::vector<double> _iteration_sums, _iteration_products;
    double _residual_sum = 0., _residual_square_sum = 0.;
    std::vector<double> _virtual_sums;
    double _born_sum = 0.;
    std::vector<double> _choice_sums, _choice_counts;
    double _iteration_std = 0.;
};

/**
 * The FKS integrand of a channel as a @ref ChannelEventGenerator runs it at
 * fixed order (see @ref EventGenerator::integrate): @ref FKSChannelIntegrand
 * at random numbers through an optional @ref VegasMapping or @ref Flow, with
 * the FKS configuration and the points with the virtual drawn per point from
 * the probabilities, fraction and ratios that its @ref FKSIterationState
 * adapts. A @ref DiscreteFlow can choose the configuration instead, from the
 * random numbers of the kinematics and with the probabilities of the state as
 * its prior.
 *
 * With @p madnis_training, the integrand gives the quantities of the
 * @ref MadnisLoss instead, without the virtual (the virtual trick then takes
 * @f$r_o@f$ times the Born at all points). The integrand of the training in
 * the latent space of the adaptive samplers is
 * @f$f(x, k) = \sigma(x, k) \, p_k@f$, with @f$\sigma@f$ the cross section of
 * @ref FKSChannelIntegrand, whose real part is divided by the probability
 * @f$p_k@f$ of the configuration @f$k@f$ and whose n-body part is not.
 *
 * `batch` is the leading batch dimension, `d` the number of random numbers of
 * @ref FKSChannelIntegrand.
 *
 * **Arguments**
 * - `batch_size` – the number of points.
 *
 * **Returns** (without @p madnis_training)
 * - `weight` – `float`, shape `(batch,)` – the cross section with the virtual
 *   trick, times the Jacobian of the adaptive mapping, as all weights here.
 * - `lo_weight`, `virtual_residual` and, where the cross section gives them,
 *   `scale_variations`, `pdf_variations` and `ew_sudakov_weights` – those of
 *   @ref FKSChannelIntegrand.
 * - `virtual`, `born`, `evaluated`, `configuration_weight` – those of
 *   @ref FKSChannelIntegrand, the last the inverse probability of the
 *   configuration.
 * - `momenta`, `event_weight`, `soft_momenta`, `soft_weight`,
 *   `collinear_momenta`, `collinear_weight`, `soft_collinear_momenta`,
 *   `soft_collinear_weight`, `born_momenta`, `nbody_weight` – the kinematics
 *   and their weights for the histograms.
 * - `fks_index` – `int`, shape `(batch,)` – the FKS configuration.
 * - `random` – `float`, shape `(batch, d)` – the inputs of the adaptive
 *   mapping.
 *
 * **Returns** (with @p madnis_training)
 * - `full_weight` – `float`, shape `(batch,)` – the weight of the point, the
 *   cross section times the Jacobian of the adaptive mapping.
 * - `weight` – `float`, shape `(batch,)` – @f$f(x, k)@f$.
 * - `latent` – `float`, shape `(batch, d)` – @f$x@f$, the random numbers of
 *   @ref FKSChannelIntegrand.
 * - `adaptive_prob` – `float`, shape `(batch,)` – the density of the adaptive
 *   samplers at the point, the density of the adaptive mapping times
 *   @f$p_k@f$.
 * - `fks_index` – `int`, shape `(batch,)` – @f$k@f$.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 * - [2] R. Frederix, S. Frixione, F. Maltoni, T. Stelzer, "Automation of
 *   next-to-leading order computations in QCD: the FKS subtraction",
 *   https://arxiv.org/abs/0908.4272
 */
class FKSIntegrand : public FunctionGenerator, public GeneratorIntegrand {
public:
    /// The optional adaptive reparametrization of the random numbers.
    using AdaptiveMapping = std::variant<std::monostate, VegasMapping, Flow>;
    /// The optional adaptive choice of the FKS configuration, by default the
    /// probabilities of the state.
    using AdaptiveDiscrete = std::variant<std::monostate, DiscreteFlow>;

    /**
     * @param integrand             The integrand of the channel.
     * @param adaptive_map          The adaptive mapping of its random numbers,
     *                              none for uniform ones.
     * @param prefix                Prefix of the globals of the state.
     * @param virtual_fraction      The fraction of the points with the virtual
     *                              in the first iteration.
     * @param min_virtual_fraction  The smallest fraction.
     * @param histograms            The histograms the points fill.
     * @param floor                 The weight of the uniform choice of the FKS
     *                              configuration.
     * @param discrete              The adaptive choice of the configuration,
     *                              a @ref DiscreteFlow with one dimension of
     *                              the configurations, conditioned on the
     *                              random numbers of the kinematics and with
     *                              a prior.
     * @param madnis_training       Whether the integrand gives the quantities
     *                              of the @ref MadnisLoss.
     * @throws std::invalid_argument if the adaptive mapping has another
     *         dimension than the random numbers, the discrete flow does not
     *         fit, the training has no @ref Flow, or the histograms are those
     *         of other particles.
     */
    FKSIntegrand(
        const FKSChannelIntegrand& integrand,
        const AdaptiveMapping& adaptive_map,
        const std::string& prefix,
        double virtual_fraction = 1.,
        double min_virtual_fraction = 0.005,
        const std::optional<FKSObservableHistograms>& histograms = std::nullopt,
        double floor = 0.1,
        const AdaptiveDiscrete& discrete = std::monostate{},
        bool madnis_training = false
    );
    /// Define the globals of the state on @p context and those of a VEGAS
    /// grid: the grid uniform, the configurations equally likely, the
    /// fraction of the first iteration and zero ratios. The globals of a
    /// @ref Flow and a @ref DiscreteFlow are left to them.
    void initialize_globals(ContextPtr context) const;
    /// The adaptive mapping.
    const AdaptiveMapping& adaptive_map() const { return _adaptive_map; }
    /// The adaptive choice of the configuration.
    const AdaptiveDiscrete& discrete() const { return _discrete; }
    /// Number of FKS configurations.
    std::size_t configuration_count() const { return _configuration_count; }
    /// Number of random numbers of the kinematics.
    std::size_t random_dim() const { return _random_dim; }
    /// Global of the probabilities of the configurations.
    const std::string& probability_name() const { return _probability_name; }
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
    /// An @ref FKSIntegrandProbability of the integrand.
    std::shared_ptr<FunctionGenerator> probability() const override;
    /// None, the integrand has no flavours to drop.
    const std::vector<std::size_t>& active_flavors() const override;
    /// None, the integrand has no channel weights.
    const std::vector<me_int_t>& channel_indices() const override;
    bool unweighted() const override { return false; }
    std::optional<std::pair<Function, std::vector<Histogram>>>
    histograms() const override;
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

    FKSChannelIntegrand _integrand;
    AdaptiveMapping _adaptive_map;
    AdaptiveDiscrete _discrete;
    std::optional<FKSObservableHistograms> _histograms;
    std::string _probability_name, _fraction_name, _ratio_name;
    double _virtual_fraction, _min_virtual_fraction, _floor;
    bool _madnis_training;
    std::size_t _random_dim, _particle_count, _configuration_count, _nlo_orders;
    NamedVector<Type> _channel_types;

    friend class FKSIntegrandPart;
    friend class FKSIntegrandProbability;
};

/**
 * The density of the adaptive samplers of an @ref FKSIntegrand at a point, for
 * the @ref MadnisLoss of its training.
 *
 * `batch` is the leading batch dimension, `d` the number of random numbers of
 * @ref FKSChannelIntegrand.
 *
 * **Arguments**
 * - `latent` – `float`, shape `(batch, d)` – the random numbers of
 *   @ref FKSChannelIntegrand.
 * - `fks_index` – `int`, shape `(batch,)` – the FKS configuration.
 *
 * **Returns**
 * - `prob` – `float`, shape `(batch,)` – the density of the adaptive mapping
 *   at `latent` times the probability of the configuration, from the
 *   @ref DiscreteFlow or the probabilities of the state.
 */
class FKSIntegrandProbability : public FunctionGenerator {
public:
    /// @param integrand  The integrand to take the density of.
    FKSIntegrandProbability(const FKSIntegrand& integrand);

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    FKSIntegrand::AdaptiveMapping _adaptive_map;
    FKSIntegrand::AdaptiveDiscrete _discrete;
    std::string _probability_name;
    std::size_t _configuration_count;
};

} // namespace madspace

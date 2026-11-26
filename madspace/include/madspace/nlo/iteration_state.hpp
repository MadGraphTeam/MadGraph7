#pragma once

#include <string>
#include <utility>
#include <vector>

#include "madspace/phasespace/integrand.hpp"

namespace madspace {

/**
 * What the iterations of a fixed-order NLO integrand (@ref FKSIntegrand,
 * @ref CSIntegrand) adapt besides its VEGAS grid, and the results of the
 * iterations kept, as the fixed-order driver of MadGraph7 keeps them per
 * channel (Channel of madgraph/mg7/nlo/core.py).
 *
 * Per iteration of the channel:
 * - with the output `fks_index`, the choice of the FKS configuration takes
 *   the probabilities @f$p_k \propto \langle |w - w_n| \, p_k \rangle_k@f$
 *   over the points of configuration @f$k@f$, mixed with the uniform choice by
 *   @p floor, as MC_integer of MadFKS, with @f$w_n@f$ the n-body part of the
 *   weight, which does not depend on the configuration (the root mean square
 *   of @f$w p_k@f$ let one large weight decide the probabilities, and its
 *   n-body part @f$w_n p_k@f$ kept them where they were);
 * - with the output `virtual`, the virtual trick takes @f$r_o@f$ as the sum of
 *   the virtual of NLO order @f$o@f$ over the sum of the Born orders where the
 *   virtual is evaluated, and multiplies its fraction @f$f@f$ by
 *   @f$2 \sigma_V / \sigma@f$ clamped to @f$[1/4, 2]@f$, @f$f@f$ in
 *   @f$[f_\text{min}, 1]@f$, with @f$\sigma_V@f$ and @f$\sigma@f$ the errors of
 *   the virtual residual and of the cross section in the iteration.
 *
 * The state starts from the probabilities, ratios and fraction in the globals
 * of the first context at its first iteration, so that it continues the
 * adaptation of an earlier state of the channel. Without the outputs of a
 * part, the state has neither that part nor its globals.
 *
 * The results are the sums of those of the cross section, its LO part, its
 * virtual residual, the scale and PDF variations and the modes of the Sudakov
 * approximation that the integrand gives, in this order, over the points of
 * the iterations since @ref reset, each iteration with its own variance.
 */
class NLOIterationState : public IterationState {
public:
    /**
     * @param event_types           The outputs of the integrand: `weight` and
     *                              the other sums it gives, with
     *                              `fks_index`, `configuration_weight` and
     *                              `nbody_weight` for the choice of the FKS
     *                              configuration, with `virtual`, `born`,
     *                              `evaluated` and `virtual_residual` for the
     *                              virtual trick.
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
     * @throws std::invalid_argument if the outputs have no `weight` or only
     *         some outputs of a part.
     */
    NLOIterationState(
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
    /// Add the outputs of the integrand for one batch.
    void add(const TensorVec& events) override;
    /// Adapt to the iteration, keep its results and write the globals of
    /// @p contexts.
    void end_iteration(const std::vector<ContextPtr>& contexts) override;
    /// Discard the results of the iterations so far.
    void reset() override;
    /// The outputs of the integrand the state takes.
    const NamedVector<Type>& event_types() const { return _event_types; }
    /// Write the globals of the state into @p context.
    void write_globals(ContextPtr context) const;
    /// Take the probabilities, ratios and fraction from the globals of
    /// @p context.
    void read_globals(ContextPtr context);
    /// Number of points of the kept iterations.
    std::size_t count() const { return _count; }
    /// Number of sums, those the integrand gives.
    std::size_t value_count() const { return _sums.size(); }
    /// The estimate and its error of the linear combination of the sums with
    /// @p coefficients, missing ones zero.
    std::pair<double, double>
    combination(const std::vector<double>& coefficients) const;
    /// Whether the state has the virtual trick.
    bool has_virtual() const { return _has_virtual; }
    /// Whether the state has the choice of the FKS configuration.
    bool has_choice() const { return _has_choice; }
    /// The fraction of the virtual trick.
    double virtual_fraction() const { return _fraction; }
    /// The ratios of the virtual trick, per NLO order, none without it.
    const std::vector<double>& virtual_ratios() const { return _ratios; }
    /// The probabilities of the FKS configurations, none without the choice.
    const std::vector<double>& configuration_probabilities() const {
        return _probabilities;
    }
    /// The standard deviation of the cross section in the last iteration.
    double iteration_std() const { return _iteration_std; }

private:
    NamedVector<Type> _event_types;
    std::string _probability_name, _fraction_name, _ratio_name;
    double _fraction, _min_fraction, _floor;
    bool _adapt_probabilities, _has_virtual, _has_choice;
    // whether the state has taken the globals of the contexts
    bool _started = false;
    std::vector<double> _ratios, _probabilities;
    // the outputs per point summed for the results, (index, width)
    std::vector<std::pair<std::size_t, std::size_t>> _value_fields;
    // the field of the virtual residual among those of the sums
    std::size_t _residual = 0;
    std::size_t _virtual = 0, _born = 0, _evaluated = 0;
    std::size_t _configuration_weight = 0, _nbody_weight = 0, _fks_index = 0;
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

} // namespace madspace

#pragma once

#include "madspace/compgraphs.hpp"
#include "madspace/driver/context.hpp"
#include "madspace/driver/tensor.hpp"

namespace madspace {

/**
 * Refines the categorical probabilities of a @ref DiscreteSampler or @ref
 * DiscreteFlow from accumulated per-category statistics.
 *
 * Counterpart of @ref DiscreteHistogram: @ref add_data feeds in the
 * per-category weight sums and counts it accumulates, and @ref optimize
 * writes the probability tensor of every `prob_names` global to every context
 * in `contexts`.
 *
 * The weights of the events of category @f$i@f$, sampled with probability
 * @f$p_i@f$, average to @f$I_i / p_i@f$, with @f$I_i@f$ the contribution of
 * the category to the integral. @ref optimize takes the new probabilities as
 * the running average of the old ones and of @f$|I_i| / \sum_j |I_j|@f$,
 * weighted by the number of events of the update against those of the
 * updates before, which start at 7000. A category without events keeps a
 * share of its old probability, so that it is never dropped.
 */
class DiscreteOptimizer {
public:
    /**
     * @param contexts   Contexts sharing the probabilities to refine.
     * @param prob_names Global names of the probability tensors, one per
     *                   discrete dimension.
     */
    DiscreteOptimizer(
        const std::vector<ContextPtr>& contexts,
        const std::vector<std::string>& prob_names
    ) :
        _contexts(contexts), _prob_names(prob_names), _sample_count(7000) {}
    /// Accumulate one batch of `values` and `counts` for each dimension of
    /// `prob_names`, interleaved as `[values_0, counts_0, values_1, ...]`.
    void add_data(const std::vector<Tensor>& values_and_counts);
    /// Update the probabilities from the accumulated data, write them back to
    /// every context and reset the accumulated data. Without data the
    /// probabilities are left as they are.
    void optimize();

private:
    std::vector<ContextPtr> _contexts;
    std::vector<std::string> _prob_names;
    std::size_t _sample_count;
    // per discrete dimension the weight sums and counts of its categories
    std::vector<std::tuple<std::vector<double>, std::vector<std::size_t>>> _data;
};

} // namespace madspace

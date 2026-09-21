#pragma once

#include "madspace/compgraphs.hpp"
#include "madspace/driver/context.hpp"
#include "madspace/driver/tensor.hpp"

namespace madspace {

/**
 * Refines the categorical probabilities of a @ref DiscreteSampler or @ref
 * DiscreteFlow from accumulated per-category statistics.
 *
 * The training-time counterpart of @ref DiscreteHistogram — @ref add_data
 * feeds in the per-category weight sums accumulated during a warm-up run, and
 * @ref optimize rewrites every `prob_names` global on every context in
 * `contexts` with the refined distribution.
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
    /// Refine the probabilities from the accumulated data and write them
    /// back to every context.
    void optimize();

private:
    std::vector<ContextPtr> _contexts;
    std::vector<std::string> _prob_names;
    double _damping;
    std::size_t _sample_count;
    std::vector<std::tuple<std::vector<std::size_t>, std::vector<double>>> _data;
};

} // namespace madspace

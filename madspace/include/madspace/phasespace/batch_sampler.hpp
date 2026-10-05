#pragma once

#include "madspace/phasespace/base.hpp"

namespace madspace {

/**
 * Concatenates the per-channel tensors of a multi-channel batch.
 *
 * Given one tensor set per channel and the per-channel batch sizes, it packs
 * them into a single contiguous batch (the inverse of the split done inside
 * @ref MultiChannelMapping). Used to assemble a mixed batch for the shared
 * integrand evaluation (see Sec. 3.1 of [1]).
 *
 * `batch` is the leading batch dimension. `i` indexes the channels.
 *
 * **Arguments**
 * - `channel<i>_in_<key>` – the per-channel input tensors.
 * - `batch_sizes` – the number of entries in each channel.
 *
 * **Returns**
 * - `channel<i>_out_<key>` – the concatenated tensors.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 */
class BatchSampler : public FunctionGenerator {
public:
    /// @param types  One tensor-type set per channel.
    BatchSampler(const std::vector<NamedVector<Type>>& types);

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<std::size_t> _channel_tensor_counts;
};

} // namespace madspace

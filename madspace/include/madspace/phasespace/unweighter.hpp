#pragma once

#include "madspace/phasespace/base.hpp"

namespace madspace {

/**
 * Rejection-samples a weighted batch into an unweighted one.
 *
 * Keeps each event with probability `weight / max_weight` and returns the
 * surviving rows (Sec. 3.4 of [1]). The first argument and return value are the
 * weights; any further tensors listed in @p types are carried along and
 * gathered for the surviving events.
 *
 * `batch` is the leading batch dimension.
 *
 * **Arguments**
 * - the tensors named in @p types (the first is the event weight), plus
 * - `max_weight` – `float`, scalar – the weight to unweight against.
 *
 * **Returns**
 * - the tensors named in @p types, restricted to the accepted events.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 3.4)
 */
class Unweighter : public FunctionGenerator {
public:
    /// @param types  The per-event tensors, the first being the weight.
    Unweighter(const NamedVector<Type>& types);

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;
};

/**
 * Partial unweighting against a running maximum weight.
 *
 * Used for the intermediate channel-wise event files: events are only
 * partially unweighted while generation continues, using the current estimate
 * of the per-channel maximum weight, and a final @ref Unweighter pass is run
 * once the target count is reached (Sec. 3.4.1 of [1]). @p quantile sets the
 * fraction of over-weight events allowed when estimating the maximum.
 *
 * `batch` is the leading batch dimension.
 *
 * **Arguments**
 * - the tensors named in @p types (the first is the event weight).
 *
 * **Returns**
 * - the tensors named in @p types, partially unweighted.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 3.4.1)
 */
class BufferUnweighter : public FunctionGenerator {
public:
    /// @param types     The per-event tensors, the first being the weight.
    /// @param quantile  Allowed fraction of over-weight events.
    BufferUnweighter(const NamedVector<Type>& types, double quantile = 0.0);

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    double _quantile;
};

} // namespace madspace

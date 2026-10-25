#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/topology.hpp"
#include "madspace/util.hpp"

namespace madspace {

/**
 * Local multi-channel weights from propagator denominators.
 *
 * Implements the single-diagram-enhanced (SDE) multi-channel weights
 * @f$\alpha_i(x)@f$ of Eq. (2.7) of [1]: each channel weight is proportional to
 * the product of the Breit-Wigner propagator denominators of its topology,
 * evaluated at the current phase-space point [2, 3]. Only the forward mappings
 * are needed, not their inverses (Sec. 2.1.1 of [1]).
 *
 * `batch` is the leading batch dimension. `c` indexes the channels.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, n, 4)` – all momenta of the event.
 *
 * **Returns**
 * - `channel_weights` – `float`, shape `(batch, c)` – the normalized
 *   per-channel weights @f$\alpha_i(x)@f$.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 2.1.1)
 * - [2] F. Maltoni, T. Stelzer, "MadEvent: automatic event generation with
 *   MadGraph", https://arxiv.org/abs/hep-ph/0208156
 * - [3] O. Mattelaer, K. Ostrolenk, "Speeding up MadGraph5_aMC@NLO",
 *   https://arxiv.org/abs/2102.00773
 */
class PropagatorChannelWeights : public FunctionGenerator {
public:
    /**
     * @param topologies      One @ref Topology per channel.
     * @param permutations    Per channel, the outgoing-particle permutations to
     *                        average the weight over.
     * @param channel_indices Per channel, the global channel indices its weight
     *                        contributes to.
     */
    PropagatorChannelWeights(
        const std::vector<Topology>& topologies,
        const nested_vector3<std::size_t>& permutations,
        const nested_vector2<std::size_t>& channel_indices
    );

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    nested_vector2<double> _momentum_factors;
    nested_vector2<me_int_t> _invariant_indices;
    nested_vector2<double> _masses;
    nested_vector2<double> _widths;
};

/**
 * Splits each channel weight among its on-shell sub-channels.
 *
 * A single decay topology admits several sampling orders, one per set of
 * propagators that can be simultaneously on shell (Sec. 2.2.7 of [1]). This
 * distributes an incoming per-channel weight over those sub-channels in
 * proportion to their on-shell propagator denominators.
 *
 * `batch` is the leading batch dimension.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, n, 4)` – all momenta of the event.
 * - `channel_weights_in` – `float`, shape `(batch, c_in)` – the per-channel
 *   weights to split.
 *
 * **Returns**
 * - `channel_weights_out` – `float`, shape `(batch, c_out)` – the per-sub-channel
 *   weights.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 2.2.7)
 */
class SubchannelWeights : public FunctionGenerator {
public:
    /**
     * @param topologies      Per channel, its list of on-shell sub-channel
     *                        @ref Topology "topologies".
     * @param permutations    Per channel, the outgoing-particle permutations.
     * @param channel_indices Per channel, the global channel indices.
     */
    SubchannelWeights(
        const nested_vector2<Topology>& topologies,
        const nested_vector3<std::size_t>& permutations,
        const nested_vector2<std::size_t>& channel_indices
    );

    /// Number of input channels.
    std::size_t channel_count() const { return _channel_indices.size(); }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    nested_vector2<double> _momentum_factors;
    nested_vector2<double> _masses;
    nested_vector2<double> _widths;
    nested_vector2<me_int_t> _invariant_indices;
    nested_vector2<me_int_t> _on_shell;
    std::vector<me_int_t> _group_sizes;
    std::vector<me_int_t> _channel_indices;
    std::vector<me_int_t> _subchannel_indices;
};

} // namespace madspace

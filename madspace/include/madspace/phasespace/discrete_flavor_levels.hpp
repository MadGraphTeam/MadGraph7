#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/mlp.hpp"

namespace madspace {

/**
 * Learned flavour sampling in three levels: the initial state @f$i@f$, the
 * final-state content @f$c@f$ of the flavours of that initial state, and the
 * flavour @f$f@f$ of that content,
 * @f$q(f) = q_A(i)\, q_B(c \mid i)\, q_C(f \mid c)@f$.
 *
 * It replaces a one-dimensional @ref DiscreteFlow over all flavours, with the
 * same inputs, conditions and outputs. Its priors are those of the flavour
 * prior: @f$q_A@f$ starts from the prior summed per initial state, @f$q_B@f$
 * and @f$q_C@f$ from the constant weights of the flavours (their factors, the
 * prior within an initial state having the same parton densities). @f$q_A@f$
 * and @f$q_B@f$ multiply their priors by the softmax of an @ref MLP of the
 * conditioning input (and the initial state, for @f$q_B@f$), mixed with the
 * plain prior as @f$(1-\epsilon) q + \epsilon q_\mathrm{prior}@f$, so that no
 * flavour gets less than @f$\epsilon@f$ times its prior. @f$q_C@f$ is not
 * learned. The levels are sampled with one random number, rescaled to the
 * chosen category after each level.
 *
 * `batch` is the leading batch dimension, `f` indexes the flavours.
 *
 * **Inputs**
 * - `random_0` – `float`, shape `(batch,)` – one uniform number.
 *
 * **Conditions**
 * - `condition` – `float`, shape `(batch, condition_dim)` – conditioning input.
 * - `prior_0` – `float`, shape `(batch, f)` – the flavour prior.
 *
 * **Outputs**
 * - `index_0` – `int`, shape `(batch,)` – the chosen flavour.
 *
 * In addition every mapping returns a `weight` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 */
class DiscreteFlavorLevels : public Mapping {
public:
    /**
     * @param flavor_initials    The initial state of each flavour, from 0.
     * @param flavor_contents    The final-state content of each flavour, from 0;
     *                           the flavours of a content share their initial
     *                           state.
     * @param flavor_weights     The constant weight of each flavour, the ratio
     *                           of its prior to that of the others of its
     *                           initial state.
     * @param prefix             Prefix for the trainable global names.
     * @param condition_dim      Width of the conditioning input, nonzero.
     * @param prior_floor        @f$\epsilon@f$, in [0, 1).
     * @param logit_bound        c > 0 bounds the learned logits to c tanh(x / c),
     *                           0 leaves them unbounded.
     * @param subnet_hidden_dim  Hidden width of the @ref MLP subnetworks.
     * @param subnet_layers      Number of layers in the subnetworks.
     * @param subnet_activation  Activation of the subnetworks.
     * @param subnet_precision   Float type of the subnetworks (see @ref MLP).
     */
    DiscreteFlavorLevels(
        const std::vector<me_int_t>& flavor_initials,
        const std::vector<me_int_t>& flavor_contents,
        const std::vector<double>& flavor_weights,
        const std::string& prefix,
        std::size_t condition_dim,
        double prior_floor = 0.,
        double logit_bound = 0.,
        std::size_t subnet_hidden_dim = 32,
        std::size_t subnet_layers = 3,
        MLP::Activation subnet_activation = MLP::leaky_relu,
        DataType subnet_precision = DataType::dt_float64
    );

    /// Number of flavours.
    std::size_t flavor_count() const { return _flavor_initials.size(); }
    /// Number of initial states.
    std::size_t initial_count() const { return _initial_count; }
    /// Number of final-state contents.
    std::size_t content_count() const { return _content_count; }
    /// Width of the conditioning input.
    std::size_t condition_dim() const { return _condition_dim; }

    /**
     * @param context  Context receiving the globals.
     * @param seed     Seed for the random initialization.
     */
    void initialize_globals(
        ContextPtr context, std::optional<std::uint64_t> seed = std::nullopt
    ) const;

private:
    Result build_forward_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override;
    Value
    mixed(FunctionBuilder& fb, Value logits, Value prior, std::size_t count) const;

    std::vector<me_int_t> _flavor_initials;
    std::vector<me_int_t> _flavor_contents;
    std::vector<me_int_t> _flavor_slots;
    std::size_t _initial_count;
    std::size_t _content_count;
    std::size_t _condition_dim;
    double _prior_floor;
    double _logit_bound;
    // per initial state, the prior of each content; per content, of each slot
    nested_vector2<double> _content_priors;
    nested_vector2<double> _slot_priors;
    // content c, slot k: flavour _slot_flavors[_slot_offsets[c] + k]
    std::vector<me_int_t> _slot_offsets;
    std::vector<me_int_t> _slot_flavors;
    MLP _initial_net;
    MLP _content_net;
};

} // namespace madspace

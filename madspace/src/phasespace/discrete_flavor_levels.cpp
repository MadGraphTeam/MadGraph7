#include "madspace/phasespace/discrete_flavor_levels.hpp"

#include "madspace/util.hpp"

#include <algorithm>
#include <stdexcept>

using namespace madspace;

namespace {

std::size_t checked_condition_dim(std::size_t condition_dim) {
    if (condition_dim == 0) {
        throw std::invalid_argument("DiscreteFlavorLevels needs a condition");
    }
    return condition_dim;
}

std::size_t index_count(const std::vector<me_int_t>& indices, const char* name) {
    me_int_t max_index = -1;
    for (me_int_t index : indices) {
        if (index < 0) {
            throw std::invalid_argument(
                std::format("DiscreteFlavorLevels: negative {} index", name)
            );
        }
        max_index = std::max(max_index, index);
    }
    return static_cast<std::size_t>(max_index + 1);
}

// The random number of a chosen category rescaled to [0, 1) within it.
Value rescaled(FunctionBuilder& fb, Value random, Value index, Value probs) {
    auto [middle, prob] = fb.sample_discrete_probs_inverse(index, probs);
    Value lower = fb.sub(middle, fb.mul(prob, 0.5));
    Value result = fb.div(fb.sub(random, lower), prob);
    return fb.min(fb.max(result, Value(0.)), Value(1. - 1e-15));
}

} // namespace

DiscreteFlavorLevels::DiscreteFlavorLevels(
    const std::vector<me_int_t>& flavor_initials,
    const std::vector<me_int_t>& flavor_contents,
    const std::vector<double>& flavor_weights,
    const std::string& prefix,
    std::size_t condition_dim,
    double prior_floor,
    std::size_t subnet_hidden_dim,
    std::size_t subnet_layers,
    MLP::Activation subnet_activation,
    DataType subnet_precision
) :
    Mapping(
        "DiscreteFlavorLevels",
        {{"random_0", batch_float}},
        {{"index_0", batch_int}},
        {{"condition", batch_float_array(condition_dim)},
         {"prior_0", batch_float_array(flavor_initials.size())}}
    ),
    _flavor_initials(flavor_initials),
    _flavor_contents(flavor_contents),
    _initial_count(index_count(flavor_initials, "initial-state")),
    _content_count(index_count(flavor_contents, "content")),
    _condition_dim(checked_condition_dim(condition_dim)),
    _prior_floor(prior_floor),
    _initial_net(
        condition_dim,
        _initial_count,
        subnet_hidden_dim,
        subnet_layers,
        subnet_activation,
        prefixed_name(prefix, "initial_"),
        subnet_precision
    ),
    _content_net(
        condition_dim + _initial_count,
        _content_count,
        subnet_hidden_dim,
        subnet_layers,
        subnet_activation,
        prefixed_name(prefix, "content_"),
        subnet_precision
    ) {
    std::size_t flavor_count = flavor_initials.size();
    if (flavor_contents.size() != flavor_count ||
        flavor_weights.size() != flavor_count) {
        throw std::invalid_argument(
            "DiscreteFlavorLevels: one initial state, content and weight per flavour"
        );
    }
    if (prior_floor < 0. || prior_floor >= 1.) {
        throw std::invalid_argument("DiscreteFlavorLevels: prior_floor in [0, 1)");
    }
    std::vector<me_int_t> content_initials(_content_count, -1);
    std::vector<std::vector<me_int_t>> content_flavors(_content_count);
    std::vector<double> content_weights(_content_count, 0.);
    std::vector<double> initial_weights(_initial_count, 0.);
    for (std::size_t f = 0; f < flavor_count; ++f) {
        if (!(flavor_weights[f] > 0.)) {
            throw std::invalid_argument("DiscreteFlavorLevels: weights must be positive");
        }
        auto& initial = content_initials.at(flavor_contents[f]);
        if (initial != -1 && initial != flavor_initials[f]) {
            throw std::invalid_argument(
                "DiscreteFlavorLevels: the flavours of a content must share their "
                "initial state"
            );
        }
        initial = flavor_initials[f];
        content_flavors.at(flavor_contents[f]).push_back(static_cast<me_int_t>(f));
        content_weights.at(flavor_contents[f]) += flavor_weights[f];
        initial_weights.at(flavor_initials[f]) += flavor_weights[f];
    }
    std::size_t slot_count = 0;
    for (std::size_t c = 0; c < _content_count; ++c) {
        if (content_flavors[c].empty()) {
            throw std::invalid_argument("DiscreteFlavorLevels: a content has no flavour");
        }
        slot_count = std::max(slot_count, content_flavors[c].size());
    }
    _content_priors.assign(_initial_count, std::vector<double>(_content_count, 0.));
    for (std::size_t c = 0; c < _content_count; ++c) {
        auto i = content_initials[c];
        _content_priors[i][c] = content_weights[c] / initial_weights[i];
    }
    _flavor_slots.assign(flavor_count, 0);
    _slot_priors.assign(_content_count, std::vector<double>(slot_count, 0.));
    _slot_flavors.assign(_content_count * slot_count, 0);
    for (std::size_t c = 0; c < _content_count; ++c) {
        _slot_offsets.push_back(static_cast<me_int_t>(c * slot_count));
        for (std::size_t k = 0; k < content_flavors[c].size(); ++k) {
            auto f = content_flavors[c][k];
            _flavor_slots[f] = static_cast<me_int_t>(k);
            _slot_priors[c][k] = flavor_weights[f] / content_weights[c];
            _slot_flavors[c * slot_count + k] = f;
        }
    }
}

Value DiscreteFlavorLevels::mixed(
    FunctionBuilder& fb, Value logits, Value prior, std::size_t count
) const {
    Value learned = fb.softmax_prior(logits, prior);
    if (_prior_floor == 0.) {
        return learned;
    }
    // constant arrays: mul does not broadcast a scalar over the options
    auto batch_size = fb.batch_size({logits});
    auto full = [&](double value) {
        return fb.full({value, batch_size, static_cast<me_int_t>(count)});
    };
    Value plain = fb.softmax_prior(full(0.), prior);
    return fb.add(
        fb.mul(learned, full(1. - _prior_floor)), fb.mul(plain, full(_prior_floor))
    );
}

Mapping::Result DiscreteFlavorLevels::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    Value condition = conditions.at(0);
    // normalised per initial state, uniform if the prior vanishes
    Value initial_prior = fb.collect_channel_weights(
        conditions.at(1), _flavor_initials, static_cast<me_int_t>(_initial_count)
    );
    Value initial_probs = mixed(
        fb,
        _initial_net.build_function(fb, {condition}).at(0),
        initial_prior,
        _initial_count
    );
    Value random = inputs.at(0);
    auto [initial, initial_det] = fb.sample_discrete_probs(random, initial_probs);
    random = rescaled(fb, random, initial, initial_probs);

    Value content_input = fb.cat(
        {condition, fb.one_hot(initial, static_cast<me_int_t>(_initial_count))}
    );
    Value content_probs = mixed(
        fb,
        _content_net.build_function(fb, {content_input}).at(0),
        fb.gather_vector(initial, _content_priors),
        _content_count
    );
    auto [content, content_det] = fb.sample_discrete_probs(random, content_probs);
    random = rescaled(fb, random, content, content_probs);

    Value slot_probs = fb.gather_vector(content, _slot_priors);
    auto [slot, slot_det] = fb.sample_discrete_probs(random, slot_probs);
    Value flavor = fb.gather_int(
        fb.add_int(fb.gather_int(content, _slot_offsets), slot), _slot_flavors
    );
    return {
        {output_types().keys(), {flavor}},
        fb.product({initial_det, content_det, slot_det})
    };
}

Mapping::Result DiscreteFlavorLevels::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    Value condition = conditions.at(0);
    Value flavor = inputs.at(0);
    Value initial = fb.gather_int(flavor, _flavor_initials);
    Value content = fb.gather_int(flavor, _flavor_contents);
    Value slot = fb.gather_int(flavor, _flavor_slots);
    Value initial_prior = fb.collect_channel_weights(
        conditions.at(1), _flavor_initials, static_cast<me_int_t>(_initial_count)
    );
    Value initial_probs = mixed(
        fb,
        _initial_net.build_function(fb, {condition}).at(0),
        initial_prior,
        _initial_count
    );
    auto [random, initial_prob] = fb.sample_discrete_probs_inverse(initial, initial_probs);
    Value content_input = fb.cat(
        {condition, fb.one_hot(initial, static_cast<me_int_t>(_initial_count))}
    );
    Value content_probs = mixed(
        fb,
        _content_net.build_function(fb, {content_input}).at(0),
        fb.gather_vector(initial, _content_priors),
        _content_count
    );
    Value content_prob = fb.sample_discrete_probs_inverse(content, content_probs)[1];
    Value slot_prob =
        fb.sample_discrete_probs_inverse(slot, fb.gather_vector(content, _slot_priors))[1];
    return {
        {input_types().keys(), {random}},
        fb.product({initial_prob, content_prob, slot_prob})
    };
}

void DiscreteFlavorLevels::initialize_globals(
    ContextPtr context, std::optional<std::uint64_t> seed
) const {
    _initial_net.initialize_globals(context, seed);
    _content_net.initialize_globals(context, seed);
}

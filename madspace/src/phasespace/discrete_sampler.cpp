#include "madspace/phasespace/discrete_sampler.h"

using namespace madspace;

DiscreteHistogram::DiscreteHistogram(const std::vector<std::size_t>& option_counts) :
    FunctionGenerator(
        "DiscreteHistogram",
        [&] {
            TypeVec arg_types(option_counts.size(), batch_int);
            arg_types.push_back(batch_float);
            return arg_types;
        }(),
        [&] {
            TypeVec ret_types;
            for (std::size_t option_count : option_counts) {
                ret_types.push_back(single_float_array(option_count));
                ret_types.push_back(single_int_array(option_count));
            }
            return ret_types;
        }()
    ),
    _option_counts(option_counts) {}

ValueVec DiscreteHistogram::build_function_impl(
    FunctionBuilder& fb, const ValueVec& args
) const {
    ValueVec results;
    auto weights = args.at(_option_counts.size());
    for (auto [option_count, input] : zip(_option_counts, args)) {
        auto [values, counts] =
            fb.discrete_histogram(input, weights, static_cast<me_int_t>(option_count));
        results.push_back(values);
        results.push_back(counts);
    }
    return results;
}

DiscreteSampler::DiscreteSampler(
    const std::vector<std::size_t>& option_counts,
    const std::string& prefix,
    const std::vector<std::size_t>& dims_with_prior
) :
    Mapping(
        "DiscreteSampler",
        TypeVec(option_counts.size(), batch_float),
        TypeVec(option_counts.size(), batch_int),
        [&] {
            TypeVec cond_types;
            for (std::size_t dim : dims_with_prior) {
                cond_types.push_back(batch_float_array(option_counts.at(dim)));
            }
            return cond_types;
        }()
    ),
    _option_counts(option_counts),
    _dim_has_prior(option_counts.size()) {
    for (std::size_t i = 0; i < option_counts.size(); ++i) {
        _prob_names.push_back(prefixed_name(prefix, std::format("prob{}", i)));
    }
    for (std::size_t dim : dims_with_prior) {
        _dim_has_prior.at(dim) = true;
    }
}

Mapping::Result DiscreteSampler::build_forward_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    return build_transform(fb, inputs, conditions, false);
}

Mapping::Result DiscreteSampler::build_inverse_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    return build_transform(fb, inputs, conditions, true);
}

Mapping::Result DiscreteSampler::build_transform(
    FunctionBuilder& fb,
    const ValueVec& inputs,
    const ValueVec& conditions,
    bool inverse
) const {
    ValueVec dets, outputs;
    std::size_t condition_index = 0;
    for (auto [input, prob_name, option_count, has_prior] :
         zip(inputs, _prob_names, _option_counts, _dim_has_prior)) {
        auto probs =
            fb.global(prob_name, DataType::dt_float, {static_cast<int>(option_count)});
        if (has_prior) {
            probs = fb.mul(probs, conditions.at(condition_index));
            ++condition_index;
        }
        auto [output, det] = inverse ? fb.sample_discrete_probs_inverse(input, probs)
                                     : fb.sample_discrete_probs(input, probs);
        outputs.push_back(output);
        dets.push_back(det);
    }
    return {outputs, fb.product(dets)};
}

void DiscreteSampler::initialize_globals(ContextPtr context) const {
    for (auto [prob_name, option_count] : zip(_prob_names, _option_counts)) {
        initialize_uniform_probs(context, prob_name, option_count);
    }
}

void madspace::initialize_uniform_probs(
    ContextPtr context, const std::string& name, std::size_t option_count
) {
    bool is_cpu = context->device() == cpu_device();
    auto prob_global = context->define_global(name, DataType::dt_float, {option_count});
    auto prob = is_cpu ? prob_global : Tensor(DataType::dt_float, {option_count});
    auto prob_view = prob.view<double, 2>()[0];
    auto prob_value = 1. / option_count;
    for (std::size_t i = 0; i < prob_view.size(); ++i) {
        prob_view[i] = prob_value;
    }
    if (!is_cpu) {
        prob_global.copy_from(prob);
    }
}

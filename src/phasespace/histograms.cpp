#include "madevent/phasespace/histograms.h"

using namespace madevent;

ObservableHistograms::ObservableHistograms(const std::vector<HistItem>& observables) :
    FunctionGenerator(
        "ObservablesHistogram",
        {batch_float, observables.at(0).observable.arg_types().at(1)},
        [&]() {
            TypeVec ret_types;
            for (auto& obs : observables) {
                ret_types.push_back(single_float_array(obs.bin_count + 2));
                ret_types.push_back(single_float_array(obs.bin_count + 2));
            }
            return ret_types;
        }()
    ),
    _observables(observables) {}

ValueVec ObservableHistograms::build_function_impl(
    FunctionBuilder& fb, const ValueVec& args
) const {
    Value weight = args.at(0), momenta = args.at(1);
    Value sqrt_s = fb.obs_mass(fb.reduce_sum_vector(momenta));
    ValueVec histograms;
    for (auto& obs : _observables) {
        Value obs_result = obs.observable.build_function(fb, args).at(0);
        auto [values, square_values] = fb.histogram(
            obs_result, weight, obs.min, obs.max, static_cast<me_int_t>(obs.bin_count)
        );
        histograms.push_back(values);
        histograms.push_back(square_values);
    }
    return histograms;
}

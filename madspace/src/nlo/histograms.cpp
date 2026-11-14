#include "madspace/nlo/histograms.hpp"

#include <format>
#include <stdexcept>

using namespace madspace;

namespace {

// the outputs of the histograms, as those of ObservableHistograms
NamedVector<Type>
histogram_returns(const std::vector<NLOHistograms::HistItem>& histograms) {
    NamedVector<Type> returns;
    for (std::size_t i = 0; i < histograms.size(); ++i) {
        auto bins = static_cast<int>(histograms.at(i).bin_count) + 2;
        returns.push_back(std::format("values{}", i), single_float_array(bins));
        returns.push_back(std::format("square_values{}", i), single_float_array(bins));
    }
    return returns;
}

} // namespace

NLOHistograms::NLOHistograms(
    const std::vector<HistItem>& histograms, std::size_t kinematics_count
) :
    FunctionGenerator(
        "NLOHistograms",
        [&] {
            NamedVector<Type> args;
            for (std::size_t k = 0; k < kinematics_count; ++k) {
                args.push_back(std::format("weight{}", k), batch_float);
                args.push_back(std::format("jet_count{}", k), batch_int);
            }
            for (std::size_t i = 0; i < histograms.size(); ++i) {
                for (std::size_t k = 0; k < kinematics_count; ++k) {
                    args.push_back(std::format("value{}_{}", i, k), batch_float);
                }
            }
            return args;
        }(),
        histogram_returns(histograms)
    ),
    _histograms(histograms),
    _kinematics_count(kinematics_count) {
    if (histograms.empty()) {
        throw std::invalid_argument("NLOHistograms: no histograms");
    }
    if (kinematics_count == 0 || kinematics_count > 64) {
        throw std::invalid_argument("NLOHistograms: from 1 to 64 kinematics per point");
    }
    for (const auto& histogram : histograms) {
        if (histogram.bin_count == 0 || !(histogram.max > histogram.min)) {
            throw std::invalid_argument(
                "NLOHistograms: a histogram needs bins and max > min"
            );
        }
    }
}

NamedVector<Value> NLOHistograms::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    ValueVec weights, jets;
    for (std::size_t k = 0; k < _kinematics_count; ++k) {
        weights.push_back(args.at(std::format("weight{}", k)));
        jets.push_back(args.at(std::format("jet_count{}", k)));
    }
    Value weight = fb.stack(weights), jet_count = fb.stack(jets);
    NamedVector<Value> result;
    for (std::size_t i = 0; i < _histograms.size(); ++i) {
        const auto& histogram = _histograms.at(i);
        ValueVec values;
        for (std::size_t k = 0; k < _kinematics_count; ++k) {
            values.push_back(args.at(std::format("value{}_{}", i, k)));
        }
        auto [center, merged] = fb.nlo_histogram_weights(
            fb.stack(values),
            weight,
            jet_count,
            std::vector<double>{
                histogram.min,
                histogram.max,
                static_cast<double>(histogram.bin_count),
                static_cast<double>(histogram.jet_count),
            }
        );
        // one histogram of the kinematics of all points: a point adds the
        // squares of its sums per bin, as the merged weights have them
        auto [centers, center_counts] = fb.batch_cat(fb.unstack(center));
        auto [sums, sum_counts] = fb.batch_cat(fb.unstack(merged));
        auto [bins, square_bins] = fb.histogram(
            centers,
            sums,
            histogram.min,
            histogram.max,
            static_cast<me_int_t>(histogram.bin_count)
        );
        result.push_back(std::format("values{}", i), bins);
        result.push_back(std::format("square_values{}", i), square_bins);
    }
    return result;
}

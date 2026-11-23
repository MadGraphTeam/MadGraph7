#include "madspace/driver/discrete_optimizer.hpp"

#include <cmath>

using namespace madspace;

void DiscreteOptimizer::add_data(const std::vector<Tensor>& values_and_counts) {
    if (_data.size() != _prob_names.size()) {
        _data.resize(_prob_names.size());
        for (auto [prob_name, data_item] : zip(_prob_names, _data)) {
            auto prob_global = _contexts.at(0)->global(prob_name);
            auto option_count = prob_global.size(1);
            auto& [weight_sums, counts] = data_item;
            weight_sums.resize(option_count);
            counts.resize(option_count);
        }
    }

    for (std::size_t i = 0; auto [prob_name, data_item] : zip(_prob_names, _data)) {
        auto& [weight_sums, counts] = data_item;
        auto values_cpu = values_and_counts.at(2 * i).cpu();
        auto counts_cpu = values_and_counts.at(2 * i + 1).cpu();
        auto values_view = values_cpu.view<double, 2>()[0];
        auto counts_view = counts_cpu.view<me_int_t, 2>()[0];
        for (std::size_t j = 0; j < values_view.size(); ++j) {
            weight_sums[j] += values_view[j];
            counts[j] += counts_view[j];
        }
        ++i;
    }
}

void DiscreteOptimizer::optimize() {
    if (_data.size() != _prob_names.size()) {
        return;
    }
    // the events of this update, those of the first dimension
    std::size_t new_count = 0;
    for (std::size_t count : std::get<1>(_data.at(0))) {
        new_count += count;
    }
    if (new_count == 0) {
        return;
    }
    double old_share = static_cast<double>(_sample_count) / (_sample_count + new_count);
    _sample_count += new_count;
    for (auto [prob_name, data_item] : zip(_prob_names, _data)) {
        auto& [weight_sums, counts] = data_item;
        Tensor prob = _contexts.at(0)->global(prob_name).cpu().copy();
        auto prob_view = prob.view<double, 2>()[0];

        // |I_i| = p_i |<w>_i| per category
        std::vector<double> contributions(weight_sums.size(), 0.);
        double norm = 0.;
        for (std::size_t i = 0; i < contributions.size(); ++i) {
            if (counts.at(i) > 0) {
                contributions.at(i) =
                    prob_view[i] * std::abs(weight_sums.at(i)) / counts.at(i);
            }
            norm += contributions.at(i);
        }
        if (norm > 0. && std::isfinite(norm)) {
            for (std::size_t i = 0; i < contributions.size(); ++i) {
                prob_view[i] = prob_view[i] * old_share +
                    contributions.at(i) / norm * (1. - old_share);
            }
            for (auto& context : _contexts) {
                context->global(prob_name).copy_from(prob);
            }
        }
        std::fill(weight_sums.begin(), weight_sums.end(), 0.);
        std::fill(counts.begin(), counts.end(), 0);
    }
}

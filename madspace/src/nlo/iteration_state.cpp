#include "madspace/nlo/iteration_state.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>

#include "madspace/util.hpp"

using namespace madspace;

namespace {

// the outputs of an integrand summed for the results, in this order
const std::vector<std::string> value_names{
    "weight",
    "lo_weight",
    "virtual_residual",
    "scale_variations",
    "pdf_variations",
    "ew_sudakov_weights",
};

// whether all or none of the names are outputs
template <typename IndexMap>
bool all_or_none(const IndexMap& index_map, const std::vector<std::string>& names) {
    std::size_t present = 0;
    for (auto& name : names) {
        present += index_map.contains(name);
    }
    if (present != 0 && present != names.size()) {
        throw std::invalid_argument(
            "NLOIterationState: the outputs of a part of the state are not all there"
        );
    }
    return present != 0;
}

} // namespace

NLOIterationState::NLOIterationState(
    const NamedVector<Type>& event_types,
    const std::string& probability_name,
    const std::string& fraction_name,
    const std::string& ratio_name,
    std::size_t configuration_count,
    double virtual_fraction,
    double min_virtual_fraction,
    double floor,
    bool adapt_probabilities
) :
    _event_types(event_types),
    _probability_name(probability_name),
    _fraction_name(fraction_name),
    _ratio_name(ratio_name),
    _fraction(virtual_fraction),
    _min_fraction(min_virtual_fraction),
    _floor(floor),
    _adapt_probabilities(adapt_probabilities) {
    auto index_map = event_types.index_map();
    if (!index_map.contains("weight")) {
        throw std::invalid_argument("NLOIterationState: the outputs have no weight");
    }
    std::size_t width = 0;
    for (auto& name : value_names) {
        if (!index_map.contains(name)) {
            continue;
        }
        if (name == "virtual_residual") {
            _residual = _value_fields.size();
        }
        auto& shape = event_types.at(name).shape;
        std::size_t size = shape.empty() ? 1 : static_cast<std::size_t>(shape.at(0));
        _value_fields.push_back({index_map.at(name), shape.empty() ? 0 : size});
        width += size;
    }
    _has_virtual =
        all_or_none(index_map, {"virtual", "born", "evaluated", "virtual_residual"});
    _has_choice =
        all_or_none(index_map, {"fks_index", "configuration_weight", "nbody_weight"});
    if (_has_virtual) {
        _virtual = index_map.at("virtual");
        _born = index_map.at("born");
        _evaluated = index_map.at("evaluated");
        _ratios.assign(event_types.at("virtual").shape.at(0), 0.);
    }
    if (_has_choice) {
        _configuration_weight = index_map.at("configuration_weight");
        _nbody_weight = index_map.at("nbody_weight");
        _fks_index = index_map.at("fks_index");
        _probabilities.assign(configuration_count, 1. / configuration_count);
    }
    _sums.assign(width, 0.);
    _covariance.assign(width * width, 0.);
    _iteration_sums.assign(width, 0.);
    _iteration_products.assign(width * width, 0.);
    _virtual_sums.assign(_ratios.size(), 0.);
    _choice_sums.assign(_probabilities.size(), 0.);
    _choice_counts.assign(_probabilities.size(), 0.);
}

void NLOIterationState::add(const TensorVec& events) {
    std::vector<Tensor> values;
    for (auto [index, width] : _value_fields) {
        values.push_back(events.at(index).cpu());
    }
    // the weight is the first of the sums
    Tensor& weight_tensor = values.at(0);
    std::size_t count = weight_tensor.size(0);
    std::size_t width = _sums.size();
    std::vector<double> point(width);
    for (std::size_t i = 0; i < count; ++i) {
        std::size_t k = 0;
        for (auto [field, tensor] : zip(_value_fields, values)) {
            if (field.second == 0) {
                point[k++] = tensor.view<double, 1>()[i];
            } else {
                auto row = tensor.view<double, 2>()[i];
                for (std::size_t j = 0; j < field.second; ++j) {
                    point[k++] = row[j];
                }
            }
        }
        for (std::size_t a = 0; a < width; ++a) {
            _iteration_sums[a] += point[a];
            for (std::size_t b = 0; b < width; ++b) {
                _iteration_products[a * width + b] += point[a] * point[b];
            }
        }
    }
    if (_has_virtual) {
        Tensor virtual_tensor = events.at(_virtual).cpu();
        Tensor born_tensor = events.at(_born).cpu();
        Tensor evaluated_tensor = events.at(_evaluated).cpu();
        auto virtuals = virtual_tensor.view<double, 2>();
        auto borns = born_tensor.view<double, 2>();
        auto evaluated = evaluated_tensor.view<double, 1>();
        auto residuals = values.at(_residual).view<double, 1>();
        for (std::size_t i = 0; i < count; ++i) {
            double residual = residuals[i];
            _residual_sum += residual;
            _residual_square_sum += residual * residual;
            if (evaluated[i] != 0.) {
                for (std::size_t o = 0; o < _virtual_sums.size(); ++o) {
                    _virtual_sums[o] += virtuals[i][o];
                }
                for (std::size_t b = 0; b < borns.size(1); ++b) {
                    _born_sum += borns[i][b];
                }
            }
        }
    }
    if (_has_choice) {
        Tensor choice_tensor = events.at(_configuration_weight).cpu();
        Tensor index_tensor = events.at(_fks_index).cpu();
        Tensor nbody_tensor = events.at(_nbody_weight).cpu();
        auto weights = weight_tensor.view<double, 1>();
        auto choice_weights = choice_tensor.view<double, 1>();
        auto indices = index_tensor.view<me_int_t, 1>();
        auto nbody = nbody_tensor.view<double, 1>();
        for (std::size_t i = 0; i < count; ++i) {
            // the n+1-body part of the integrand of the configuration, w p_k
            // without the n-body part
            double integrand = (weights[i] - nbody[i]) / choice_weights[i];
            auto index = static_cast<std::size_t>(indices[i]);
            _choice_sums.at(index) += std::abs(integrand);
            _choice_counts.at(index) += 1.;
        }
    }
    _iteration_count += count;
}

void NLOIterationState::end_iteration(const std::vector<ContextPtr>& contexts) {
    if (!_started && !contexts.empty()) {
        read_globals(contexts.at(0));
    }
    _started = true;
    std::size_t n = _iteration_count;
    std::size_t width = _sums.size();
    if (n > 0) {
        _count += n;
        for (std::size_t a = 0; a < width; ++a) {
            _sums[a] += _iteration_sums[a];
            for (std::size_t b = 0; b < width; ++b) {
                _covariance[a * width + b] += _iteration_products[a * width + b] -
                    _iteration_sums[a] * _iteration_sums[b] / n;
            }
        }
        // the standard deviations of the cross section and of the residual
        double mean = _iteration_sums[0] / n;
        double variance = std::max(_iteration_products[0] / n - mean * mean, 0.);
        _iteration_std = std::sqrt(variance);
        if (_has_virtual) {
            double residual_mean = _residual_sum / n;
            double residual_variance =
                std::max(_residual_square_sum / n - residual_mean * residual_mean, 0.);
            double error_total = std::sqrt(variance / n);
            double error_virtual = std::sqrt(residual_variance / n);
            if (_born_sum != 0.) {
                for (std::size_t o = 0; o < _ratios.size(); ++o) {
                    _ratios[o] = _virtual_sums[o] / _born_sum;
                }
            }
            if (error_total > 0.) {
                double factor = std::clamp(2. * error_virtual / error_total, 0.25, 2.);
                _fraction = std::max(std::min(_fraction * factor, 1.), _min_fraction);
            }
        }
        if (_has_choice) {
            double target_sum = 0.;
            std::vector<double> target(_probabilities.size(), 0.);
            for (std::size_t k = 0; k < target.size(); ++k) {
                if (_choice_counts[k] > 0.) {
                    target[k] = _choice_sums[k] / _choice_counts[k];
                    target_sum += target[k];
                }
            }
            if (_adapt_probabilities && target_sum > 0.) {
                double uniform = 1. / target.size();
                for (std::size_t k = 0; k < target.size(); ++k) {
                    _probabilities[k] =
                        (1. - _floor) * target[k] / target_sum + _floor * uniform;
                }
            }
        }
    }
    _iteration_count = 0;
    std::fill(_iteration_sums.begin(), _iteration_sums.end(), 0.);
    std::fill(_iteration_products.begin(), _iteration_products.end(), 0.);
    _residual_sum = 0.;
    _residual_square_sum = 0.;
    std::fill(_virtual_sums.begin(), _virtual_sums.end(), 0.);
    _born_sum = 0.;
    std::fill(_choice_sums.begin(), _choice_sums.end(), 0.);
    std::fill(_choice_counts.begin(), _choice_counts.end(), 0.);
    for (auto& context : contexts) {
        write_globals(context);
    }
}

void NLOIterationState::reset() {
    _count = 0;
    std::fill(_sums.begin(), _sums.end(), 0.);
    std::fill(_covariance.begin(), _covariance.end(), 0.);
}

void NLOIterationState::write_globals(ContextPtr context) const {
    auto write = [&](const std::string& name, const std::vector<double>& values) {
        auto global = context->global(name);
        Tensor tensor(DataType::dt_float64, global.shape());
        auto view = tensor.view<double, 2>()[0];
        for (std::size_t i = 0; i < values.size(); ++i) {
            view[i] = values[i];
        }
        global.copy_from(tensor);
    };
    if (_has_choice) {
        write(_probability_name, _probabilities);
    }
    if (_has_virtual) {
        write(_ratio_name, _ratios);
        auto global = context->global(_fraction_name);
        Tensor tensor(DataType::dt_float64, global.shape());
        tensor.view<double, 1>()[0] = _fraction;
        global.copy_from(tensor);
    }
}

void NLOIterationState::read_globals(ContextPtr context) {
    auto read = [&](const std::string& name, std::vector<double>& values) {
        Tensor tensor = context->global(name).cpu();
        auto view = tensor.view<double, 2>()[0];
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = view[i];
        }
    };
    if (_has_choice) {
        read(_probability_name, _probabilities);
    }
    if (_has_virtual) {
        read(_ratio_name, _ratios);
        Tensor tensor = context->global(_fraction_name).cpu();
        _fraction = tensor.view<double, 1>()[0];
    }
}

std::pair<double, double>
NLOIterationState::combination(const std::vector<double>& coefficients) const {
    if (_count == 0) {
        return {0., 0.};
    }
    std::size_t width = _sums.size();
    if (coefficients.size() > width) {
        throw std::invalid_argument(
            std::format(
                "NLOIterationState: {} coefficients for {} sums",
                coefficients.size(),
                width
            )
        );
    }
    double mean = 0., variance = 0.;
    for (std::size_t a = 0; a < coefficients.size(); ++a) {
        mean += coefficients[a] * _sums[a];
        for (std::size_t b = 0; b < coefficients.size(); ++b) {
            variance += coefficients[a] * _covariance[a * width + b] * coefficients[b];
        }
    }
    double count = static_cast<double>(_count);
    return {mean / count, std::sqrt(std::max(variance, 0.)) / count};
}

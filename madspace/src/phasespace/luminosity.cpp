#include "madspace/phasespace/luminosity.hpp"

#include <cmath>
#include <stdexcept>

using namespace madspace;

Luminosity::Luminosity(
    double s_lab,
    double s_hat_min,
    double s_hat_max,
    double invariant_power,
    double mass,
    double width
) :
    Mapping(
        "Luminosity",
        {{"r_s", batch_float}, {"r_x", batch_float}},
        {{"x1", batch_float}, {"x2", batch_float}, {"s_hat", batch_float}},
        {}
    ),
    _s_lab(s_lab),
    _s_hat_min(s_hat_min),
    _s_hat_max(s_hat_max == 0 ? s_lab : s_hat_max),
    _invariant(invariant_power, mass, width) {}

Mapping::Result Luminosity::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto r_s = inputs[0], r_x = inputs[1];
    auto result = _invariant.build_forward(fb, {r_s}, {_s_hat_min, _s_hat_max});
    auto s_hat = result["s"];
    auto [x1, x2, det_x] = fb.r_to_x1x2(r_x, s_hat, _s_lab);
    auto det = fb.mul(result["det"], det_x);
    return {{{"x1", x1}, {"x2", x2}, {"s_hat", s_hat}}, det};
}

Mapping::Result Luminosity::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto x1 = inputs[0], x2 = inputs[1], s_hat = inputs[2];
    auto result = _invariant.build_inverse(fb, {s_hat}, {_s_hat_min, _s_hat_max});
    auto [r_x, det_x] = fb.x1x2_to_r(x1, x2, _s_lab);
    auto det = fb.mul(result["det"], det_x);
    return {{{"r_s", result["r"]}, {"r_x", r_x}}, det};
}

FKSLeptonLuminosity::FKSLeptonLuminosity(
    double s_lab, double beta, double s_hat_min, bool s_hat_min_condition
) :
    Mapping(
        "FKSLeptonLuminosity",
        {{"r1", batch_float}, {"r2", batch_float}},
        {{"x1", batch_float},
         {"x2", batch_float},
         {"s_hat", batch_float},
         {"one_minus_x1", batch_float},
         {"one_minus_x2", batch_float}},
        [&] {
            NamedVector<Type> conditions;
            if (s_hat_min_condition) {
                conditions.push_back("s_hat_min", batch_float);
            }
            return conditions;
        }()
    ),
    _s_lab(s_lab),
    _beta(beta),
    _s_hat_min(s_hat_min),
    _s_hat_min_condition(s_hat_min_condition) {
    if (!(s_lab > 0.) || !std::isfinite(s_lab)) {
        throw std::invalid_argument("FKSLeptonLuminosity: s_lab must be positive");
    }
    if (!(beta < 1.) || !std::isfinite(beta)) {
        throw std::invalid_argument("FKSLeptonLuminosity: beta must be below 1");
    }
    if (!(s_hat_min >= 0.) || !(s_hat_min < s_lab)) {
        throw std::invalid_argument(
            "FKSLeptonLuminosity: s_hat_min must be in [0, s_lab)"
        );
    }
}

Value FKSLeptonLuminosity::tau_min(
    FunctionBuilder& fb, const NamedVector<Value>& conditions
) const {
    if (_s_hat_min_condition) {
        return fb.mul(conditions.at("s_hat_min"), 1. / _s_lab);
    }
    return _s_hat_min / _s_lab;
}

Mapping::Result FKSLeptonLuminosity::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto tau = tau_min(fb, conditions);
    auto [x1, omx1, det1] = fb.lepton_x_sample(inputs.at(0), tau, _beta);
    auto [x2, omx2, det2] = fb.lepton_x_sample(inputs.at(1), fb.div(tau, x1), _beta);
    auto s_hat = fb.mul(fb.mul(x1, x2), _s_lab);
    return {
        {{"x1", x1},
         {"x2", x2},
         {"s_hat", s_hat},
         {"one_minus_x1", omx1},
         {"one_minus_x2", omx2}},
        fb.mul(det1, det2)
    };
}

Mapping::Result FKSLeptonLuminosity::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto tau = tau_min(fb, conditions);
    auto x1 = inputs.at(0);
    auto [r1, det1] = fb.lepton_x_sample_inverse(inputs.at(3), tau, _beta);
    auto [r2, det2] = fb.lepton_x_sample_inverse(inputs.at(4), fb.div(tau, x1), _beta);
    return {{{"r1", r1}, {"r2", r2}}, fb.mul(det1, det2)};
}

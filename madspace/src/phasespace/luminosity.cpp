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
    // Invariant names its output "invariant", not "s"
    auto s_hat = result["invariant"];
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
    return {{{"r_s", result["random"]}, {"r_x", r_x}}, det};
}

LeptonLuminosity::LeptonLuminosity(
    double s_lab, double s_hat_min, double sampling_power
) :
    Mapping(
        "LeptonLuminosity",
        {{"r1", batch_float}, {"r2", batch_float}},
        {{"x1", batch_float},
         {"omx1", batch_float},
         {"x2", batch_float},
         {"omx2", batch_float},
         {"s_hat", batch_float}},
        {}
    ),
    _s_lab(s_lab),
    _s_hat_min(s_hat_min),
    _power(sampling_power) {
    if (s_lab <= 0.) {
        throw std::invalid_argument("LeptonLuminosity: s_lab must be positive");
    }
    if (s_hat_min < 0. || s_hat_min > s_lab) {
        throw std::invalid_argument(
            "LeptonLuminosity: s_hat_min must lie between 0 and s_lab"
        );
    }
    if (sampling_power >= 1. || sampling_power < 0.) {
        // p >= 1 makes (1-x)^-p non-integrable, so the sampling would have no
        // normalization; a negative p would push events away from the
        // endpoint, which is where the density lives.
        throw std::invalid_argument(
            "LeptonLuminosity: sampling_power must lie in [0, 1)"
        );
    }
}

Mapping::Result LeptonLuminosity::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto r1 = inputs[0], r2 = inputs[1];
    auto batch = fb.batch_size({r1});
    double tau_min = _s_hat_min / _s_lab;
    // 1-x = r^(1/(1-p)) * (1-xmin); the Jacobian of that is
    // (1-xmin)^(1-p)/(1-p) * (1-x)^p, and the (1-x)^p is left out on purpose
    // -- see the class documentation.
    auto exponent = fb.full({1. / (1. - _power), batch});
    auto one = fb.full({1., batch});
    auto norm = fb.full({1. / (1. - _power), batch});

    // first beam: xmin = tau_min, so the span is a constant
    double span1 = 1. - tau_min;
    auto omx1 = fb.mul(fb.pow(r1, exponent), fb.full({span1, batch}));
    auto x1 = fb.sub(one, omx1);
    auto det1 = fb.full({std::pow(span1, 1. - _power) / (1. - _power), batch});

    // second beam: xmin = tau_min/x1, which the first draw fixes
    auto span2 = fb.sub(one, fb.div(fb.full({tau_min, batch}), x1));
    auto omx2 = fb.mul(fb.pow(r2, exponent), span2);
    auto x2 = fb.sub(one, omx2);
    auto det2 = fb.mul(fb.pow(span2, fb.full({1. - _power, batch})), norm);

    auto s_hat = fb.mul(fb.mul(x1, x2), fb.full({_s_lab, batch}));
    return {
        {{"x1", x1}, {"omx1", omx1}, {"x2", x2}, {"omx2", omx2}, {"s_hat", s_hat}},
        fb.mul(det1, det2)
    };
}

Mapping::Result LeptonLuminosity::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto x1 = inputs[0], omx1 = inputs[1], omx2 = inputs[3];
    auto batch = fb.batch_size({x1});
    double tau_min = _s_hat_min / _s_lab;
    auto one = fb.full({1., batch});
    auto inv_exponent = fb.full({1. - _power, batch});
    // An inverse map returns its own Jacobian, d(r)/d(x), not the forward
    // one: kernel_uniform_invariant gives gs = s_max - s_min while
    // kernel_uniform_invariant_inverse gives gs = 1/(s_max - s_min). So these
    // are the reciprocals of the two factors build_forward_impl returns.
    auto inv_norm = fb.full({1. - _power, batch});
    // as a negative power rather than a reciprocal, so that a degenerate
    // span (x1 at its lower limit) gives zero instead of an infinity -- the
    // same reason lepton_split writes omx_b^(-p)
    auto neg_exponent = fb.full({_power - 1., batch});

    double span1 = 1. - tau_min;
    auto r1 = fb.pow(fb.div(omx1, fb.full({span1, batch})), inv_exponent);
    auto det1 = fb.full({(1. - _power) / std::pow(span1, 1. - _power), batch});

    auto span2 = fb.sub(one, fb.div(fb.full({tau_min, batch}), x1));
    auto r2 = fb.pow(fb.div(omx2, span2), inv_exponent);
    auto det2 = fb.mul(fb.pow(span2, neg_exponent), inv_norm);

    return {{{"r1", r1}, {"r2", r2}}, fb.mul(det1, det2)};
}

LeptonLuminosityInvariant::LeptonLuminosityInvariant(
    double s_lab,
    double s_hat_min,
    double s_hat_max,
    double sampling_power,
    double mass,
    double width,
    double invariant_power
) :
    Mapping(
        "LeptonLuminosityInvariant",
        {{"r_s", batch_float}, {"r_x", batch_float}, {"r_swap", batch_float}},
        {{"x1", batch_float},
         {"omx1", batch_float},
         {"x2", batch_float},
         {"omx2", batch_float},
         {"s_hat", batch_float}},
        {}
    ),
    _s_lab(s_lab),
    _s_hat_min(s_hat_min),
    _s_hat_max(s_hat_max == 0. ? s_lab : s_hat_max),
    _power(sampling_power),
    _invariant(invariant_power, mass, width) {
    if (s_lab <= 0.) {
        throw std::invalid_argument(
            "LeptonLuminosityInvariant: s_lab must be positive"
        );
    }
    if (sampling_power >= 1. || sampling_power < 0.) {
        throw std::invalid_argument(
            "LeptonLuminosityInvariant: sampling_power must lie in [0, 1)"
        );
    }
}

LeptonSHatResult madspace::lepton_s_hat(
    FunctionBuilder& fb, Value s_min, Value s_max, Value r, double power
) {
    auto batch = fb.batch_size({s_max});
    auto span = fb.sub(s_max, s_min);
    auto v = fb.pow(r, fb.full({1. / (1. - power), batch}));
    // s_gap is the product, not s_max - s_hat recomputed: with p = 0.96 it is
    // r^25, which for 22% of uniform r is smaller than the spacing of doubles
    // at s_max, so the subtraction below gives back exactly zero. Everything
    // downstream needs the gap, not s_hat, and this is the only place it
    // still exists exactly.
    auto s_gap = fb.mul(span, v);
    auto s_hat = fb.sub(s_max, s_gap);
    auto det = fb.mul(
        fb.mul(span, fb.pow(v, fb.full({power, batch}))),
        fb.full({1. / (1. - power), batch})
    );
    return {s_hat, s_gap, det};
}

LeptonSplitResult madspace::lepton_split(
    FunctionBuilder& fb,
    Value s_hat,
    Value omtau,
    Value r_x,
    Value r_swap,
    double s_lab,
    double power
) {
    auto batch = fb.batch_size({s_hat});
    auto one = fb.full({1., batch});
    auto tau = fb.div(s_hat, fb.full({s_lab, batch}));

    // The drawn fraction is the larger one, over [sqrt(tau), 1]. The span is
    // built as (1-tau)/(1+sqrt(tau)) rather than 1-sqrt(tau): the sampling
    // puts 22% of its points where tau has rounded to exactly 1, and the
    // subtraction gives zero there, which sends det_a to zero and throws the
    // point away. This form keeps it, because omtau never went through a
    // cancellation.
    auto x_min = fb.sqrt(tau);
    auto span = fb.div(omtau, fb.add(one, x_min));
    auto omx_a = fb.mul(fb.pow(r_x, fb.full({1. / (1. - power), batch})), span);
    auto x_a = fb.sub(one, omx_a);
    auto det_a = fb.mul(
        fb.pow(span, fb.full({1. - power, batch})),
        fb.full({1. / (1. - power), batch})
    );

    // the other follows from tau; 1/x_a is the Jacobian of x2 = tau/x1, and
    // omx_b^(-p) is the factor the density expects to have been cancelled by
    // the sampling, which for this beam never happened
    // x_b keeps the division: it is what makes x_a * x_b equal tau exactly,
    // which the momenta are built from, and tau/x_a is accurate at both ends.
    auto x_b = fb.div(tau, x_a);
    // 1 - x_b is not, so omx_b is built as (x_a - tau)/x_a = (omtau-omx_a)/x_a
    // instead. At tau -> 1 the operands of `1 - x_b` are both ~1 and their
    // difference is pure rounding, whereas omtau and omx_a are both small and
    // exact. omx_a <= span <= omtau keeps this non-negative, and at small tau
    // -- where the two are instead both ~1 -- the cancellation is bounded by
    // omx_a <= omtau/(1+sqrt(tau)), leaving a relative error of order
    // eps/sqrt(tau) on a quantity that is O(1) there anyway.
    auto omx_b = fb.div(fb.sub(omtau, omx_a), x_a);
    // Written as omx_b^(-p) rather than 1/omx_b^p on purpose: pow returns 0
    // for a non-positive base, so a degenerate point gets weight zero, where
    // the reciprocal would divide by zero and poison the whole batch with an
    // infinity.
    auto det_b = fb.div(fb.pow(omx_b, fb.full({-power, batch})), x_a);

    auto swap = fb.sample_discrete(r_swap, fb.full({static_cast<me_int_t>(2), batch}));
    auto index = swap[0];
    // Jacobian of (s_hat, r) -> (x1, x2), the same convention r_to_x1x2 uses:
    // the 1/s_lab is the d(s_hat) -> d(tau) conversion, and without it the
    // mapping is off by exactly s_lab. The factor 2 is the 50/50 beam choice.
    auto det = fb.product(
        {det_a, det_b, fb.full({2. / s_lab, batch})}
    );
    return {
        fb.gather(index, fb.stack({x_a, x_b})),
        fb.gather(index, fb.stack({x_b, x_a})),
        fb.gather(index, fb.stack({omx_a, omx_b})),
        fb.gather(index, fb.stack({omx_b, omx_a})),
        det
    };
}

Mapping::Result LeptonLuminosityInvariant::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto r_s = inputs[0], r_x = inputs[1], r_swap = inputs[2];
    auto batch = fb.batch_size({r_s});
    auto inv = _invariant.build_forward(fb, {r_s}, {_s_hat_min, _s_hat_max});
    auto s_hat = inv["invariant"];
    // This channel samples s_hat through a resonance, not from the endpoint,
    // so there is no exact gap to carry and 1-tau is the best available. That
    // is fine here: the Breit-Wigner puts its weight at the resonance, not at
    // tau -> 1 where the cancellation bites.
    auto omtau =
        fb.sub(fb.full({1., batch}), fb.div(s_hat, fb.full({_s_lab, batch})));
    auto split = lepton_split(fb, s_hat, omtau, r_x, r_swap, _s_lab, _power);
    // lepton_split already converts d(s_hat) to d(tau), matching r_to_x1x2
    return {
        {{"x1", split.x1},
         {"omx1", split.omx1},
         {"x2", split.x2},
         {"omx2", split.omx2},
         {"s_hat", s_hat}},
        fb.mul(inv["det"], split.det)
    };
}

Mapping::Result LeptonLuminosityInvariant::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    // Not implemented: the forward map is a two-channel construction whose
    // inverse needs the swap recovered from the fractions, and nothing in the
    // integrator asks for it -- a multichannel mapping is used forward only.
    // Failing loudly beats returning something plausible and wrong.
    throw std::runtime_error(
        "LeptonLuminosityInvariant: no inverse map; this is a forward-only "
        "sampling channel"
    );
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

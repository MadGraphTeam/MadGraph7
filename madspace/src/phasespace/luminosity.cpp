#include "madspace/phasespace/luminosity.hpp"

#include <cmath>
#include <limits>
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
    double s_lab,
    double s_hat_min,
    double sampling_power,
    bool reduced_density,
    bool s_hat_min_condition
) :
    Mapping(
        "LeptonLuminosity",
        {{"r1", batch_float}, {"r2", batch_float}},
        {{"x1", batch_float},
         {"omx1", batch_float},
         {"x2", batch_float},
         {"omx2", batch_float},
         {"s_hat", batch_float}},
        [&] {
            NamedVector<Type> conditions;
            if (s_hat_min_condition) {
                conditions.push_back("s_hat_min", batch_float);
            }
            return conditions;
        }()
    ),
    _s_lab(s_lab),
    _s_hat_min(s_hat_min),
    _power(sampling_power),
    _reduced_density(reduced_density),
    _s_hat_min_condition(s_hat_min_condition) {
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

std::pair<Value, Value> LeptonLuminosity::tau_min(
    FunctionBuilder& fb, const NamedVector<Value>& conditions
) const {
    // tau_min and 1 - tau_min, the latter as (s_lab - s_hat_min) / s_lab, so
    // that it keeps its digits for a bound close to s_lab. The bound of the
    // condition is clipped to [0, s_lab]: below 0 there is none, and above
    // s_lab no range of x is left, which gives x1 = x2 = 1 and a zero
    // Jacobian. The same operations on the bound of the constructor give the
    // same mapping to the last digit.
    if (_s_hat_min_condition) {
        auto s_hat_min = fb.min(fb.max(conditions.at("s_hat_min"), 0.), _s_lab);
        return {fb.div(s_hat_min, _s_lab), fb.div(fb.sub(_s_lab, s_hat_min), _s_lab)};
    }
    return {_s_hat_min / _s_lab, (_s_lab - _s_hat_min) / _s_lab};
}

Mapping::Result LeptonLuminosity::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    // the power of 1 - x in the Jacobian of each beam: none with a reduced
    // density, which carries (1 - x)^p itself -- see the class documentation
    double power = _reduced_density ? 0. : _power;
    auto [tau, omtau] = tau_min(fb, conditions);
    // The first beam has x_min = tau_min, the second x_min = tau_min / x1, and
    // the width of each is 1 - x_min. That of the second is 1 minus the
    // rounded tau_min / x1, the exact length of the range that x2 is then
    // sampled over, and the inverse builds it from x1 in the same way, so that
    // the two agree also where x1 approaches tau_min.
    auto [x1, omx1, det1] = fb.lepton_x_sample(inputs.at(0), tau, omtau, _power, power);
    auto x_min2 = fb.div(tau, x1);
    auto [x2, omx2, det2] =
        fb.lepton_x_sample(inputs.at(1), x_min2, fb.sub(1., x_min2), _power, power);
    auto s_hat = fb.mul(fb.mul(x1, x2), _s_lab);
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
    // An inverse map returns its own Jacobian, d(r)/d(x), the reciprocal of
    // that of the forward map. A point with x1 < tau_min, from another
    // channel, leaves x2 no range and gives r2 = 0 and a zero Jacobian.
    double power = _reduced_density ? 0. : _power;
    auto [tau, omtau] = tau_min(fb, conditions);
    auto x1 = inputs.at(0), omx1 = inputs.at(1), omx2 = inputs.at(3);
    auto [r1, det1] = fb.lepton_x_sample_inverse(omx1, omtau, _power, power);
    auto [r2, det2] = fb.lepton_x_sample_inverse(
        omx2, fb.sub(1., fb.div(tau, x1)), _power, power
    );
    return {{{"r1", r1}, {"r2", r2}}, fb.mul(det1, det2)};
}

LeptonLuminosityInvariant::LeptonLuminosityInvariant(
    double s_lab,
    double s_hat_min,
    double s_hat_max,
    double sampling_power,
    double mass,
    double width,
    double invariant_power,
    bool reduced_density,
    bool s_hat_min_condition
) :
    Mapping(
        "LeptonLuminosityInvariant",
        {{"r_s", batch_float}, {"r_x", batch_float}, {"r_swap", batch_float}},
        {{"x1", batch_float},
         {"omx1", batch_float},
         {"x2", batch_float},
         {"omx2", batch_float},
         {"s_hat", batch_float}},
        [&] {
            NamedVector<Type> conditions;
            if (s_hat_min_condition) {
                conditions.push_back("s_hat_min", batch_float);
            }
            return conditions;
        }()
    ),
    _s_lab(s_lab),
    _s_hat_min(s_hat_min),
    _s_hat_max(s_hat_max == 0. ? s_lab : s_hat_max),
    _power(sampling_power),
    _reduced_density(reduced_density),
    _s_hat_min_condition(s_hat_min_condition),
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

LeptonSHatInverseResult madspace::lepton_s_hat_inverse(
    FunctionBuilder& fb, Value s_gap, Value s_min, Value s_max, double power
) {
    auto batch = fb.batch_size({s_max});
    auto span = fb.sub(s_max, s_min);
    auto v = fb.div(s_gap, span);
    auto r = fb.pow(v, fb.full({1. - power, batch}));
    // the reciprocal of span v^p / (1 - p); pow gives zero for v = 0, where
    // the forward Jacobian vanishes, so that the inverse is infinite there as
    // the reciprocal of zero
    auto det = fb.div(
        fb.full({1. - power, batch}), fb.mul(span, fb.pow(v, fb.full({power, batch})))
    );
    return {r, det};
}

LeptonSplitResult madspace::lepton_split(
    FunctionBuilder& fb,
    Value s_hat,
    Value omtau,
    Value r_x,
    Value r_swap,
    double s_lab,
    double power,
    double jacobian_power
) {
    auto batch = fb.batch_size({s_hat});
    auto tau = fb.div(s_hat, fb.full({s_lab, batch}));

    // The drawn fraction is the larger one, over [sqrt(tau), 1], sampled as
    // LeptonLuminosity samples a beam. The width is built as
    // (1-tau)/(1+sqrt(tau)) rather than 1-sqrt(tau): the sampling puts 22% of
    // its points where tau has rounded to exactly 1, and the subtraction
    // gives zero there, which sends det_a to zero and throws the point away.
    // This form keeps it, because omtau never went through a cancellation.
    auto x_min = fb.sqrt(tau);
    auto span = fb.div(omtau, fb.add(fb.full({1., batch}), x_min));
    auto [x_a, omx_a, det_a] =
        fb.lepton_x_sample(r_x, x_min, span, power, jacobian_power);

    // the other follows from tau; 1/x_a is the Jacobian of x2 = tau/x1, and
    // with a reduced density omx_b^(-p) is the factor the density expects to
    // have been cancelled by the sampling, which for this beam never happened
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
    // omx_b^(jacobian_power - p), which is omx_b^(-p) for a reduced density
    // and one for a full one. Written as a power on purpose: pow returns 0 for
    // a non-positive base, so a degenerate point gets weight zero, where the
    // reciprocal would divide by zero and poison the whole batch with an
    // infinity.
    auto det_b =
        fb.div(fb.pow(omx_b, fb.full({jacobian_power - power, batch})), x_a);

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

LeptonSplitInverseResult madspace::lepton_split_inverse(
    FunctionBuilder& fb,
    Value x1,
    Value x2,
    Value omx1,
    Value omx2,
    double s_lab,
    double power,
    double jacobian_power
) {
    auto batch = fb.batch_size({x1});
    // the drawn fraction is the larger one, with the smaller 1 - x
    auto x_a = fb.max(x1, x2);
    auto omx_a = fb.min(omx1, omx2);
    auto omx_b = fb.max(omx1, omx2);
    // 1 - tau = (1 - x_a) + x_a (1 - x_b), and tau = x_a x_b as the forward map
    // builds x_b = tau / x_a
    auto omtau = fb.add(omx_a, fb.mul(x_a, omx_b));
    auto tau = fb.mul(x1, x2);
    auto x_min = fb.sqrt(tau);
    auto span = fb.div(omtau, fb.add(fb.full({1., batch}), x_min));
    auto [r_x, det_a] = fb.lepton_x_sample_inverse(omx_a, span, power, jacobian_power);
    // the second beam carries it where 1 - x2 < 1 - x1, as index 1 of
    // sample_discrete; x1 and x2 are often both one
    auto second =
        fb.cut_one(fb.sub(omx1, omx2), 0., std::numeric_limits<double>::infinity());
    auto r_swap = fb.add(fb.full({0.25, batch}), fb.mul(second, 0.5));
    // the reciprocals of 1 / x_a, omx_b^(jacobian_power - p) and 2 / s_lab
    auto det = fb.product({
        det_a,
        x_a,
        fb.pow(omx_b, fb.full({power - jacobian_power, batch})),
        fb.full({0.5 * s_lab, batch}),
    });
    return {r_x, r_swap, omtau, det};
}

Mapping::Result LeptonLuminosityInvariant::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto r_s = inputs[0], r_x = inputs[1], r_swap = inputs[2];
    auto batch = fb.batch_size({r_s});
    // the bound of the condition clipped to [0, s_hat_max], as in
    // LeptonLuminosity
    Value s_hat_min = _s_hat_min_condition
        ? fb.min(fb.max(conditions.at("s_hat_min"), 0.), _s_hat_max)
        : Value(_s_hat_min);
    auto inv = _invariant.build_forward(fb, {r_s}, {s_hat_min, _s_hat_max});
    auto s_hat = inv["invariant"];
    // This channel samples s_hat through a resonance, not from the endpoint,
    // so there is no exact gap to carry and 1-tau is the best available. That
    // is fine here: the Breit-Wigner puts its weight at the resonance, not at
    // tau -> 1 where the cancellation bites.
    auto omtau =
        fb.sub(fb.full({1., batch}), fb.div(s_hat, fb.full({_s_lab, batch})));
    auto split = lepton_split(
        fb, s_hat, omtau, r_x, r_swap, _s_lab, _power, _reduced_density ? 0. : _power
    );
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

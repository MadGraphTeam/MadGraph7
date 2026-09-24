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
    auto norm = fb.full({1. / (1. - _power), batch});

    double span1 = 1. - tau_min;
    auto r1 = fb.pow(fb.div(omx1, fb.full({span1, batch})), inv_exponent);
    auto det1 = fb.full({std::pow(span1, 1. - _power) / (1. - _power), batch});

    auto span2 = fb.sub(one, fb.div(fb.full({tau_min, batch}), x1));
    auto r2 = fb.pow(fb.div(omx2, span2), inv_exponent);
    auto det2 = fb.mul(fb.pow(span2, inv_exponent), norm);

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

std::pair<Value, Value> madspace::lepton_s_hat(
    FunctionBuilder& fb, Value s_min, Value s_max, Value r, double power
) {
    auto batch = fb.batch_size({s_max});
    auto span = fb.sub(s_max, s_min);
    auto v = fb.pow(r, fb.full({1. / (1. - power), batch}));
    auto s_hat = fb.sub(s_max, fb.mul(span, v));
    auto det = fb.mul(
        fb.mul(span, fb.pow(v, fb.full({power, batch}))),
        fb.full({1. / (1. - power), batch})
    );
    return {s_hat, det};
}

LeptonSplitResult madspace::lepton_split(
    FunctionBuilder& fb,
    Value s_hat,
    Value r_x,
    Value r_swap,
    double s_lab,
    double power
) {
    auto batch = fb.batch_size({s_hat});
    auto one = fb.full({1., batch});
    auto tau = fb.div(s_hat, fb.full({s_lab, batch}));

    // the drawn fraction is the larger one, over [sqrt(tau), 1]
    auto x_min = fb.sqrt(tau);
    auto span = fb.sub(one, x_min);
    auto omx_a = fb.mul(fb.pow(r_x, fb.full({1. / (1. - power), batch})), span);
    auto x_a = fb.sub(one, omx_a);
    auto det_a = fb.mul(
        fb.pow(span, fb.full({1. - power, batch})),
        fb.full({1. / (1. - power), batch})
    );

    // the other follows from tau; 1/x_a is the Jacobian of x2 = tau/x1, and
    // omx_b^(-p) is the factor the density expects to have been cancelled by
    // the sampling, which for this beam never happened
    auto x_b = fb.div(tau, x_a);
    auto omx_b = fb.sub(one, x_b);
    // Written as omx_b^(-p) rather than 1/omx_b^p on purpose. Once tau is
    // sampled right at 1, x_b = tau/x_a can round to just above 1 and omx_b
    // goes negative; pow returns 0 for a non-positive base, so this form
    // gives such a point weight zero -- which is what an x > 1 configuration
    // deserves -- where the reciprocal would divide by that zero and poison
    // the whole batch with an infinity.
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
    auto split = lepton_split(fb, s_hat, r_x, r_swap, _s_lab, _power);
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

#include "madspace/phasespace/invariants.hpp"

using namespace madspace;

Invariant::Invariant(double power, double mass, double width, bool return_virtuality) :
    Mapping(
        "Invariant",
        {{"random", batch_float}},
        return_virtuality
            ? NamedVector<Type>{{"invariant", batch_float}, {"virtuality", batch_float}}
            : NamedVector<Type>{{"invariant", batch_float}},
        {{"invariant_min", batch_float}, {"invariant_max", batch_float}}
    ),
    _power(power),
    _mass(mass),
    _width(width),
    _return_virtuality(return_virtuality) {}

Mapping::Result Invariant::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto r = inputs[0], s_min = conditions[0], s_max = conditions[1];
    Value s, virt, det;
    if (_width != 0) {
        std::tie(s, virt, det) =
            fb.breit_wigner_invariant(r, _mass, _width, s_min, s_max);
    } else if (_power == 0) {
        std::tie(s, det) = fb.uniform_invariant(r, s_min, s_max);
        virt = s;
    } else if (_power == 1) {
        std::tie(s, virt, det) = fb.stable_invariant(r, _mass, s_min, s_max);
    } else {
        std::tie(s, virt, det) = fb.stable_invariant_nu(r, _mass, _power, s_min, s_max);
    }
    if (_return_virtuality) {
        return {{{"invariant", s}}, det};
    } else {
        return {{{"invariant", s}, {"virtuality", virt}}, det};
    }
}

Mapping::Result Invariant::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto s = inputs[0], s_min = conditions[0], s_max = conditions[1];
    Value r, det;
    if (_width != 0) {
        std::tie(r, det) =
            fb.breit_wigner_invariant_inverse(s, _mass, _width, s_min, s_max);
    } else if (_power == 0) {
        std::tie(r, det) = fb.uniform_invariant_inverse(s, s_min, s_max);
    } else if (_power == 1) {
        std::tie(r, det) = fb.stable_invariant_inverse(s, _mass, s_min, s_max);
    } else {
        std::tie(r, det) =
            fb.stable_invariant_nu_inverse(s, _mass, _power, s_min, s_max);
    }
    return {{{"random", r}}, det};
}

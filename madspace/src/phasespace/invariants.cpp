#include "madspace/phasespace/invariants.h"

using namespace madspace;

Mapping::Result Invariant::build_forward_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    auto r = inputs[0], s_min = conditions[0], s_max = conditions[1];
    auto [s, det] = _width != 0
        ? fb.breit_wigner_invariant(r, _mass, _width, s_min, s_max)
        : _power == 0 ? fb.uniform_invariant(r, s_min, s_max)
        : _power == 1 ? fb.stable_invariant(r, _mass, s_min, s_max)
                      : fb.stable_invariant_nu(r, _mass, _power, s_min, s_max);
    return {{s}, det};
}

Mapping::Result Invariant::build_inverse_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    auto s = inputs[0], s_min = conditions[0], s_max = conditions[1];
    auto [r, det] = _width != 0
        ? fb.breit_wigner_invariant_inverse(s, _mass, _width, s_min, s_max)
        : _power == 0 ? fb.uniform_invariant_inverse(s, s_min, s_max)
        : _power == 1 ? fb.stable_invariant_inverse(s, _mass, s_min, s_max)
                      : fb.stable_invariant_nu_inverse(s, _mass, _power, s_min, s_max);
    return {{r}, det};
}

#include "madspace/phasespace/luminosity.h"

using namespace madspace;

Mapping::Result Luminosity::build_forward_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    auto r_s = inputs[0], r_x = inputs[1];
    auto [s_hat_vec, det_s] =
        _invariant.build_forward(fb, {r_s}, {_s_hat_min, _s_hat_max});
    auto s_hat = s_hat_vec[0];
    auto [x1, x2, det_x] = fb.r_to_x1x2(r_x, s_hat, _s_lab);
    auto det = fb.mul(det_s, det_x);
    return {{x1, x2, s_hat}, det};
}

Mapping::Result Luminosity::build_inverse_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    auto x1 = inputs[0], x2 = inputs[1], s_hat = inputs[2];
    auto [r_s_vec, det_s] =
        _invariant.build_inverse(fb, {s_hat}, {_s_hat_min, _s_hat_max});
    auto r_s = r_s_vec[0];
    auto [r_x, det_x] = fb.x1x2_to_r(x1, x2, _s_lab);
    auto det = fb.mul(det_s, det_x);
    return {{r_s, r_x}, det};
}

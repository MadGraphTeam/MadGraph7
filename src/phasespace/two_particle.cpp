#include "madevent/phasespace/two_particle.h"

using namespace madevent;

Mapping::Result TwoBodyDecay::build_forward_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    auto r_phi = inputs.at(0), r_cos_theta = inputs.at(1);
    auto m0 = inputs.at(2), m1 = inputs.at(3), m2 = inputs.at(4);
    auto [p1, p2, det] = _com
        ? fb.two_body_decay_com(r_phi, r_cos_theta, m0, m1, m2)
        : fb.two_body_decay(r_phi, r_cos_theta, m0, m1, m2, inputs.at(5));
    return {{p1, p2}, det};
}

Mapping::Result TwoBodyDecay::build_inverse_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    auto p1 = inputs.at(0), p2 = inputs.at(1);
    if (_com) {
        auto [r_phi, r_cos_theta, m0, m1, m2, det] =
            fb.two_body_decay_com_inverse(p1, p2);
        return {{r_phi, r_cos_theta, m0, m1, m2}, det};
    } else {
        auto [r_phi, r_cos_theta, m0, m1, m2, p0, det] =
            fb.two_body_decay_inverse(p1, p2);
        return {{r_phi, r_cos_theta, m0, m1, m2, p0}, det};
    }
}

Mapping::Result TwoToTwoParticleScattering::build_forward_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    auto r_phi = inputs.at(0), r_inv = inputs.at(1), m1 = inputs.at(2),
         m2 = inputs.at(3);
    auto p_in1 = conditions.at(0), p_in2 = conditions.at(1);
    auto [t_min, t_max] = fb.t_inv_min_max(p_in1, p_in2, m1, m2);
    auto [t_vec, det_inv] = _invariant.build_forward(fb, {r_inv}, {t_min, t_max});
    auto [p1, p2, det_scatter] = _com
        ? fb.two_to_two_particle_scattering_com(
              r_phi, p_in1, p_in2, t_vec.at(0), m1, m2
          )
        : fb.two_to_two_particle_scattering(r_phi, p_in1, p_in2, t_vec.at(0), m1, m2);
    return {{p1, p2}, fb.mul(det_inv, det_scatter)};
}

Mapping::Result TwoToTwoParticleScattering::build_inverse_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    auto p1 = inputs.at(0), p2 = inputs.at(1);
    auto p_in1 = conditions.at(0), p_in2 = conditions.at(1);
    auto [t_abs, t_min, t_max] = fb.t_inv_value_and_min_max(p_in1, p_in2, p1, p2);
    auto [r_inv_vec, det_inv] = _invariant.build_inverse(fb, {t_abs}, {t_min, t_max});
    auto [r_phi, m1, m2, det_scatter] = _com
        ? fb.two_to_two_particle_scattering_com_inverse(p1, p2, p_in1, p_in2)
        : fb.two_to_two_particle_scattering_inverse(p1, p2, p_in1, p_in2);
    return {{r_phi, r_inv_vec.at(0), m1, m2}, fb.mul(det_inv, det_scatter)};
}

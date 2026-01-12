#include "madevent/phasespace/three_particle.h"

using namespace madevent;

Mapping::Result ThreeBodyDecay::build_forward_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    auto r_e1 = inputs.at(0), r_e2 = inputs.at(1);
    auto r_phi = inputs.at(2), r_cos_theta = inputs.at(3), r_beta = inputs.at(4);
    auto m0 = inputs.at(5), m1 = inputs.at(6), m2 = inputs.at(7), m3 = inputs.at(8);
    auto [p1, p2, p3, det] = _com
        ? fb.three_body_decay_com(
              r_e1, r_e2, r_phi, r_cos_theta, r_beta, m0, m1, m2, m3
          )
        : fb.three_body_decay(
              r_e1, r_e2, r_phi, r_cos_theta, r_beta, m0, m1, m2, m3, inputs.at(9)
          );
    return {{p1, p2, p3}, det};
}

Mapping::Result ThreeBodyDecay::build_inverse_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    auto p1 = inputs.at(0), p2 = inputs.at(1), p3 = inputs.at(2);
    if (_com) {
        auto [r_e1, r_e2, r_phi, r_cos_theta, r_beta, m0, m1, m2, m3, det] =
            fb.three_body_decay_com_inverse(p1, p2, p3);
        return {{r_e1, r_e2, r_phi, r_cos_theta, r_beta, m0, m1, m2, m3}, det};
    } else {
        auto [r_e1, r_e2, r_phi, r_cos_theta, r_beta, m0, m1, m2, m3, p0, det] =
            fb.three_body_decay_inverse(p1, p2, p3);
        return {{r_e1, r_e2, r_phi, r_cos_theta, r_beta, m0, m1, m2, m3, p0}, det};
    }
}

Mapping::Result TwoToThreeParticleScattering::build_forward_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    auto r_choice = inputs.at(0), r_s23 = inputs.at(1), r_t1 = inputs.at(2),
         m1 = inputs.at(3), m2 = inputs.at(4);
    auto p_a = conditions.at(0), p_b = conditions.at(1), p_3 = conditions.at(2);
    auto [t1_min, t1_max] = fb.t_inv_min_max(p_a, fb.sub(p_b, p_3), m1, m2);
    auto [t1_vec, det_t1] = _t_invariant.build_forward(fb, {r_t1}, {t1_min, t1_max});
    auto [s23_min, s23_max] = fb.s23_min_max(p_a, p_b, p_3, t1_vec.at(0), m1, m2);
    auto [s23_vec, det_s23] =
        _s_invariant.build_forward(fb, {r_s23}, {s23_min, s23_max});
    auto det_inv = fb.mul(det_t1, det_s23);
    auto [index_choice, index_det] = fb.sample_discrete(r_choice, 2);
    auto [p1, p2, det_scatter] = fb.two_to_three_particle_scattering(
        index_choice, p_a, p_b, p_3, s23_vec.at(0), t1_vec.at(0), m1, m2
    );
    auto det_scatter_23 = fb.mul(index_det, det_scatter);
    return {{p1, p2}, fb.mul(det_inv, det_scatter_23)};
}

Mapping::Result TwoToThreeParticleScattering::build_inverse_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    auto p1 = inputs.at(0), p2 = inputs.at(1);
    auto p_a = conditions.at(0), p_b = conditions.at(1), p_3 = conditions.at(2);
    auto [t1_abs, t1_min, t1_max] =
        fb.t_inv_value_and_min_max(p_a, fb.sub(p_b, p_3), p1, p2);
    auto [r_t1_vec, det_t1] =
        _t_invariant.build_inverse(fb, {t1_abs}, {t1_min, t1_max});
    auto [s23, s23_min, s23_max] =
        fb.s23_value_and_min_max(p_a, p_b, p_3, t1_abs, p1, p2);
    auto [r_s23_vec, det_s23] =
        _s_invariant.build_inverse(fb, {s23}, {s23_min, s23_max});
    auto det_inv = fb.mul(det_t1, det_s23);
    auto [m1, m2, index_choice, det_scatter] =
        fb.two_to_three_particle_scattering_inverse(p1, p2, p_3, p_a, p_b, t1_abs, s23);
    auto [r_choice, index_det] = fb.sample_discrete_inverse(index_choice, 2);
    auto det_scatter_23 = fb.mul(index_det, det_scatter);
    return {
        {r_choice, r_s23_vec.at(0), r_t1_vec.at(0), m1, m2},
        fb.mul(det_inv, det_scatter_23)
    };
}

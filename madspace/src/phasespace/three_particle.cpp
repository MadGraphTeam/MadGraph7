#include "madspace/phasespace/three_particle.hpp"

using namespace madspace;

ThreeBodyDecay::ThreeBodyDecay(bool com) :
    Mapping(
        "ThreeBodyDecay",
        [&] {
            NamedVector<Type> input_types{
                {"random_energy1", batch_float},
                {"random_energy2", batch_float},
                {"random_phi", batch_float},
                {"random_cos_theta", batch_float},
                {"random_beta", batch_float},
                {"mass0", batch_float},
                {"mass1", batch_float},
                {"mass2", batch_float},
                {"mass3", batch_float},
            };
            if (!com) {
                input_types.push_back("com_momentum", batch_four_vec);
            }
            return input_types;
        }(),
        {{"momentum1", batch_four_vec},
         {"momentum2", batch_four_vec},
         {"momentum3", batch_four_vec}},
        {}
    ),
    _com(com) {}

Mapping::Result ThreeBodyDecay::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto [p1, p2, p3, det] = _com
        ? fb.three_body_decay_com(
              inputs["random_energy1"],
              inputs["random_energy2"],
              inputs["random_phi"],
              inputs["random_cos_theta"],
              inputs["random_beta"],
              inputs["mass0"],
              inputs["mass1"],
              inputs["mass2"],
              inputs["mass3"]
          )
        : fb.three_body_decay(
              inputs["random_energy1"],
              inputs["random_energy2"],
              inputs["random_phi"],
              inputs["random_cos_theta"],
              inputs["random_beta"],
              inputs["mass0"],
              inputs["mass1"],
              inputs["mass2"],
              inputs["mass3"],
              inputs["com_momentum"]
          );
    return {{{"momentum1", p1}, {"momentum2", p2}, {"momentum3", p3}}, det};
}

Mapping::Result ThreeBodyDecay::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    if (_com) {
        auto [r_e1, r_e2, r_phi, r_cos_theta, r_beta, m0, m1, m2, m3, det] =
            fb.three_body_decay_com_inverse(
                inputs["momentum1"], inputs["momentum2"], inputs["momentum3"]
            );
        return {
            {input_types().keys(),
             {r_e1, r_e2, r_phi, r_cos_theta, r_beta, m0, m1, m2, m3}},
            det
        };
    } else {
        auto [r_e1, r_e2, r_phi, r_cos_theta, r_beta, m0, m1, m2, m3, p0, det] =
            fb.three_body_decay_inverse(
                inputs["momentum1"], inputs["momentum2"], inputs["momentum3"]
            );
        return {
            {input_types().keys(),
             {r_e1, r_e2, r_phi, r_cos_theta, r_beta, m0, m1, m2, m3, p0}},
            det
        };
    }
}

TwoToThreeParticleScattering::TwoToThreeParticleScattering(
    double t_invariant_power,
    double t_mass,
    double t_width,
    double s_invariant_power,
    double s_mass,
    double s_width,
    bool has_cut,
    bool arcsine_s23,
    bool p12_condition
) :
    Mapping(
        "TwoToThreeParticleScattering",
        {{"discrete_choice", batch_int},
         {"random_s23", batch_float},
         {"random_t1", batch_float},
         {"mass1", batch_float},
         {"mass2", batch_float}},
        {{"momentum1", batch_four_vec}, {"momentum2", batch_four_vec}},
        [&] {
            NamedVector<Type> cond{
                {"momentum_in1", batch_four_vec},
                {p12_condition ? "momentum12" : "momentum_in2", batch_four_vec},
                {"momentum3", batch_four_vec}
            };
            if (has_cut) {
                cond.push_back("etmin_1", batch_float);
                cond.push_back("etmin_2", batch_float);
                cond.push_back("drcut", batch_float);
                cond.push_back("s23_min_cut", batch_float);
            }
            return cond;
        }()
    ),
    _t_invariant(t_invariant_power, t_mass, t_width),
    _s_invariant(s_invariant_power, s_mass, s_width),
    _s_power(s_invariant_power),
    _s_mass(s_mass),
    _s_width(s_width),
    _has_cut(has_cut),
    _arcsine_s23(arcsine_s23),
    _p12_condition(p12_condition) {}

std::array<Value, 3> TwoToThreeParticleScattering::split_conditions(
    FunctionBuilder& fb, const NamedVector<Value>& conditions
) const {
    // The kernels take the outgoing system p_12 = p1 + p2 itself. By default it
    // is formed from the incoming momenta as pa + pb - p3; a caller that already
    // holds p_12 more precisely (ColorOrderedMapping, where it can be a soft
    // system next to the beams) passes it as the second condition instead.
    auto p_a = conditions.at(0), p_3 = conditions.at(2);
    if (_p12_condition) {
        auto p_12 = conditions.at(1);
        return {p_a, p_12, fb.sub(p_12, p_a)};
    }
    auto p_b = conditions.at(1);
    return {p_a, fb.sub(fb.add(p_a, p_b), p_3), fb.sub(p_b, p_3)};
}

Mapping::Result TwoToThreeParticleScattering::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto index_choice = inputs.at(0), r_s23 = inputs.at(1), r_t1 = inputs.at(2),
         m1 = inputs.at(3), m2 = inputs.at(4);
    auto p_3 = conditions.at(2);
    auto [p_a, p_12, p_c] = split_conditions(fb, conditions);
    auto [t1_min, t1_max] = _has_cut
        ? fb.t_inv_min_max_cut(p_a, p_c, m1, m2, conditions.at(3), conditions.at(4))
        : fb.t_inv_min_max(p_a, p_c, m1, m2);
    auto t_inv_result = _t_invariant.build_forward(fb, {r_t1}, {t1_min, t1_max});
    auto [s23_min, s23_max] = _has_cut
        ? fb.s23_min_max_cut(
              p_a,
              p_12,
              p_3,
              t_inv_result["invariant"],
              m1,
              m2,
              conditions.at(3),
              conditions.at(4),
              conditions.at(5),
              conditions.at(6)
          )
        : fb.s23_min_max(p_a, p_12, p_3, t_inv_result["invariant"], m1, m2);
    // The scattering kernel takes the position of s23 in its kinematic range,
    // u and 1 - u. Without cuts the sampled and the kinematic range coincide.
    auto [s23_phys_min, s23_phys_max] = _has_cut
        ? fb.s23_min_max(p_a, p_12, p_3, t_inv_result["invariant"], m1, m2)
        : std::array<Value, 2>{s23_min, s23_max};
    Value u, u_c, det_s23;
    if (fused_s23()) {
        // arcsine map and importance sampling in one step, carrying u and
        // 1 - u at full relative precision up to the edges
        auto [u_out, u_c_out, det_out] = fb.s23_arcsine_sample(
            r_s23,
            s23_min,
            s23_max,
            s23_phys_min,
            s23_phys_max,
            Value(_s_power),
            Value(_s_mass)
        );
        u = u_out;
        u_c = u_c_out;
        det_s23 = det_out;
    } else {
        Value x_s23 = r_s23;
        Value det_x;
        if (_arcsine_s23) {
            auto [x, det_arcsine] =
                fb.s23_arcsine(r_s23, s23_min, s23_max, s23_phys_min, s23_phys_max);
            x_s23 = x;
            det_x = det_arcsine;
        }
        auto s23_inv_result =
            _s_invariant.build_forward(fb, {x_s23}, {s23_min, s23_max});
        det_s23 = _arcsine_s23 ? fb.mul(s23_inv_result["det"], det_x)
                               : s23_inv_result["det"];
        auto [u_out, u_c_out] =
            fb.s23_position(s23_inv_result["invariant"], s23_phys_min, s23_phys_max);
        u = u_out;
        u_c = u_c_out;
    }
    auto det_inv = fb.mul(t_inv_result["det"], det_s23);
    auto [p1, p2, det_scatter] = fb.two_to_three_particle_scattering(
        index_choice,
        p_a,
        p_12,
        p_3,
        u,
        u_c,
        t_inv_result["invariant"],
        m1,
        m2
    );
    return {{{"momentum1", p1}, {"momentum2", p2}}, fb.mul(det_inv, det_scatter)};
}

Mapping::Result TwoToThreeParticleScattering::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto p1 = inputs.at(0), p2 = inputs.at(1);
    auto p_3 = conditions.at(2);
    auto [p_a, p_12, p_c] = split_conditions(fb, conditions);
    auto [t1_abs, t1_min, t1_max] = _has_cut
        ? fb.t_inv_value_and_min_max_cut(
              p_a, p_c, p1, p2, conditions.at(3), conditions.at(4)
          )
        : fb.t_inv_value_and_min_max(p_a, p_c, p1, p2);
    auto t_inv_result = _t_invariant.build_inverse(fb, {t1_abs}, {t1_min, t1_max});
    auto [s23, s23_min, s23_max] = _has_cut
        ? fb.s23_value_and_min_max_cut(
              p_a,
              p_3,
              t1_abs,
              p1,
              p2,
              conditions.at(3),
              conditions.at(4),
              conditions.at(5),
              conditions.at(6)
          )
        : fb.s23_value_and_min_max(p_a, p_3, t1_abs, p1, p2);
    auto [m1, m2, index_choice, u, u_c, det_scatter] =
        fb.two_to_three_particle_scattering_inverse(p1, p2, p_3, p_a, p_12, t1_abs);
    Value s23_phys_min = s23_min, s23_phys_max = s23_max;
    if (_has_cut && _arcsine_s23) {
        auto [s23_phys, phys_min, phys_max] =
            fb.s23_value_and_min_max(p_a, p_3, t1_abs, p1, p2);
        s23_phys_min = phys_min;
        s23_phys_max = phys_max;
    }
    Value r_s23, det_s23;
    if (fused_s23()) {
        // u and 1 - u read off the azimuth, at full relative precision
        auto [r, det_r] = fb.s23_arcsine_sample_inverse(
            u,
            u_c,
            s23_min,
            s23_max,
            s23_phys_min,
            s23_phys_max,
            Value(_s_power),
            Value(_s_mass)
        );
        r_s23 = r;
        det_s23 = det_r;
    } else {
        auto s23_inv_result =
            _s_invariant.build_inverse(fb, {s23}, {s23_min, s23_max});
        r_s23 = s23_inv_result["random"];
        det_s23 = s23_inv_result["det"];
        if (_arcsine_s23) {
            auto [r, det_r] = fb.s23_arcsine_inverse(
                r_s23, s23_min, s23_max, s23_phys_min, s23_phys_max
            );
            r_s23 = r;
            det_s23 = fb.mul(det_s23, det_r);
        }
    }
    auto det_inv = fb.mul(t_inv_result["det"], det_s23);
    return {
        {{"discrete_choice", index_choice},
         {"random_s23", r_s23},
         {"random_t1", t_inv_result["random"]},
         {"mass1", m1},
         {"mass2", m2}},
        fb.mul(det_inv, det_scatter)
    };
}

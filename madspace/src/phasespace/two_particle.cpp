#include "madspace/phasespace/two_particle.hpp"

using namespace madspace;

namespace {
bool has_cos_theta_window(bool com, double pt_min, double y_max1, double y_max2) {
    return com && (pt_min > 0. || y_max1 >= 0. || y_max2 >= 0.);
}
} // namespace

TwoBodyDecay::TwoBodyDecay(bool com, double pt_min, double y_max1, double y_max2) :
    Mapping(
        "TwoBodyDecay",
        [&] {
            NamedVector<Type> input_types{
                {"random_phi", batch_float},
                {"random_cos_theta", batch_float},
                {"mass0", batch_float},
                {"mass1", batch_float},
                {"mass2", batch_float},
            };
            if (!com) {
                input_types.push_back("com_momentum", batch_four_vec);
            }
            return input_types;
        }(),
        {{"momentum1", batch_four_vec}, {"momentum2", batch_four_vec}},
        has_cos_theta_window(com, pt_min, y_max1, y_max2)
            ? NamedVector<Type>{{"x1", batch_float}, {"x2", batch_float}}
            : NamedVector<Type>{}
    ),
    _com(com),
    _window(has_cos_theta_window(com, pt_min, y_max1, y_max2)),
    _pt_min(pt_min),
    _y_max1(y_max1),
    _y_max2(y_max2) {}

Mapping::Result TwoBodyDecay::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto r_phi = inputs.at(0), r_cos_theta = inputs.at(1);
    auto m0 = inputs.at(2), m1 = inputs.at(3), m2 = inputs.at(4);
    Value det_window = 1.;
    if (_window) {
        auto [r_window, det] = fb.com_cos_theta_window(
            r_cos_theta,
            m0,
            m1,
            m2,
            conditions.at(0),
            conditions.at(1),
            _pt_min,
            _y_max1,
            _y_max2
        );
        r_cos_theta = r_window;
        det_window = det;
    }
    auto [p1, p2, det] = _com
        ? fb.two_body_decay_com(r_phi, r_cos_theta, m0, m1, m2)
        : fb.two_body_decay(r_phi, r_cos_theta, m0, m1, m2, inputs.at(5));
    return {
        {{"momentum1", p1}, {"momentum2", p2}}, _window ? fb.mul(det, det_window) : det
    };
}

Mapping::Result TwoBodyDecay::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto p1 = inputs.at(0), p2 = inputs.at(1);
    if (_com) {
        auto [r_phi, r_cos_theta, m0, m1, m2, det] =
            fb.two_body_decay_com_inverse(p1, p2);
        if (_window) {
            auto [r_free, det_window] = fb.com_cos_theta_window_inverse(
                r_cos_theta,
                m0,
                m1,
                m2,
                conditions.at(0),
                conditions.at(1),
                _pt_min,
                _y_max1,
                _y_max2
            );
            r_cos_theta = r_free;
            det = fb.mul(det, det_window);
        }
        return {
            {{"random_phi", r_phi},
             {"random_cos_theta", r_cos_theta},
             {"mass0", m0},
             {"mass1", m1},
             {"mass2", m2}},
            det
        };
    } else {
        auto [r_phi, r_cos_theta, m0, m1, m2, p0, det] =
            fb.two_body_decay_inverse(p1, p2);
        return {
            {{"random_phi", r_phi},
             {"random_cos_theta", r_cos_theta},
             {"mass0", m0},
             {"mass1", m1},
             {"mass2", m2},
             {"com_momentum", p0}},
            det
        };
    }
}

TwoToTwoParticleScattering::TwoToTwoParticleScattering(
    bool com,
    double invariant_power,
    double mass,
    double width,
    bool has_cut,
    double y_max1,
    double y_max2,
    double beam_sign,
    bool massive_beams
) :
    Mapping(
        "TwoToTwoParticleScattering",
        {{"random_phi", batch_float},
         {"random_inv", batch_float},
         {"mass1", batch_float},
         {"mass2", batch_float}},
        {{"momentum1", batch_four_vec}, {"momentum2", batch_four_vec}},
        [&] {
            NamedVector<Type> cond{
                {"momentum_in1", batch_four_vec}, {"momentum_in2", batch_four_vec}
            };
            if (has_cut) {
                cond.push_back("etmin_1", batch_float);
                cond.push_back("etmin_2", batch_float);
            }
            if (y_max1 >= 0. || y_max2 >= 0.) {
                cond.push_back("x1", batch_float);
                cond.push_back("x2", batch_float);
            }
            return cond;
        }()
    ),
    _com(com),
    _invariant(invariant_power, mass, width),
    _has_cut(has_cut),
    _rapidity_window(y_max1 >= 0. || y_max2 >= 0.),
    _y_max1(y_max1),
    _y_max2(y_max2),
    _beam_sign(beam_sign),
    _massive_beams(massive_beams) {}

std::array<Value, 2> TwoToTwoParticleScattering::t_range(
    FunctionBuilder& fb, const NamedVector<Value>& conditions, Value m1, Value m2
) const {
    auto p_in1 = conditions.at(0), p_in2 = conditions.at(1);
    if (_has_cut) {
        auto range = _massive_beams
            ? &FunctionBuilder::t_inv_min_max_cut_massive
            : &FunctionBuilder::t_inv_min_max_cut;
        auto [t_min, t_max] =
            (fb.*range)(p_in1, p_in2, m1, m2, conditions.at(2), conditions.at(3));
        return {t_min, t_max};
    }
    auto range = _massive_beams
        ? &FunctionBuilder::t_inv_min_max_massive
        : &FunctionBuilder::t_inv_min_max;
    auto [t_min, t_max] = (fb.*range)(p_in1, p_in2, m1, m2);
    return {t_min, t_max};
}

std::array<Value, 2> TwoToTwoParticleScattering::rapidity_clamp(
    FunctionBuilder& fb,
    const NamedVector<Value>& conditions,
    Value t_min,
    Value t_max,
    Value m1,
    Value m2
) const {
    if (!_rapidity_window) {
        return {t_min, t_max};
    }
    std::size_t x_index = _has_cut ? 4 : 2;
    auto clamp = _massive_beams
        ? &FunctionBuilder::t_rapidity_clamp_massive
        : &FunctionBuilder::t_rapidity_clamp;
    auto [lo, hi] = (fb.*clamp)(
        t_min,
        t_max,
        conditions.at(0),
        conditions.at(1),
        m1,
        m2,
        conditions.at(x_index),
        conditions.at(x_index + 1),
        _beam_sign,
        _y_max1,
        _y_max2
    );
    return {lo, hi};
}

// With massive beams the range can reach below |t| = 0, and the Invariant
// samples |t| shifted; see Invariant::build_forward_shifted.
std::array<Value, 2> TwoToTwoParticleScattering::sample_t(
    FunctionBuilder& fb, Value r, Value t_min, Value t_max
) const {
    if (!_massive_beams) {
        auto result = _invariant.build_forward(fb, {r}, {t_min, t_max});
        return {result["invariant"], result["det"]};
    }
    return _invariant.build_forward_shifted(fb, r, t_min, t_max);
}

std::array<Value, 2> TwoToTwoParticleScattering::unsample_t(
    FunctionBuilder& fb, Value t_abs, Value t_min, Value t_max
) const {
    if (!_massive_beams) {
        auto result = _invariant.build_inverse(fb, {t_abs}, {t_min, t_max});
        return {result["random"], result["det"]};
    }
    return _invariant.build_inverse_shifted(fb, t_abs, t_min, t_max);
}

Mapping::Result TwoToTwoParticleScattering::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto r_phi = inputs.at(0), r_inv = inputs.at(1), m1 = inputs.at(2),
         m2 = inputs.at(3);
    auto p_in1 = conditions.at(0), p_in2 = conditions.at(1);
    auto [t_min_kin, t_max_kin] = t_range(fb, conditions, m1, m2);
    auto [t_min, t_max] = rapidity_clamp(fb, conditions, t_min_kin, t_max_kin, m1, m2);
    auto [t_abs, det_t] = sample_t(fb, r_inv, t_min, t_max);
    auto [p1, p2, det_scatter] = _com
        ? fb.two_to_two_particle_scattering_com(r_phi, p_in1, p_in2, t_abs, m1, m2)
        : fb.two_to_two_particle_scattering(r_phi, p_in1, p_in2, t_abs, m1, m2);
    return {{{"momentum1", p1}, {"momentum2", p2}}, fb.mul(det_t, det_scatter)};
}

Mapping::Result TwoToTwoParticleScattering::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    return inverse(fb, inputs, conditions, std::nullopt);
}

NamedVector<Value> TwoToTwoParticleScattering::build_inverse_with_masses(
    FunctionBuilder& fb,
    const ValueVec& inputs,
    const ValueVec& conditions,
    Value m1,
    Value m2
) const {
    auto [outputs, det] = inverse(
        fb,
        {output_types().keys(), inputs},
        {condition_types().keys(), conditions},
        std::array<Value, 2>{m1, m2}
    );
    outputs.push_back("det", det);
    return outputs;
}

Mapping::Result TwoToTwoParticleScattering::inverse(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions,
    const std::optional<std::array<Value, 2>>& masses
) const {
    auto p1 = inputs.at(0), p2 = inputs.at(1);
    auto p_in1 = conditions.at(0), p_in2 = conditions.at(1);
    std::array<Value, 3> t_values;
    if (_has_cut) {
        auto value_and_range = _massive_beams
            ? &FunctionBuilder::t_inv_value_and_min_max_cut_massive
            : &FunctionBuilder::t_inv_value_and_min_max_cut;
        t_values = (fb.*value_and_range)(
            p_in1, p_in2, p1, p2, conditions.at(2), conditions.at(3)
        );
    } else {
        auto value_and_range = _massive_beams
            ? &FunctionBuilder::t_inv_value_and_min_max_massive
            : &FunctionBuilder::t_inv_value_and_min_max;
        t_values = (fb.*value_and_range)(p_in1, p_in2, p1, p2);
    }
    Value t_abs = t_values.at(0);
    std::array<Value, 2> t_kin{t_values.at(1), t_values.at(2)};
    auto [mass1, mass2] =
        masses ? *masses : std::array<Value, 2>{fb.obs_mass(p1), fb.obs_mass(p2)};
    if (masses) {
        // |t| without cancellations of order E^2, and the range of the forward
        // map, from the masses
        t_abs = fb.t_inv_value(p_in1, p_in2, p1, p2, mass1, mass2);
        t_kin = t_range(fb, conditions, mass1, mass2);
    }
    auto [t_min, t_max] = _rapidity_window
        ? rapidity_clamp(fb, conditions, t_kin.at(0), t_kin.at(1), mass1, mass2)
        : t_kin;
    auto [r_inv, det_t] = unsample_t(fb, t_abs, t_min, t_max);
    auto [r_phi, m1, m2, det_scatter] = _com
        ? fb.two_to_two_particle_scattering_com_inverse(p1, p2, p_in1, p_in2)
        : fb.two_to_two_particle_scattering_inverse(p1, p2, p_in1, p_in2);
    return {
        {{"random_phi", r_phi}, {"random_inv", r_inv}, {"mass1", m1}, {"mass2", m2}},
        fb.mul(det_t, det_scatter)
    };
}

DoubleT::DoubleT(
    double t1_invariant_power,
    double t1_mass,
    double t1_width,
    double t2_invariant_power,
    double t2_mass,
    double t2_width,
    bool has_cut,
    double y_max1,
    double y_max2,
    double beam_sign,
    bool massive_beams
) :
    Mapping(
        "DoubleT",
        {{"random_phi", batch_float},
         {"random_t1", batch_float},
         {"random_t2", batch_float}},
        {{"momentum1", batch_four_vec}, {"momentum2", batch_four_vec}},
        [&] {
            NamedVector<Type> cond{
                {"momentum_in1", batch_four_vec},
                {"momentum_in2", batch_four_vec},
                {"mass1", batch_float},
                {"mass_rest_min", batch_float}
            };
            if (has_cut) {
                cond.push_back("etmin_i", batch_float);
                cond.push_back("etmin_ir", batch_float);
            }
            if (y_max1 >= 0. || y_max2 >= 0.) {
                cond.push_back("x1", batch_float);
                cond.push_back("x2", batch_float);
            }
            return cond;
        }()
    ),
    _t1_invariant(t1_invariant_power, t1_mass, t1_width),
    _t2_invariant(t2_invariant_power, t2_mass, t2_width),
    _has_cut(has_cut),
    _rapidity_window(y_max1 >= 0. || y_max2 >= 0.),
    _y_max1(y_max1),
    _y_max2(y_max2),
    _beam_sign(beam_sign),
    _massive_beams(massive_beams) {}

std::array<Value, 2> DoubleT::rapidity_clamp_t1(
    FunctionBuilder& fb,
    const NamedVector<Value>& conditions,
    Value t1_min,
    Value t1_max
) const {
    if (!_rapidity_window) {
        return {t1_min, t1_max};
    }
    std::size_t x_index = _has_cut ? 6 : 4;
    auto clamp = _massive_beams
        ? &FunctionBuilder::t1_rapidity_clamp_doublet_massive
        : &FunctionBuilder::t1_rapidity_clamp_doublet;
    auto [lo, hi] = (fb.*clamp)(
        t1_min,
        t1_max,
        conditions.at(0),
        conditions.at(1),
        conditions.at(2),
        conditions.at(3),
        _has_cut ? conditions.at(4) : Value(0.),
        _has_cut ? conditions.at(5) : Value(0.),
        conditions.at(x_index),
        conditions.at(x_index + 1),
        _beam_sign,
        _y_max1,
        _y_max2
    );
    return {lo, hi};
}

std::array<Value, 2> DoubleT::rapidity_clamp(
    FunctionBuilder& fb,
    const NamedVector<Value>& conditions,
    Value t2_min,
    Value t2_max,
    Value t1_abs
) const {
    if (!_rapidity_window) {
        return {t2_min, t2_max};
    }
    std::size_t x_index = _has_cut ? 6 : 4;
    auto clamp = _massive_beams
        ? &FunctionBuilder::t2_rapidity_clamp_doublet_massive
        : &FunctionBuilder::t2_rapidity_clamp_doublet;
    auto [lo, hi] = (fb.*clamp)(
        t2_min,
        t2_max,
        conditions.at(0),
        conditions.at(1),
        conditions.at(2),
        t1_abs,
        conditions.at(x_index),
        conditions.at(x_index + 1),
        _beam_sign,
        _y_max1,
        _y_max2
    );
    return {lo, hi};
}

// With massive beams a range can reach below |t| = 0; see
// Invariant::build_forward_shifted.
std::array<Value, 2> DoubleT::sample_t(
    FunctionBuilder& fb, const Invariant& invariant, Value r, Value t_min, Value t_max
) const {
    if (!_massive_beams) {
        auto result = invariant.build_forward(fb, {r}, {t_min, t_max});
        return {result["invariant"], result["det"]};
    }
    return invariant.build_forward_shifted(fb, r, t_min, t_max);
}

std::array<Value, 2> DoubleT::unsample_t(
    FunctionBuilder& fb,
    const Invariant& invariant,
    Value t_abs,
    Value t_min,
    Value t_max
) const {
    if (!_massive_beams) {
        auto result = invariant.build_inverse(fb, {t_abs}, {t_min, t_max});
        return {result["random"], result["det"]};
    }
    return invariant.build_inverse_shifted(fb, t_abs, t_min, t_max);
}

Mapping::Result DoubleT::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto r_phi = inputs.at(0), r_t1 = inputs.at(1), r_t2 = inputs.at(2);
    auto p_in1 = conditions.at(0), p_in2 = conditions.at(1);
    auto m1 = conditions.at(2);
    auto mir_min = conditions.at(3);
    auto etmin_i = _has_cut ? conditions.at(4) : Value(0.);
    auto etmin_ir = _has_cut ? conditions.at(5) : Value(0.);

    auto t1_range = _massive_beams
        ? &FunctionBuilder::t1_inv_min_max_doublet_massive
        : &FunctionBuilder::t1_inv_min_max_doublet;
    auto [t1_min_kin, t1_max_kin] =
        (fb.*t1_range)(p_in1, p_in2, m1, mir_min, etmin_i, etmin_ir);
    auto [t1_min, t1_max] = rapidity_clamp_t1(fb, conditions, t1_min_kin, t1_max_kin);
    auto [t1_abs, det_t1] = sample_t(fb, _t1_invariant, r_t1, t1_min, t1_max);

    auto t2_range = _massive_beams
        ? &FunctionBuilder::t2_inv_min_max_doublet_massive
        : &FunctionBuilder::t2_inv_min_max_doublet;
    auto [t2_min_kin, t2_max_kin] =
        (fb.*t2_range)(p_in1, p_in2, m1, mir_min, t1_abs, etmin_i, etmin_ir);
    auto [t2_min, t2_max] =
        rapidity_clamp(fb, conditions, t2_min_kin, t2_max_kin, t1_abs);
    auto [t2_abs, det_t2] = sample_t(fb, _t2_invariant, r_t2, t2_min, t2_max);

    auto scattering = _massive_beams
        ? &FunctionBuilder::double_t_scattering_massive
        : &FunctionBuilder::double_t_scattering;
    auto [p1, p2, det_scatter] =
        (fb.*scattering)(r_phi, p_in1, p_in2, t1_abs, t2_abs, m1);

    auto det_inv = fb.mul(det_t1, det_t2);
    return {{{"momentum1", p1}, {"momentum2", p2}}, fb.mul(det_inv, det_scatter)};
}

Mapping::Result DoubleT::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto p1 = inputs.at(0), p2 = inputs.at(1);
    auto p_in1 = conditions.at(0), p_in2 = conditions.at(1);
    auto m1 = conditions.at(2);
    auto mir_min = conditions.at(3);
    auto etmin_i = _has_cut ? conditions.at(4) : Value(0.);
    auto etmin_ir = _has_cut ? conditions.at(5) : Value(0.);

    auto scattering_inverse = _massive_beams
        ? &FunctionBuilder::double_t_scattering_inverse_massive
        : &FunctionBuilder::double_t_scattering_inverse;
    auto [r_phi, det_scatter] = (fb.*scattering_inverse)(p1, p2, p_in1, p_in2);

    auto t1_value_and_range = _massive_beams
        ? &FunctionBuilder::t1_inv_value_and_min_max_doublet_massive
        : &FunctionBuilder::t1_inv_value_and_min_max_doublet;
    auto [t1_abs, t1_min_kin, t1_max_kin] =
        (fb.*t1_value_and_range)(p_in1, p_in2, p1, m1, mir_min, etmin_i, etmin_ir);
    auto [t1_min, t1_max] = rapidity_clamp_t1(fb, conditions, t1_min_kin, t1_max_kin);
    auto [r_t1, det_t1] = unsample_t(fb, _t1_invariant, t1_abs, t1_min, t1_max);

    auto t2_value_and_range = _massive_beams
        ? &FunctionBuilder::t2_inv_value_and_min_max_doublet_massive
        : &FunctionBuilder::t2_inv_value_and_min_max_doublet;
    auto [t2_abs, t2_min_kin, t2_max_kin] = (fb.*t2_value_and_range)(
        p_in1, p_in2, p1, m1, mir_min, t1_abs, etmin_i, etmin_ir
    );
    auto [t2_min, t2_max] =
        rapidity_clamp(fb, conditions, t2_min_kin, t2_max_kin, t1_abs);
    auto [r_t2, det_t2] = unsample_t(fb, _t2_invariant, t2_abs, t2_min, t2_max);

    auto det_inv = fb.mul(det_t1, det_t2);
    return {
        {{"random_phi", r_phi}, {"random_t1", r_t1}, {"random_t2", r_t2}},
        fb.mul(det_inv, det_scatter)
    };
}

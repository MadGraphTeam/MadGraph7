#include "madspace/phasespace/fks_radiation.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

using namespace madspace;

namespace {

constexpr std::size_t INCOMING_COUNT = 2;

std::size_t checked_born_count(
    std::size_t born_particle_count, std::size_t i_fks, std::size_t j_fks, bool leptonic
) {
    if (born_particle_count < INCOMING_COUNT + 1) {
        throw std::invalid_argument("FKSRadiation: the Born needs a final state");
    }
    auto real_count = born_particle_count + 1;
    if (i_fks >= real_count || j_fks >= real_count) {
        throw std::invalid_argument(
            "FKSRadiation: i_fks and j_fks must be below the number of particles "
            "of the real process, " +
            std::to_string(real_count)
        );
    }
    if (i_fks < INCOMING_COUNT || i_fks == j_fks) {
        throw std::invalid_argument(
            "FKSRadiation: i_fks must be a final-state particle different from j_fks"
        );
    }
    if (j_fks < INCOMING_COUNT && leptonic) {
        throw std::invalid_argument(
            "FKSRadiation: an incoming emitter changes the momentum fractions, it "
            "needs leptonic = false"
        );
    }
    return born_particle_count;
}

std::vector<std::size_t> checked_resonance(
    std::size_t born_particle_count,
    std::size_t i_fks,
    std::size_t j_fks,
    std::vector<std::size_t> resonance
) {
    if (resonance.empty()) {
        return resonance;
    }
    std::sort(resonance.begin(), resonance.end());
    if (j_fks < INCOMING_COUNT) {
        throw std::invalid_argument(
            "FKSRadiation: a resonance needs a final-state emitter"
        );
    }
    if (std::adjacent_find(resonance.begin(), resonance.end()) != resonance.end()) {
        throw std::invalid_argument(
            "FKSRadiation: every member of the resonance must appear once"
        );
    }
    for (auto r : resonance) {
        if (r < INCOMING_COUNT || r > born_particle_count) {
            throw std::invalid_argument(
                "FKSRadiation: the members of the resonance must be final-state "
                "particles of the real process"
            );
        }
    }
    auto has = [&](std::size_t r) {
        return std::binary_search(resonance.begin(), resonance.end(), r);
    };
    if (!has(i_fks) || !has(j_fks) || resonance.size() < 3) {
        throw std::invalid_argument(
            "FKSRadiation: the resonance must contain i, j and at least one more "
            "particle, which recoils"
        );
    }
    return resonance;
}

NamedVector<Type>
radiation_input_types(std::size_t born_particle_count, bool leptonic) {
    NamedVector<Type> types{
        {"r_xi", batch_float},
        {"r_y", batch_float},
        {"r_phi", batch_float},
        {"born_momenta", batch_four_vec_array(born_particle_count)},
    };
    if (!leptonic) {
        types.push_back("x1", batch_float);
        types.push_back("x2", batch_float);
    }
    return types;
}

NamedVector<Type>
radiation_output_types(std::size_t born_particle_count, bool leptonic) {
    NamedVector<Type> types{{"momenta", batch_four_vec_array(born_particle_count + 1)}};
    if (!leptonic) {
        types.push_back("x1", batch_float);
        types.push_back("x2", batch_float);
    }
    return types;
}

} // namespace

FKSRadiation::FKSRadiation(
    std::size_t born_particle_count,
    std::size_t i_fks,
    std::size_t j_fks,
    bool leptonic,
    double s_hat_min,
    double emitter_mass,
    InitialMap initial_map,
    const std::vector<double>& born_masses,
    const std::vector<std::size_t>& resonance
) :
    Mapping(
        "FKSRadiation",
        radiation_input_types(
            checked_born_count(born_particle_count, i_fks, j_fks, leptonic), leptonic
        ),
        radiation_output_types(born_particle_count, leptonic),
        {}
    ),
    _born_particle_count(born_particle_count),
    _i_fks(i_fks),
    _j_fks(j_fks),
    _leptonic(leptonic),
    _s_hat_min(s_hat_min),
    _emitter_mass(emitter_mass),
    _initial_map(initial_map),
    _final_masses(born_particle_count - INCOMING_COUNT, -1.),
    _resonance(checked_resonance(born_particle_count, i_fks, j_fks, resonance)) {
    if (!_resonance.empty()) {
        _born_resonance_mask.assign(born_particle_count, 0.);
        _real_resonance_mask.assign(born_particle_count + 1, 0.);
        for (auto r : _resonance) {
            _real_resonance_mask.at(r) = 1.;
            if (r != i_fks) {
                _born_resonance_mask.at(born_index(r)) = 1.;
            }
        }
    }
    if (!born_masses.empty()) {
        if (born_masses.size() != born_particle_count) {
            throw std::invalid_argument(
                "FKSRadiation: born_masses needs one mass per particle of the Born"
            );
        }
        _final_masses.assign(born_masses.begin() + INCOMING_COUNT, born_masses.end());
    }
    if (s_hat_min < 0.) {
        throw std::invalid_argument("FKSRadiation: s_hat_min must not be negative");
    }
    if (emitter_mass < 0. || (emitter_mass > 0. && j_fks < INCOMING_COUNT)) {
        throw std::invalid_argument(
            "FKSRadiation: emitter_mass must not be negative, and incoming emitters "
            "are massless"
        );
    }
}

std::vector<std::string> FKSRadiation::counter_events() const {
    return {"soft", "collinear", "soft_collinear"};
}

std::tuple<Value, Value, Value> FKSRadiation::build_sample(
    FunctionBuilder& fb, Value r_xi, Value r_y, Value sqrt_s, Value p_jbar
) const {
    if (_emitter_mass > 0.) {
        auto [xi_hat, y, det] =
            fb.fks_final_massive_sample(r_xi, r_y, sqrt_s, p_jbar, _emitter_mass);
        return {xi_hat, y, det};
    }
    // y_hat for an incoming emitter, y otherwise
    auto [xi_hat, y, det] = fb.fks_xi_y_sample(r_xi, r_y);
    return {xi_hat, y, det};
}

FKSRadiation::Emission FKSRadiation::build_emission(
    FunctionBuilder& fb,
    Value xi_hat,
    Value y_var,
    Value r_phi,
    Value born,
    Value sqrt_s,
    Value x1,
    Value x2
) const {
    auto p_born = fb.unstack(born);
    if (initial_state()) {
        double beam_sign = _j_fks == 0 ? 1. : -1.;
        bool asymmetric = _initial_map == InitialMap::asymmetric;
        auto
            [p_in1,
             p_in2,
             p_i,
             x1_real,
             x2_real,
             xi,
             y,
             xi_min,
             xi_max,
             det_xi,
             p_i_hat] = asymmetric
            ? fb.fks_initial_asymmetric_radiate(
                  xi_hat, y_var, r_phi, sqrt_s, x1, x2, _s_hat_min, beam_sign
              )
            : fb.fks_initial_radiate(
                  xi_hat, y_var, r_phi, sqrt_s, x1, x2, _s_hat_min, beam_sign
              );
        ValueVec final_state(p_born.begin() + INCOMING_COUNT, p_born.end());
        auto boosted = fb.unstack(
            asymmetric
                ? fb.fks_initial_light_cone_map(
                      fb.stack(final_state),
                      Value(_final_masses),
                      xi,
                      y,
                      r_phi,
                      beam_sign
                  )
                : fb.fks_initial_recoil_boost(
                      fb.stack(final_state), xi, y, r_phi, beam_sign
                  )
        );
        ValueVec real{p_in1, p_in2};
        for (std::size_t r = INCOMING_COUNT, k = 0; r < _born_particle_count + 1; ++r) {
            real.push_back(r == _i_fks ? p_i : boosted.at(k++));
        }
        return {
            fb.stack(real), x1_real, x2_real, xi, y, xi_max, det_xi, Value(1.), p_i_hat
        };
    }

    auto jb = born_index(_j_fks);
    auto p_jbar = p_born.at(jb);
    Value p_i, p_j, xi, xi_max, det_xi, rat(1.), p_i_hat;
    if (_emitter_mass > 0.) {
        auto [p_i_m, p_j_m, xi_m, xi_max_m, xi_norm, det_xi_m, p_i_hat_m] =
            fb.fks_final_massive_radiate(
                xi_hat, y_var, r_phi, sqrt_s, p_jbar, _emitter_mass
            );
        p_i = p_i_m;
        p_j = p_j_m;
        xi = xi_m;
        xi_max = xi_max_m;
        det_xi = det_xi_m;
        rat = fb.div(xi_max_m, xi_norm);
        p_i_hat = p_i_hat_m;
    } else {
        auto [p_i_m, p_j_m, xi_m, xi_max_m, det_xi_m, p_i_hat_m] =
            fb.fks_final_massless_radiate(xi_hat, y_var, r_phi, sqrt_s, p_jbar);
        p_i = p_i_m;
        p_j = p_j_m;
        xi = xi_m;
        xi_max = xi_max_m;
        det_xi = det_xi_m;
        p_i_hat = p_i_hat_m;
    }

    ValueVec spectators;
    for (std::size_t b = INCOMING_COUNT; b < _born_particle_count; ++b) {
        if (b != jb) {
            spectators.push_back(p_born.at(b));
        }
    }
    if (!spectators.empty()) {
        auto boosted = fb.unstack(
            fb.fks_recoil_boost(fb.stack(spectators), fb.add(p_i, p_j), p_jbar, sqrt_s)
        );
        for (std::size_t b = INCOMING_COUNT, k = 0; b < _born_particle_count; ++b) {
            if (b != jb) {
                p_born.at(b) = boosted.at(k++);
            }
        }
    }
    p_born.at(jb) = p_j;
    ValueVec real;
    for (std::size_t r = 0; r < _born_particle_count + 1; ++r) {
        real.push_back(r == _i_fks ? p_i : p_born.at(born_index(r)));
    }
    return {fb.stack(real), x1, x2, xi, y_var, xi_max, det_xi, rat, p_i_hat};
}

Mapping::Result FKSRadiation::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto events = build_counter_events_impl(fb, inputs, false);
    NamedVector<Value> outputs{{"momenta", events.at("momenta")}};
    if (!_leptonic) {
        outputs.push_back("x1", events.at("x1"));
        outputs.push_back("x2", events.at("x2"));
    }
    return {outputs, events.at("weight")};
}

NamedVector<Value> FKSRadiation::build_counter_events(
    FunctionBuilder& fb, const NamedVector<Value>& inputs
) const {
    return build_counter_events_impl(fb, inputs, true);
}

NamedVector<Value> FKSRadiation::build_counter_events_impl(
    FunctionBuilder& fb, const NamedVector<Value>& inputs, bool with_counter_events
) const {
    auto born = inputs.at("born_momenta");
    auto sqrt_s = fb.obs_sqrt_s(born);
    Value x1, x2;
    if (!_leptonic) {
        x1 = inputs.at("x1");
        x2 = inputs.at("x2");
    }
    // with a resonance the emission maps its members in their rest frame, at
    // the mass of the resonance
    bool resonant = !_resonance.empty();
    Value frame = born, sqrt_s_frame = sqrt_s, k_res;
    if (resonant) {
        auto [p_frame, k, sqrt_s_k] =
            fb.fks_resonance_frame(born, Value(_born_resonance_mask), sqrt_s);
        frame = p_frame;
        k_res = k;
        sqrt_s_frame = sqrt_s_k;
    }
    auto p_born = fb.unstack(born);
    auto p_jbar = initial_state() ? Value() : fb.unstack(frame).at(born_index(_j_fks));
    auto r_phi = inputs.at("r_phi");
    auto [xi_hat, y_var, det_sample] =
        build_sample(fb, inputs.at("r_xi"), inputs.at("r_y"), sqrt_s_frame, p_jbar);
    auto emission = [&](Value xi_hat_k, Value y_k) {
        auto result =
            build_emission(fb, xi_hat_k, y_k, r_phi, frame, sqrt_s_frame, x1, x2);
        if (!resonant) {
            return result;
        }
        // the members and i per unit xi back into the frame of the Born, the
        // others as in the Born (the rows of the members are not read)
        auto rows = fb.unstack(result.momenta);
        rows.push_back(result.p_i_hat);
        auto filler = p_born.at(born_index(_j_fks));
        ValueVec others;
        for (std::size_t r = 0; r < _born_particle_count + 1; ++r) {
            others.push_back(r == _i_fks ? filler : p_born.at(born_index(r)));
        }
        others.push_back(filler);
        auto mask = _real_resonance_mask;
        mask.push_back(1.);
        auto back = fb.unstack(fb.fks_resonance_frame_inverse(
            fb.stack(rows), fb.stack(others), Value(mask), k_res
        ));
        result.p_i_hat = back.back();
        back.pop_back();
        result.momenta = fb.stack(back);
        return result;
    };
    auto event = emission(xi_hat, y_var);

    NamedVector<Value> outputs{{"momenta", event.momenta}};
    if (!_leptonic) {
        outputs.push_back("x1", event.x1);
        outputs.push_back("x2", event.x2);
    }
    outputs.push_back("weight", fb.mul(fb.mul(event.xi, event.det_xi), det_sample));
    if (!with_counter_events) {
        return outputs;
    }
    outputs.push_back("xi", event.xi);
    outputs.push_back("xi_hat", xi_hat);
    outputs.push_back("y", event.y);
    outputs.push_back("xi_max", event.xi_max);
    outputs.push_back("p_i_hat", event.p_i_hat);
    for (const auto& name : counter_events()) {
        bool soft = name != "collinear", collinear = name != "soft";
        // a massive emitter has no collinear counter-events: they keep the
        // layout with the event and det = 0
        bool exists = !(collinear && _emitter_mass > 0.);
        auto cnt = exists
            ? emission(soft ? Value(0.) : xi_hat, collinear ? Value(1.) : y_var)
            : event;
        outputs.push_back(name + "_momenta", cnt.momenta);
        if (!_leptonic) {
            outputs.push_back(name + "_x1", cnt.x1);
            outputs.push_back(name + "_x2", cnt.x2);
        }
        outputs.push_back(
            name + "_det",
            fb.fks_counter_event_det(
                fb.mul(cnt.det_xi, det_sample),
                cnt.xi,
                cnt.y,
                xi_hat,
                event.rat,
                soft ? 1. : 0.,
                collinear ? 1. : 0.,
                exists ? 1. : 0.
            )
        );
        outputs.push_back(name + "_p_i_hat", cnt.p_i_hat);
        outputs.push_back(name + "_xi", cnt.xi);
        outputs.push_back(name + "_xi_max", cnt.xi_max);
    }
    return outputs;
}

Mapping::Result FKSRadiation::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    return initial_state()
        ? build_initial_inverse(fb, inputs)
        : build_final_inverse(fb, inputs);
}

Mapping::Result FKSRadiation::build_final_inverse(
    FunctionBuilder& fb, const NamedVector<Value>& inputs
) const {
    auto momenta = inputs.at(0);
    auto sqrt_s = fb.obs_sqrt_s(momenta);
    // with a resonance the members are clustered in their rest frame
    bool resonant = !_resonance.empty();
    Value frame = momenta, k_res;
    if (resonant) {
        auto [p_frame, k, sqrt_s_k] =
            fb.fks_resonance_frame(momenta, Value(_real_resonance_mask), sqrt_s);
        frame = p_frame;
        k_res = k;
        sqrt_s = sqrt_s_k;
    }
    auto p_real = fb.unstack(frame);
    auto p_i = p_real.at(_i_fks), p_j = p_real.at(_j_fks);

    Value p_jbar, r_xi, r_y, r_phi, xi, det_xi_inv, det_sample_inv;
    if (_emitter_mass > 0.) {
        auto [p_jbar_m, xi_hat, y, r_phi_m, xi_m, xi_max, xi_norm, det_m] =
            fb.fks_final_massive_cluster(p_i, p_j, sqrt_s, _emitter_mass);
        auto [r_xi_m, r_y_m, det] = fb.fks_final_massive_sample_inverse(
            xi_hat, y, sqrt_s, p_jbar_m, _emitter_mass
        );
        p_jbar = p_jbar_m;
        r_phi = r_phi_m;
        xi = xi_m;
        det_xi_inv = det_m;
        r_xi = r_xi_m;
        r_y = r_y_m;
        det_sample_inv = det;
    } else {
        auto [p_jbar_m, xi_hat, y, r_phi_m, xi_m, xi_max, det_m] =
            fb.fks_final_massless_cluster(p_i, p_j, sqrt_s);
        auto [r_xi_m, r_y_m, det] = fb.fks_xi_y_sample_inverse(xi_hat, y);
        p_jbar = p_jbar_m;
        r_phi = r_phi_m;
        xi = xi_m;
        det_xi_inv = det_m;
        r_xi = r_xi_m;
        r_y = r_y_m;
        det_sample_inv = det;
    }

    ValueVec spectators;
    for (std::size_t r = INCOMING_COUNT; r < _born_particle_count + 1; ++r) {
        if (r != _i_fks && r != _j_fks) {
            spectators.push_back(p_real.at(r));
        }
    }
    ValueVec boosted;
    if (!spectators.empty()) {
        boosted = fb.unstack(fb.fks_recoil_boost_inverse(
            fb.stack(spectators), fb.add(p_i, p_j), p_jbar, sqrt_s
        ));
    }

    ValueVec born;
    for (std::size_t r = 0, k = 0; r < _born_particle_count + 1; ++r) {
        if (r == _i_fks) {
            continue;
        }
        if (r == _j_fks) {
            born.push_back(p_jbar);
        } else if (r < INCOMING_COUNT) {
            born.push_back(p_real.at(r));
        } else {
            born.push_back(boosted.at(k++));
        }
    }
    auto born_momenta = fb.stack(born);
    if (resonant) {
        // the members back into the frame of the real emission, the others as
        // there (the row of jbar is not read)
        auto p_orig = fb.unstack(momenta);
        ValueVec others;
        for (std::size_t r = 0; r < _born_particle_count + 1; ++r) {
            if (r != _i_fks) {
                others.push_back(p_orig.at(r));
            }
        }
        born_momenta = fb.fks_resonance_frame_inverse(
            born_momenta, fb.stack(others), Value(_born_resonance_mask), k_res
        );
    }
    auto det = fb.div(fb.mul(det_xi_inv, det_sample_inv), xi);
    NamedVector<Value> outputs{
        {"r_xi", r_xi}, {"r_y", r_y}, {"r_phi", r_phi}, {"born_momenta", born_momenta}
    };
    if (!_leptonic) {
        outputs.push_back("x1", inputs.at(1));
        outputs.push_back("x2", inputs.at(2));
    }
    return {outputs, det};
}

Mapping::Result FKSRadiation::build_initial_inverse(
    FunctionBuilder& fb, const NamedVector<Value>& inputs
) const {
    double beam_sign = _j_fks == 0 ? 1. : -1.;
    bool asymmetric = _initial_map == InitialMap::asymmetric;
    auto p_real = fb.unstack(inputs.at(0));
    auto
        [sqrt_s,
         x1_born,
         x2_born,
         xi_hat,
         y_hat,
         r_phi,
         xi,
         y,
         xi_min,
         xi_max,
         det_xi_inv] = asymmetric
        ? fb.fks_initial_asymmetric_cluster(
              p_real.at(0),
              p_real.at(1),
              p_real.at(_i_fks),
              inputs.at(1),
              inputs.at(2),
              _s_hat_min,
              beam_sign
          )
        : fb.fks_initial_cluster(
              p_real.at(0),
              p_real.at(1),
              p_real.at(_i_fks),
              inputs.at(1),
              inputs.at(2),
              _s_hat_min,
              beam_sign
          );
    auto [r_xi, r_y, det_sample_inv] = fb.fks_xi_y_sample_inverse(xi_hat, y_hat);

    ValueVec final_state;
    for (std::size_t r = INCOMING_COUNT; r < _born_particle_count + 1; ++r) {
        if (r != _i_fks) {
            final_state.push_back(p_real.at(r));
        }
    }
    auto boosted = fb.unstack(
        asymmetric
            ? fb.fks_initial_light_cone_map_inverse(
                  fb.stack(final_state), Value(_final_masses), xi, y, r_phi, beam_sign
              )
            : fb.fks_initial_recoil_boost_inverse(
                  fb.stack(final_state), xi, y, r_phi, beam_sign
              )
    );
    auto [p_in1, p_in2] = fb.com_p_in(sqrt_s);
    ValueVec born{p_in1, p_in2};
    born.insert(born.end(), boosted.begin(), boosted.end());

    auto det = fb.div(fb.mul(det_xi_inv, det_sample_inv), xi);
    return {
        {{"r_xi", r_xi},
         {"r_y", r_y},
         {"r_phi", r_phi},
         {"born_momenta", fb.stack(born)},
         {"x1", x1_born},
         {"x2", x2_born}},
        det
    };
}

#include "madspace/phasespace/fks_radiation.hpp"

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
    std::size_t born_particle_count, std::size_t i_fks, std::size_t j_fks, bool leptonic
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
    _leptonic(leptonic) {}

Mapping::Result FKSRadiation::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    return initial_state()
        ? build_initial_forward(fb, inputs)
        : build_final_forward(fb, inputs);
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

Mapping::Result FKSRadiation::build_final_forward(
    FunctionBuilder& fb, const NamedVector<Value>& inputs
) const {
    auto born = inputs.at(3);
    auto sqrt_s = fb.obs_sqrt_s(born);
    auto p_born = fb.unstack(born);
    auto jb = born_index(_j_fks);

    auto [xi_hat, y, det_sample] = fb.fks_xi_y_sample(inputs.at(0), inputs.at(1));
    auto [p_i, p_j, xi, xi_max, det_xi] =
        fb.fks_final_massless_radiate(xi_hat, y, inputs.at(2), sqrt_s, p_born.at(jb));

    ValueVec spectators;
    for (std::size_t b = INCOMING_COUNT; b < _born_particle_count; ++b) {
        if (b != jb) {
            spectators.push_back(p_born.at(b));
        }
    }
    if (!spectators.empty()) {
        auto boosted = fb.unstack(
            fb.fks_recoil_boost(fb.stack(spectators), fb.add(p_i, p_j), sqrt_s)
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
    auto det = fb.mul(fb.mul(xi, det_xi), det_sample);
    NamedVector<Value> outputs{{"momenta", fb.stack(real)}};
    if (!_leptonic) {
        outputs.push_back("x1", inputs.at(4));
        outputs.push_back("x2", inputs.at(5));
    }
    return {outputs, det};
}

Mapping::Result FKSRadiation::build_final_inverse(
    FunctionBuilder& fb, const NamedVector<Value>& inputs
) const {
    auto momenta = inputs.at(0);
    auto sqrt_s = fb.obs_sqrt_s(momenta);
    auto p_real = fb.unstack(momenta);
    auto p_i = p_real.at(_i_fks), p_j = p_real.at(_j_fks);

    auto [p_jbar, xi_hat, y, r_phi, xi, xi_max, det_xi_inv] =
        fb.fks_final_massless_cluster(p_i, p_j, sqrt_s);
    auto [r_xi, r_y, det_sample_inv] = fb.fks_xi_y_sample_inverse(xi_hat, y);

    ValueVec spectators;
    for (std::size_t r = INCOMING_COUNT; r < _born_particle_count + 1; ++r) {
        if (r != _i_fks && r != _j_fks) {
            spectators.push_back(p_real.at(r));
        }
    }
    ValueVec boosted;
    if (!spectators.empty()) {
        boosted = fb.unstack(
            fb.fks_recoil_boost_inverse(fb.stack(spectators), fb.add(p_i, p_j), sqrt_s)
        );
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
    auto det = fb.div(fb.mul(det_xi_inv, det_sample_inv), xi);
    NamedVector<Value> outputs{
        {"r_xi", r_xi}, {"r_y", r_y}, {"r_phi", r_phi}, {"born_momenta", fb.stack(born)}
    };
    if (!_leptonic) {
        outputs.push_back("x1", inputs.at(1));
        outputs.push_back("x2", inputs.at(2));
    }
    return {outputs, det};
}

Mapping::Result FKSRadiation::build_initial_forward(
    FunctionBuilder& fb, const NamedVector<Value>& inputs
) const {
    double beam_sign = _j_fks == 0 ? 1. : -1.;
    auto born = inputs.at(3);
    auto sqrt_s = fb.obs_sqrt_s(born);
    auto p_born = fb.unstack(born);

    auto [xi_hat, y, det_sample] = fb.fks_xi_y_sample(inputs.at(0), inputs.at(1));
    auto r_phi = inputs.at(2);
    auto [p_in1, p_in2, p_i, x1, x2, xi, xi_max, det_xi] = fb.fks_initial_radiate(
        xi_hat, y, r_phi, sqrt_s, inputs.at(4), inputs.at(5), beam_sign
    );

    ValueVec final_state(p_born.begin() + INCOMING_COUNT, p_born.end());
    auto boosted = fb.unstack(
        fb.fks_initial_recoil_boost(fb.stack(final_state), xi, y, r_phi, beam_sign)
    );

    ValueVec real{p_in1, p_in2};
    for (std::size_t r = INCOMING_COUNT, k = 0; r < _born_particle_count + 1; ++r) {
        real.push_back(r == _i_fks ? p_i : boosted.at(k++));
    }
    auto det = fb.mul(fb.mul(xi, det_xi), det_sample);
    return {{{"momenta", fb.stack(real)}, {"x1", x1}, {"x2", x2}}, det};
}

Mapping::Result FKSRadiation::build_initial_inverse(
    FunctionBuilder& fb, const NamedVector<Value>& inputs
) const {
    double beam_sign = _j_fks == 0 ? 1. : -1.;
    auto p_real = fb.unstack(inputs.at(0));
    auto [sqrt_s, x1_born, x2_born, xi_hat, y, r_phi, xi, xi_max, det_xi_inv] =
        fb.fks_initial_cluster(
            p_real.at(0),
            p_real.at(1),
            p_real.at(_i_fks),
            inputs.at(1),
            inputs.at(2),
            beam_sign
        );
    auto [r_xi, r_y, det_sample_inv] = fb.fks_xi_y_sample_inverse(xi_hat, y);

    ValueVec final_state;
    for (std::size_t r = INCOMING_COUNT; r < _born_particle_count + 1; ++r) {
        if (r != _i_fks) {
            final_state.push_back(p_real.at(r));
        }
    }
    auto boosted = fb.unstack(fb.fks_initial_recoil_boost_inverse(
        fb.stack(final_state), xi, y, r_phi, beam_sign
    ));
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

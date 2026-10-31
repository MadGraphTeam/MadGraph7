#include "madspace/phasespace/fks_radiation.hpp"

#include <stdexcept>
#include <string>

using namespace madspace;

namespace {

constexpr std::size_t INCOMING_COUNT = 2;

std::size_t checked_born_count(
    std::size_t born_particle_count, std::size_t i_fks, std::size_t j_fks
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
    if (i_fks < INCOMING_COUNT || j_fks < INCOMING_COUNT || i_fks == j_fks) {
        throw std::invalid_argument(
            "FKSRadiation: i_fks and j_fks must be two different final-state "
            "particles (initial-state emitters are not supported yet)"
        );
    }
    return born_particle_count;
}

} // namespace

FKSRadiation::FKSRadiation(
    std::size_t born_particle_count, std::size_t i_fks, std::size_t j_fks
) :
    Mapping(
        "FKSRadiation",
        {{"r_xi", batch_float},
         {"r_y", batch_float},
         {"r_phi", batch_float},
         {"born_momenta",
          batch_four_vec_array(checked_born_count(born_particle_count, i_fks, j_fks))}},
        {{"momenta", batch_four_vec_array(born_particle_count + 1)}},
        {}
    ),
    _born_particle_count(born_particle_count),
    _i_fks(i_fks),
    _j_fks(j_fks) {}

Mapping::Result FKSRadiation::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
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
    return {{{"momenta", fb.stack(real)}}, det};
}

Mapping::Result FKSRadiation::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
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
    return {
        {{"r_xi", r_xi},
         {"r_y", r_y},
         {"r_phi", r_phi},
         {"born_momenta", fb.stack(born)}},
        det
    };
}

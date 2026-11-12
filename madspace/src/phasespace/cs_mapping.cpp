#include "madspace/phasespace/cs_mapping.hpp"

#include <algorithm>
#include <stdexcept>

#include "madspace/phasespace/fks_mapping.hpp"

using namespace madspace;

namespace {

std::size_t incoming_count(const PhaseSpaceMapping& born) {
    return born.is_decay() ? 1 : 2;
}

// whether the momentum fractions of a Born mapping are one, without densities
// or for a decay
bool fixed_fractions(const PhaseSpaceMapping& born) {
    return born.leptonic() || born.is_decay();
}

CSMapping::Family family_of(
    const PhaseSpaceMapping& born,
    std::size_t emitted,
    std::size_t emitter,
    std::size_t spectator
) {
    std::size_t incoming = incoming_count(born);
    std::size_t count = born.particle_count() + 1;
    if (emitted < incoming || emitted >= count) {
        throw std::invalid_argument(
            "CSMapping: the emitted parton has to be in the final state of the "
            "real process"
        );
    }
    if (emitter >= count || spectator >= count || emitter == emitted ||
        spectator == emitted || emitter == spectator) {
        throw std::invalid_argument(
            "CSMapping: the emitted parton, the emitter and the spectator need "
            "three positions in the real process"
        );
    }
    bool initial_emitter = emitter < incoming, initial_spectator = spectator < incoming;
    if ((initial_emitter || initial_spectator) && fixed_fractions(born)) {
        throw std::invalid_argument(
            "CSMapping: an incoming emitter or spectator needs the momentum "
            "fractions of hadron beams"
        );
    }
    if (born.sqrt_s_hat_condition()) {
        throw std::invalid_argument(
            "CSMapping: a Born mapping with the sqrt_s_hat condition belongs to "
            "lepton beams"
        );
    }
    // with emitter and spectator in the final state the emission keeps s_hat
    // of the Born, so a floor on it bounds the real emission too
    bool floor_only =
        !initial_emitter && !initial_spectator && born.cuts().only_sqrt_s_min();
    if (!born.cuts().empty() && !floor_only) {
        throw std::invalid_argument(
            "CSMapping: the Born mapping must not have cuts, they belong to the "
            "real emission and the subtraction terms (except a floor on "
            "sqrt(s_hat) for a final-state emitter and spectator)"
        );
    }
    if (initial_emitter) {
        return initial_spectator
            ? CSMapping::initial_initial
            : CSMapping::initial_final;
    }
    return initial_spectator ? CSMapping::final_initial : CSMapping::final_final;
}

std::vector<double> real_masses(const PhaseSpaceMapping& born, std::size_t emitted) {
    auto masses = born.external_masses();
    masses.insert(masses.begin() + emitted, 0.);
    return masses;
}

NamedVector<Type> cs_input_types(const PhaseSpaceMapping& born) {
    const auto& born_types = born.input_types();
    auto names = born_types.keys();
    NamedVector<Type> types;
    for (std::size_t k = 0; k < born_types.size(); ++k) {
        const auto& name = names.at(k);
        types.push_back(
            name,
            name == "random"
                ? batch_float_array(born.random_dim() + 3)
                : born_types.at(k)
        );
    }
    return types;
}

NamedVector<Type> cs_output_types(const PhaseSpaceMapping& born) {
    return {
        {"momenta", batch_four_vec_array(born.particle_count() + 1)},
        {"x1", batch_float},
        {"x2", batch_float},
    };
}

} // namespace

CSMapping::CSMapping(
    const PhaseSpaceMapping& born,
    std::size_t emitted,
    std::size_t emitter,
    std::size_t spectator
) :
    Mapping(
        "CSMapping", cs_input_types(born), cs_output_types(born), born.condition_types()
    ),
    _born(born),
    _emitted(emitted),
    _emitter(emitter),
    _spectator(spectator),
    _family(family_of(born, emitted, emitter, spectator)),
    _incoming(incoming_count(born)),
    _born_masses(born.external_masses()),
    _real_masses(real_masses(born, emitted)) {
    // the masses of emitter and spectator, the same in every permutation of
    // the Born mapping, and massless incoming partons
    for (auto real : {emitter, spectator}) {
        double mass = _born_masses.at(born_position(real));
        if (mass < 0. || (real < _incoming && mass != 0.)) {
            throw std::invalid_argument(
                "CSMapping: the emitter and the spectator need the same mass in "
                "every permutation of the Born mapping, zero if incoming"
            );
        }
    }
}

Mapping::Result CSMapping::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    bool boosted = !fixed_fractions(_born) || _born.beam_rapidity() != 0.;
    auto r = fb.unstack(inputs.at("random"));
    std::size_t n_born = _born.random_dim();
    NamedVector<Value> born_inputs{
        {"random", fb.stack(ValueVec(r.begin(), r.begin() + n_born))}
    };
    if (discrete_dim() > 0) {
        born_inputs.push_back("discrete", inputs.at("discrete"));
    }
    auto born = _born.build_forward(fb, born_inputs, conditions);
    ValueVec x{born.at("x1"), born.at("x2")};

    // the emission in the center-of-mass frame of the Born
    auto [plus, minus] = fks_born_frame(fb, _born, x.at(0), x.at(1), conditions);
    auto born_cm = born.at("momenta");
    if (boosted) {
        born_cm = fb.fks_beam_boost(born_cm, Value(_born_masses), minus, plus);
    }
    auto p = fb.unstack(born_cm);
    auto r_1 = r.at(n_born), r_2 = r.at(n_born + 1), r_phi = r.at(n_born + 2);
    // the Born momenta at their positions in the real emission, then i
    ValueVec real(p);
    real.insert(real.begin() + _emitted, Value());
    Value det;
    if (_family == final_final) {
        auto j = born_position(_emitter), k = born_position(_spectator);
        auto [p_i, p_j, p_k, y, z, det_cs] = fb.cs_final_final_radiate(
            r_1, r_2, r_phi, p.at(j), p.at(k), _born_masses.at(j), _born_masses.at(k)
        );
        real.at(_emitted) = p_i;
        real.at(_emitter) = p_j;
        real.at(_spectator) = p_k;
        det = det_cs;
    } else if (_family == final_initial || _family == initial_final) {
        bool initial_emitter = _family == initial_final;
        auto a = initial_emitter ? _emitter : _spectator;
        auto f_real = initial_emitter ? _spectator : _emitter;
        auto f = born_position(f_real);
        auto [p_a, p_i, p_f, x_cs, z, det_cs] = fb.cs_initial_final_radiate(
            r_1, r_2, r_phi, p.at(a), p.at(f), x.at(a), _born_masses.at(f)
        );
        real.at(a) = p_a;
        real.at(_emitted) = p_i;
        real.at(f_real) = p_f;
        x.at(a) = fb.div(x.at(a), x_cs);
        det = det_cs;
    } else {
        auto a = _emitter, b = _spectator;
        auto finals = fb.stack(ValueVec(p.begin() + _incoming, p.end()));
        auto [p_a, p_i, p_out, x_cs, v, det_cs] = fb.cs_initial_initial_radiate(
            r_1, r_2, r_phi, p.at(a), p.at(b), finals, x.at(a)
        );
        real = fb.unstack(p_out);
        real.insert(real.begin(), {p.at(0), p.at(1)});
        real.at(a) = p_a;
        real.insert(real.begin() + _emitted, p_i);
        x.at(a) = fb.div(x.at(a), x_cs);
        det = det_cs;
    }
    auto momenta = fb.stack(real);
    if (boosted) {
        momenta = fb.fks_beam_boost(momenta, Value(_real_masses), plus, minus);
    }
    return {
        {{"momenta", momenta}, {"x1", x.at(0)}, {"x2", x.at(1)}},
        fb.mul(born.at("det"), det)
    };
}

NamedVector<Value> CSMapping::build_projection(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    bool boosted = !fixed_fractions(_born) || _born.beam_rapidity() != 0.;
    ValueVec x{inputs.at("x1"), inputs.at("x2")};

    // the projection in the center-of-mass frame of the real emission
    auto [plus, minus] = fks_born_frame(fb, _born, x.at(0), x.at(1), conditions);
    auto real_cm = inputs.at("momenta");
    if (boosted) {
        real_cm = fb.fks_beam_boost(real_cm, Value(_real_masses), minus, plus);
    }
    auto p = fb.unstack(real_cm);
    ValueVec born(p);
    born.erase(born.begin() + _emitted);
    NamedVector<Value> outputs;
    auto& p_i = p.at(_emitted);
    if (_family == final_final) {
        auto j = born_position(_emitter), k = born_position(_spectator);
        auto [p_ij, p_k, r_1, r_2, r_phi, y, z, det] = fb.cs_final_final_project(
            p_i,
            p.at(_emitter),
            p.at(_spectator),
            _born_masses.at(j),
            _born_masses.at(k)
        );
        born.at(j) = p_ij;
        born.at(k) = p_k;
        outputs = {
            {"r_1", r_1},
            {"r_2", r_2},
            {"r_phi", r_phi},
            {"y", y},
            {"z", z},
            {"det", det}
        };
    } else if (_family == final_initial || _family == initial_final) {
        bool initial_emitter = _family == initial_final;
        auto a = initial_emitter ? _emitter : _spectator;
        auto f_real = initial_emitter ? _spectator : _emitter;
        auto f = born_position(f_real);
        auto [p_a, p_f, r_1, r_2, r_phi, x_cs, z, det] = fb.cs_initial_final_project(
            p.at(a), p_i, p.at(f_real), x.at(a), _born_masses.at(f)
        );
        born.at(a) = p_a;
        born.at(f) = p_f;
        x.at(a) = fb.mul(x.at(a), x_cs);
        outputs = {
            {"r_1", r_1},
            {"r_2", r_2},
            {"r_phi", r_phi},
            {"x", x_cs},
            {"z", z},
            {"det", det}
        };
    } else {
        auto a = _emitter, b = _spectator;
        ValueVec finals(born.begin() + _incoming, born.end());
        auto [p_a, p_out, r_1, r_2, r_phi, x_cs, v, det] =
            fb.cs_initial_initial_project(
                p.at(a), p.at(b), p_i, fb.stack(finals), x.at(a)
            );
        auto finals_born = fb.unstack(p_out);
        std::copy(finals_born.begin(), finals_born.end(), born.begin() + _incoming);
        born.at(a) = p_a;
        x.at(a) = fb.mul(x.at(a), x_cs);
        outputs = {
            {"r_1", r_1},
            {"r_2", r_2},
            {"r_phi", r_phi},
            {"x", x_cs},
            {"v", v},
            {"det", det}
        };
    }
    auto born_lab = fb.stack(born);
    if (boosted) {
        born_lab = fb.fks_beam_boost(born_lab, Value(_born_masses), plus, minus);
    }
    outputs.push_back("born_momenta", born_lab);
    outputs.push_back("born_x1", x.at(0));
    outputs.push_back("born_x2", x.at(1));
    return outputs;
}

Mapping::Result CSMapping::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto projection = build_projection(fb, inputs, conditions);
    auto born = _born.build_inverse(
        fb,
        {{"momenta", projection.at("born_momenta")},
         {"x1", projection.at("born_x1")},
         {"x2", projection.at("born_x2")}},
        conditions
    );
    auto r = fb.unstack(born.at("random"));
    r.push_back(projection.at("r_1"));
    r.push_back(projection.at("r_2"));
    r.push_back(projection.at("r_phi"));
    NamedVector<Value> outputs{{"random", fb.stack(r)}};
    if (discrete_dim() > 0) {
        outputs.push_back("discrete", born.at("discrete"));
    }
    return {outputs, fb.mul(born.at("det"), projection.at("det"))};
}

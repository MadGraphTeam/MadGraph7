#include "madspace/phasespace/cs_mapping.hpp"

#include <algorithm>
#include <array>
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
    if (born.is_decay() && initial_emitter) {
        throw std::invalid_argument(
            "CSMapping: the decaying particle cannot be the emitter"
        );
    }
    bool decay_spectator = born.is_decay() && initial_spectator;
    if ((initial_emitter || initial_spectator) && fixed_fractions(born) &&
        !decay_spectator) {
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
    if (decay_spectator) {
        if (born.particle_count() < 3) {
            throw std::invalid_argument(
                "CSMapping: a decaying spectator needs a recoil system, a second "
                "final-state particle of the Born"
            );
        }
        return CSMapping::final_decay;
    }
    if (initial_emitter) {
        return initial_spectator
            ? CSMapping::initial_initial
            : CSMapping::initial_final;
    }
    return initial_spectator ? CSMapping::final_initial : CSMapping::final_final;
}

std::vector<double> masses_of_real(const PhaseSpaceMapping& born, std::size_t emitted) {
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
    std::size_t spectator,
    double cutoff
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
    _real_masses(masses_of_real(born, emitted)),
    _cutoff(cutoff) {
    if (!(cutoff >= 0. && cutoff < 1.)) {
        throw std::invalid_argument("CSMapping: the cutoff has to be in [0, 1)");
    }
    // the masses of emitter and spectator, the same in every permutation of
    // the Born mapping, and massless incoming partons (but a decaying one)
    for (auto real : {emitter, spectator}) {
        double mass = _born_masses.at(born_position(real));
        bool massless = real < _incoming && _family != final_decay;
        if (mass < 0. || (massless && mass != 0.)) {
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
            r_1,
            r_2,
            r_phi,
            p.at(j),
            p.at(k),
            _born_masses.at(j),
            _born_masses.at(k),
            _cutoff
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
            r_1, r_2, r_phi, p.at(a), p.at(f), x.at(a), _born_masses.at(f), _cutoff
        );
        real.at(a) = p_a;
        real.at(_emitted) = p_i;
        real.at(f_real) = p_f;
        x.at(a) = fb.div(x.at(a), x_cs);
        det = det_cs;
    } else if (_family == final_decay) {
        // the other final-state particles of the Born, in their order
        auto j = born_position(_emitter);
        ValueVec others;
        for (std::size_t n = _incoming; n < p.size(); ++n) {
            if (n != j) {
                others.push_back(p.at(n));
            }
        }
        auto [p_i, p_j, p_out, y, z, det_cs] = fb.cs_final_decay_radiate(
            r_1, r_2, r_phi, p.at(j), fb.stack(others), _born_masses.at(j), _cutoff
        );
        auto recoil = fb.unstack(p_out);
        ValueVec born_real(p);
        for (std::size_t n = _incoming, o = 0; n < p.size(); ++n) {
            if (n != j) {
                born_real.at(n) = recoil.at(o++);
            }
        }
        born_real.at(j) = p_j;
        real = born_real;
        real.insert(real.begin() + _emitted, p_i);
        det = det_cs;
    } else {
        auto a = _emitter, b = _spectator;
        auto finals = fb.stack(ValueVec(p.begin() + _incoming, p.end()));
        auto [p_a, p_i, p_out, x_cs, v, det_cs] = fb.cs_initial_initial_radiate(
            r_1, r_2, r_phi, p.at(a), p.at(b), finals, x.at(a), _cutoff
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
            _born_masses.at(k),
            _cutoff
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
            p.at(a), p_i, p.at(f_real), x.at(a), _born_masses.at(f), _cutoff
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
    } else if (_family == final_decay) {
        auto j = born_position(_emitter);
        ValueVec others;
        for (std::size_t n = _incoming; n < born.size(); ++n) {
            if (n != j) {
                others.push_back(born.at(n));
            }
        }
        auto [p_ij, p_out, r_1, r_2, r_phi, y, z, det] = fb.cs_final_decay_project(
            p_i, p.at(_emitter), fb.stack(others), _born_masses.at(j), _cutoff
        );
        auto recoil = fb.unstack(p_out);
        for (std::size_t n = _incoming, o = 0; n < born.size(); ++n) {
            if (n != j) {
                born.at(n) = recoil.at(o++);
            }
        }
        born.at(j) = p_ij;
        outputs = {
            {"r_1", r_1},
            {"r_2", r_2},
            {"r_phi", r_phi},
            {"y", y},
            {"z", z},
            {"det", det}
        };
    } else {
        auto a = _emitter, b = _spectator;
        ValueVec finals(born.begin() + _incoming, born.end());
        auto [p_a, p_out, r_1, r_2, r_phi, x_cs, v, det] =
            fb.cs_initial_initial_project(
                p.at(a), p.at(b), p_i, fb.stack(finals), x.at(a), _cutoff
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

namespace {

const std::vector<CSMapping>& checked_dipoles(
    const std::vector<CSMapping>& mappings,
    const std::vector<std::string>& kernels,
    const std::vector<double>& symmetry
) {
    if (mappings.empty() || kernels.size() != mappings.size() ||
        symmetry.size() != mappings.size()) {
        throw std::invalid_argument(
            "CSDipoles: one kernel and one symmetry factor for each of at least one "
            "dipole"
        );
    }
    const auto& first = mappings.front();
    for (const auto& mapping : mappings) {
        if (mapping.particle_count() != first.particle_count() ||
            mapping.condition_types().keys() != first.condition_types().keys()) {
            throw std::invalid_argument(
                "CSDipoles: the dipoles need the same real process and conditions"
            );
        }
    }
    return mappings;
}

double kernel_index(CSMapping::Family family, const std::string& kernel) {
    bool final_emitter = family == CSMapping::final_final ||
        family == CSMapping::final_initial || family == CSMapping::final_decay;
    std::vector<std::string> names{"qg", "gg", "gq"};
    if (!final_emitter) {
        names = {"qg", "gg", "qq", "gq"};
    }
    if (family == CSMapping::final_decay && kernel != "qg") {
        throw std::invalid_argument(
            "CSDipoles: a decaying spectator has the dipoles of a quark emitter "
            "only (qg)"
        );
    }
    for (std::size_t k = 0; k < names.size(); ++k) {
        if (names[k] == kernel) {
            return static_cast<double>(k);
        }
    }
    throw std::invalid_argument("CSDipoles: unknown kernel " + kernel);
}

NamedVector<Type> dipole_return_types(const std::vector<CSMapping>& mappings) {
    if (mappings.empty()) {
        throw std::invalid_argument("CSDipoles: no dipoles");
    }
    int d = static_cast<int>(mappings.size());
    int n = static_cast<int>(mappings.front().particle_count()) - 1;
    return {
        {"born_momenta", Type(DataType::dt_float64, batch_size, {d, n, 4})},
        {"born_x1", batch_float_array(d)},
        {"born_x2", batch_float_array(d)},
        {"color_weight", batch_float_array(d)},
        {"spin_weight", batch_float_array(d)},
        {"spin_vector", batch_four_vec_array(d)},
    };
}

NamedVector<Type> dipole_arg_types(const std::vector<CSMapping>& mappings) {
    auto types = mappings.front().output_types();
    const auto& conditions = mappings.front().condition_types();
    auto names = conditions.keys();
    for (std::size_t k = 0; k < conditions.size(); ++k) {
        types.push_back(names.at(k), conditions.at(k));
    }
    return types;
}

} // namespace

CSDipoles::CSDipoles(
    const std::vector<CSMapping>& mappings,
    const std::vector<std::string>& kernels,
    const std::vector<double>& symmetry,
    double alpha_ff,
    double alpha_fi,
    double alpha_if,
    double alpha_ii,
    double kappa
) :
    FunctionGenerator(
        "CSDipoles",
        dipole_arg_types(checked_dipoles(mappings, kernels, symmetry)),
        dipole_return_types(mappings)
    ),
    _mappings(mappings),
    _symmetry(symmetry),
    _alpha{alpha_ff, alpha_fi, alpha_if, alpha_ii},
    _kappa(kappa) {
    for (double alpha : _alpha) {
        if (!(alpha > 0. && alpha <= 1.)) {
            throw std::invalid_argument("CSDipoles: alpha has to be in (0, 1]");
        }
    }
    for (std::size_t d = 0; d < mappings.size(); ++d) {
        _kernels.push_back(kernel_index(mappings[d].family(), kernels[d]));
    }
}

NamedVector<Value> CSDipoles::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    const auto& first = _mappings.front();
    NamedVector<Value> inputs, conditions;
    for (const auto& name : first.output_types().keys()) {
        inputs.push_back(name, args.at(name));
    }
    for (const auto& name : first.condition_types().keys()) {
        conditions.push_back(name, args.at(name));
    }
    auto p = fb.unstack(args.at("momenta"));
    ValueVec born_momenta, x1, x2, color, spin, vectors;
    for (std::size_t d = 0; d < _mappings.size(); ++d) {
        const auto& mapping = _mappings[d];
        auto projection = mapping.build_projection(fb, inputs, conditions);
        auto born = fb.unstack(projection.at("born_momenta"));
        auto i = mapping.emitted(), j = mapping.emitter(), k = mapping.spectator();
        auto bj = j < i ? j : j - 1, bk = k < i ? k : k - 1;
        const auto& masses = mapping.real_masses();
        auto family = mapping.family();
        // the dipoles of a decaying spectator take the alpha of the final-final
        // family
        double alpha =
            _alpha[family == CSMapping::final_decay ? CSMapping::final_final : family];
        std::vector<double> options{
            _kernels[d], alpha, _kappa, masses.at(j), masses.at(k)
        };
        std::array<Value, 3> weights;
        if (family == CSMapping::final_final) {
            weights = fb.cs_final_final_dipole(
                p.at(i), p.at(j), p.at(k), born.at(bj), born.at(bk), Value(options)
            );
        } else if (family == CSMapping::final_initial) {
            weights = fb.cs_final_initial_dipole(
                p.at(i), p.at(j), p.at(k), born.at(bj), born.at(bk), Value(options)
            );
        } else if (family == CSMapping::initial_final) {
            weights = fb.cs_initial_final_dipole(
                p.at(j), p.at(i), p.at(k), born.at(bj), born.at(bk), Value(options)
            );
        } else if (family == CSMapping::final_decay) {
            weights = fb.cs_final_decay_dipole(
                p.at(i), p.at(j), p.at(k), born.at(bj), born.at(bk), Value(options)
            );
        } else {
            weights =
                fb.cs_initial_initial_dipole(p.at(j), p.at(k), p.at(i), Value(options));
        }
        born_momenta.push_back(projection.at("born_momenta"));
        x1.push_back(projection.at("born_x1"));
        x2.push_back(projection.at("born_x2"));
        color.push_back(fb.mul(weights[0], _symmetry[d]));
        spin.push_back(fb.mul(weights[1], _symmetry[d]));
        vectors.push_back(weights[2]);
    }
    return {
        {"born_momenta", fb.stack(born_momenta)},
        {"born_x1", fb.stack(x1)},
        {"born_x2", fb.stack(x2)},
        {"color_weight", fb.stack(color)},
        {"spin_weight", fb.stack(spin)},
        {"spin_vector", fb.stack(vectors)},
    };
}

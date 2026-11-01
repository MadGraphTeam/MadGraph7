#include "madspace/phasespace/fks_mapping.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace madspace;

namespace {

constexpr std::size_t INCOMING_COUNT = 2;

const PhaseSpaceMapping&
checked_born(const PhaseSpaceMapping& born, std::size_t j_fks) {
    // A final-state emitter leaves s_hat of the Born unchanged, so a floor on
    // it (from fks_sqrt_s_hat_min) bounds the real emission too. An incoming
    // emitter raises s_hat, its Born has to cover everything below.
    bool floor_only = j_fks >= INCOMING_COUNT && born.cuts().only_sqrt_s_min();
    if (!born.cuts().empty() && !floor_only) {
        throw std::invalid_argument(
            "FKSMapping: the Born mapping must not have cuts, they belong to the "
            "real emission and the counter-events (except a floor on sqrt(s_hat) "
            "for a final-state emitter)"
        );
    }
    return born;
}

std::vector<double> real_masses(const PhaseSpaceMapping& born, std::size_t i_fks) {
    auto masses = born.external_masses();
    if (i_fks > masses.size()) {
        throw std::invalid_argument("FKSMapping: i_fks is out of range");
    }
    masses.insert(masses.begin() + i_fks, 0.);
    return masses;
}

double
emitter_mass(const PhaseSpaceMapping& born, std::size_t i_fks, std::size_t j_fks) {
    if (j_fks < INCOMING_COUNT) {
        return 0.;
    }
    auto masses = real_masses(born, i_fks);
    if (j_fks >= masses.size() || masses.at(j_fks) < 0.) {
        throw std::invalid_argument(
            "FKSMapping: the emitter needs a position in the real process and the "
            "same mass in every permutation of the Born mapping"
        );
    }
    return masses.at(j_fks);
}

NamedVector<Type> fks_input_types(const PhaseSpaceMapping& born) {
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

NamedVector<Type> fks_output_types(const PhaseSpaceMapping& born) {
    return {
        {"momenta", batch_four_vec_array(born.particle_count() + 1)},
        {"x1", batch_float},
        {"x2", batch_float},
    };
}

} // namespace

double madspace::fks_sqrt_s_hat_min(
    const Cuts& cuts, const std::vector<double>& outgoing_masses
) {
    std::size_t n = outgoing_masses.size();
    double mass_sum = 0., mass2_sum = 0.;
    for (double m : outgoing_masses) {
        mass_sum += m;
        mass2_sum += m * m;
    }
    double bound = std::max(mass_sum, cuts.sqrt_s_min());

    // the transverse masses: in the partonic centre-of-mass frame sqrt(s_hat)
    // is the sum of the energies, each object has at least its transverse
    // mass. i can take any transverse momentum, so the objects need not
    // balance.
    auto pt_min = cuts.pt_min();
    std::vector<double> transverse_mass(outgoing_masses);
    if (pt_min.size() == n) {
        for (std::size_t k = 0; k < n; ++k) {
            transverse_mass.at(k) = std::hypot(outgoing_masses.at(k), pt_min.at(k));
        }
    }
    double transverse_sum = 0.;
    for (double m_t : transverse_mass) {
        transverse_sum += m_t;
    }
    bound = std::max(bound, transverse_sum);

    // the pair masses: a pair with m_ab >= cut has at least that energy, the
    // other objects their transverse masses, so cut + their transverse
    // masses. And the invariant mass of a sum of future-pointing momenta is at
    // least that of any part, so all pairs together bound it with the
    // invariant mass of the objects, m^2 = sum_{a<b} m_ab^2
    // - (n - 2) sum_k m_k^2.
    auto pair_bound = [&](std::size_t a, std::size_t b, double cut) {
        return cut + transverse_sum - transverse_mass.at(a) - transverse_mass.at(b);
    };
    auto m_inv_min = cuts.m_inv_min();
    if (m_inv_min.size() == n) {
        double pair2_sum = 0.;
        bool any_cut = false;
        for (std::size_t a = 0; a < n; ++a) {
            for (std::size_t b = a + 1; b < n; ++b) {
                double cut = m_inv_min.at(a).at(b);
                double pair = outgoing_masses.at(a) + outgoing_masses.at(b);
                if (cut > 0.) {
                    any_cut = true;
                    bound = std::max(bound, pair_bound(a, b, cut));
                    pair = std::max(pair, cut);
                }
                pair2_sum += pair * pair;
            }
        }
        if (any_cut && n > 2) {
            double mass2 = pair2_sum - (static_cast<double>(n) - 2.) * mass2_sum;
            if (mass2 > 0.) {
                bound = std::max(bound, std::sqrt(mass2));
            }
        }
    }
    // a pair mass cut that one of several pairs has to pass: the weakest pair
    for (const auto& item : cuts.pair_mass_any_min()) {
        double item_bound = std::numeric_limits<double>::infinity();
        for (auto [a, b] : item.pairs) {
            if (a >= n || b >= n) {
                item_bound = 0.;
                break;
            }
            item_bound = std::min(item_bound, pair_bound(a, b, item.min));
        }
        if (std::isfinite(item_bound)) {
            bound = std::max(bound, item_bound);
        }
    }
    return bound;
}

FKSMapping::FKSMapping(
    const PhaseSpaceMapping& born,
    std::size_t i_fks,
    std::size_t j_fks,
    double s_hat_min
) :
    Mapping(
        "FKSMapping",
        fks_input_types(checked_born(born, j_fks)),
        fks_output_types(born),
        born.condition_types()
    ),
    _born(born),
    _radiation(
        born.particle_count(),
        i_fks,
        j_fks,
        born.leptonic(),
        s_hat_min,
        emitter_mass(born, i_fks, j_fks)
    ),
    _born_masses(born.external_masses()),
    _real_masses(real_masses(born, i_fks)) {}

std::pair<Value, Value> FKSMapping::born_frame(
    FunctionBuilder& fb, Value x1, Value x2, const NamedVector<Value>& conditions
) const {
    // the rapidity log(x1 / x2) / 2 of the Born in the beams' frame, which
    // moves with beam_rapidity in the lab. With mirror_index 1 the first leg
    // comes from the second beam, along -z: then x2 belongs to the parton
    // along +z. The products with 1 and 0 select exactly.
    double rapidity = _born.beam_rapidity();
    double exp_plus = std::exp(rapidity), exp_minus = std::exp(-rapidity);
    if (_born.leptonic()) {
        return {exp_plus, exp_minus};
    }
    Value x_plus = x1, x_minus = x2;
    if (_born.mirror_beams()) {
        auto mirror = conditions.at("mirror_index");
        auto keep = fb.gather(mirror, std::vector<double>{1., 0.});
        auto swap = fb.gather(mirror, std::vector<double>{0., 1.});
        x_plus = fb.add(fb.mul(keep, x1), fb.mul(swap, x2));
        x_minus = fb.add(fb.mul(keep, x2), fb.mul(swap, x1));
    }
    if (rapidity == 0.) {
        return {x_plus, x_minus};
    }
    return {fb.mul(x_plus, exp_plus), fb.mul(x_minus, exp_minus)};
}

Value FKSMapping::standard_orientation(
    FunctionBuilder& fb, Value momenta, const NamedVector<Value>& conditions
) const {
    // the emission needs the first incoming parton along +z: with
    // mirror_index 1 it comes along -z, and the mirror (a rotation by pi about
    // the x axis) turns it around, in either direction
    if (!_born.mirror_beams()) {
        return momenta;
    }
    return fb.mirror_momenta(momenta, conditions.at("mirror_index"));
}

Mapping::Result FKSMapping::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto events = build_events(fb, inputs, conditions, false);
    return {
        {{"momenta", events.at("momenta")},
         {"x1", events.at("x1")},
         {"x2", events.at("x2")}},
        events.at("weight")
    };
}

NamedVector<Value> FKSMapping::build_counter_events(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    return build_events(fb, inputs, conditions, true);
}

NamedVector<Value> FKSMapping::build_events(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions,
    bool with_counter_events
) const {
    bool boosted = !_born.leptonic() || _born.beam_rapidity() != 0.;
    auto r = fb.unstack(inputs.at("random"));
    std::size_t n_born = _born.random_dim();
    NamedVector<Value> born_inputs{
        {"random", fb.stack(ValueVec(r.begin(), r.begin() + n_born))}
    };
    if (discrete_dim() > 0) {
        born_inputs.push_back("discrete", inputs.at("discrete"));
    }
    auto born = _born.build_forward(fb, born_inputs, conditions);
    auto x1_born = born.at("x1"), x2_born = born.at("x2");
    if (_born.leptonic()) {
        // x1 = x2 = 1 for every point, as a batch, so that the FKS
        // configurations of a Born combine in a MultiChannelMapping or
        // MultiChannelFunction (which concatenate their outputs)
        auto batch_size = fb.batch_size({born.at("momenta")});
        x1_born = x2_born = fb.full({Value(1.), batch_size});
    }
    auto det_born = born.at("det");

    auto [plus, minus] = born_frame(fb, x1_born, x2_born, conditions);
    auto born_cm = born.at("momenta");
    if (boosted) {
        born_cm = fb.fks_beam_boost(born_cm, Value(_born_masses), minus, plus);
    }
    born_cm = standard_orientation(fb, born_cm, conditions);
    NamedVector<Value> rad_inputs{
        {"r_xi", r.at(n_born)},
        {"r_y", r.at(n_born + 1)},
        {"r_phi", r.at(n_born + 2)},
        {"born_momenta", born_cm},
    };
    if (!_born.leptonic()) {
        rad_inputs.push_back("x1", x1_born);
        rad_inputs.push_back("x2", x2_born);
    }
    auto real = with_counter_events
        ? _radiation.build_counter_events(fb, rad_inputs)
        : _radiation.build_forward(fb, rad_inputs);
    // build_forward names its weight det
    auto weight = with_counter_events ? real.at("weight") : real.at("det");

    // the momenta of the real emission and of the counter-events into the lab,
    // with their momentum fractions (those of the Born without PDFs) and
    // weights times the weight of the Born
    auto to_lab = [&](const std::string& prefix, Value det, NamedVector<Value>& out) {
        auto momenta =
            standard_orientation(fb, real.at(prefix + "momenta"), conditions);
        if (boosted) {
            momenta = fb.fks_beam_boost(momenta, Value(_real_masses), plus, minus);
        }
        out.push_back(prefix + "momenta", momenta);
        out.push_back(
            prefix + "x1", _born.leptonic() ? x1_born : real.at(prefix + "x1")
        );
        out.push_back(
            prefix + "x2", _born.leptonic() ? x2_born : real.at(prefix + "x2")
        );
        out.push_back(
            prefix + (prefix.empty() ? "weight" : "det"), fb.mul(det_born, det)
        );
    };
    NamedVector<Value> outputs;
    to_lab("", weight, outputs);
    if (!with_counter_events) {
        return outputs;
    }
    outputs.push_back("born_momenta", born.at("momenta"));
    outputs.push_back("born_x1", x1_born);
    outputs.push_back("born_x2", x2_born);
    outputs.push_back("born_weight", det_born);
    for (const auto& name : {"xi", "xi_hat", "y", "xi_max"}) {
        outputs.push_back(name, real.at(name));
    }
    for (const auto& name : _radiation.counter_events()) {
        to_lab(name + "_", real.at(name + "_det"), outputs);
    }
    return outputs;
}

NamedVector<Type> FKSMapping::counter_event_types() const {
    std::size_t n = _born_masses.size();
    NamedVector<Type> types{
        {"momenta", batch_four_vec_array(n + 1)},
        {"x1", batch_float},
        {"x2", batch_float},
        {"weight", batch_float},
        {"born_momenta", batch_four_vec_array(n)},
        {"born_x1", batch_float},
        {"born_x2", batch_float},
        {"born_weight", batch_float},
        {"xi", batch_float},
        {"xi_hat", batch_float},
        {"y", batch_float},
        {"xi_max", batch_float},
    };
    for (const auto& name : _radiation.counter_events()) {
        types.push_back(name + "_momenta", batch_four_vec_array(n + 1));
        types.push_back(name + "_x1", batch_float);
        types.push_back(name + "_x2", batch_float);
        types.push_back(name + "_det", batch_float);
    }
    return types;
}

namespace {

const Mapping& as_mapping(const FKSKinematics::Configurations& mapping) {
    return std::visit([](const auto& m) -> const Mapping& { return m; }, mapping);
}

NamedVector<Type> fks_configuration_conditions(const std::vector<FKSMapping>& configs) {
    if (configs.empty()) {
        throw std::invalid_argument(
            "FKSConfigurationMapping: needs at least one configuration"
        );
    }
    const auto& first = configs.at(0);
    for (const auto& config : configs) {
        if (config.input_types().keys() != first.input_types().keys() ||
            config.condition_types().keys() != first.condition_types().keys() ||
            config.random_dim() != first.random_dim() ||
            config.discrete_dim() != first.discrete_dim() ||
            config.particle_count() != first.particle_count() ||
            config.born().leptonic() != first.born().leptonic() ||
            config.born().external_masses() != first.born().external_masses()) {
            throw std::invalid_argument(
                "FKSConfigurationMapping: the configurations must be FKSMappings of "
                "the same Born mapping"
            );
        }
    }
    auto conditions = first.condition_types();
    conditions.push_back("fks_index", batch_int);
    return conditions;
}

} // namespace

FKSConfigurationMapping::FKSConfigurationMapping(
    const std::vector<FKSMapping>& configurations
) :
    Mapping(
        "FKSConfigurationMapping",
        configurations.empty()
            ? NamedVector<Type>{}
            : configurations.at(0).input_types(),
        configurations.empty()
            ? NamedVector<Type>{}
            : configurations.at(0).output_types(),
        fks_configuration_conditions(configurations)
    ),
    _configurations(configurations) {}

NamedVector<Value> FKSConfigurationMapping::build_split(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions,
    const std::function<NamedVector<Value>(
        const FKSMapping&, const NamedVector<Value>&, const NamedVector<Value>&
    )>& build
) const {
    auto fks_index = conditions.at("fks_index");
    auto split_indices = fb.batch_split_by_index(
        fks_index, static_cast<me_int_t>(_configurations.size())
    );
    auto condition_names = conditions.keys();
    std::vector<std::string> result_names;
    std::vector<ValueVec> merge_args;
    for (auto [config, indices] : zip(_configurations, split_indices)) {
        NamedVector<Value> part_inputs, part_conditions;
        for (auto [name, input] : zip(inputs.keys(), inputs)) {
            part_inputs.push_back(name, fb.batch_gather(indices, input));
        }
        for (std::size_t k = 0; k < conditions.size(); ++k) {
            if (condition_names.at(k) != "fks_index") {
                part_conditions.push_back(
                    condition_names.at(k), fb.batch_gather(indices, conditions.at(k))
                );
            }
        }
        auto results = build(config, part_inputs, part_conditions);
        if (result_names.empty()) {
            result_names = results.keys();
            merge_args.resize(results.size());
        }
        for (auto [result, args] : zip(results, merge_args)) {
            args.push_back(result);
            args.push_back(indices);
        }
    }
    NamedVector<Value> merged;
    for (auto [name, args] : zip(result_names, merge_args)) {
        merged.push_back(name, fb.batch_merge_by_index(args));
    }
    return merged;
}

Mapping::Result FKSConfigurationMapping::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto merged = build_split(
        fb,
        inputs,
        conditions,
        [&](const FKSMapping& config, const auto& in, const auto& cond) {
            return config.build_forward(fb, in, cond);
        }
    );
    auto det = merged.at("det");
    NamedVector<Value> outputs;
    for (const auto& name : output_types().keys()) {
        outputs.push_back(name, merged.at(name));
    }
    return {outputs, det};
}

Mapping::Result FKSConfigurationMapping::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto merged = build_split(
        fb,
        inputs,
        conditions,
        [&](const FKSMapping& config, const auto& in, const auto& cond) {
            return config.build_inverse(fb, in, cond);
        }
    );
    auto det = merged.at("det");
    NamedVector<Value> outputs;
    for (const auto& name : input_types().keys()) {
        outputs.push_back(name, merged.at(name));
    }
    return {outputs, det};
}

NamedVector<Value> FKSConfigurationMapping::build_counter_events(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    return build_split(
        fb,
        inputs,
        conditions,
        [&](const FKSMapping& config, const auto& in, const auto& cond) {
            return config.build_counter_events(fb, in, cond);
        }
    );
}

FKSKinematics::FKSKinematics(const Configurations& mapping) :
    FunctionGenerator(
        "FKSKinematics",
        [&] {
            const auto& m = as_mapping(mapping);
            auto types = m.input_types();
            const auto& conditions = m.condition_types();
            auto names = conditions.keys();
            for (std::size_t k = 0; k < conditions.size(); ++k) {
                types.push_back(names.at(k), conditions.at(k));
            }
            return types;
        }(),
        std::visit([](const auto& m) { return m.counter_event_types(); }, mapping)
    ),
    _mapping(mapping) {}

NamedVector<Value> FKSKinematics::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    const auto& m = as_mapping(_mapping);
    NamedVector<Value> inputs, conditions;
    for (const auto& name : m.input_types().keys()) {
        inputs.push_back(name, args.at(name));
    }
    for (const auto& name : m.condition_types().keys()) {
        conditions.push_back(name, args.at(name));
    }
    return std::visit(
        [&](const auto& mapping) {
            return mapping.build_counter_events(fb, inputs, conditions);
        },
        _mapping
    );
}

Mapping::Result FKSMapping::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    bool boosted = !_born.leptonic() || _born.beam_rapidity() != 0.;
    auto momenta = inputs.at("momenta");
    auto x1 = inputs.at("x1"), x2 = inputs.at("x2");

    // into the centre-of-mass frame of the Born: that of the incoming momenta
    // for a final-state emitter, of the recoil against i for an incoming one
    Value plus, minus;
    if (_radiation.initial_state()) {
        auto p = fb.unstack(momenta);
        auto [lc_plus, lc_minus] =
            fb.fks_recoil_light_cone(p.at(0), p.at(1), p.at(i_fks()));
        plus = lc_plus;
        minus = lc_minus;
    } else {
        std::tie(plus, minus) = born_frame(fb, x1, x2, conditions);
    }
    auto real_cm = momenta;
    if (boosted) {
        real_cm = fb.fks_beam_boost(momenta, Value(_real_masses), minus, plus);
    }
    real_cm = standard_orientation(fb, real_cm, conditions);
    NamedVector<Value> rad_inputs{{"momenta", real_cm}};
    if (!_born.leptonic()) {
        rad_inputs.push_back("x1", x1);
        rad_inputs.push_back("x2", x2);
    }
    auto rad = _radiation.build_inverse(fb, rad_inputs);
    auto x1_born = _born.leptonic() ? x1 : rad.at("x1");
    auto x2_born = _born.leptonic() ? x2 : rad.at("x2");

    auto born_lab = standard_orientation(fb, rad.at("born_momenta"), conditions);
    if (boosted) {
        auto [born_plus, born_minus] = born_frame(fb, x1_born, x2_born, conditions);
        born_lab =
            fb.fks_beam_boost(born_lab, Value(_born_masses), born_plus, born_minus);
    }
    auto born = _born.build_inverse(
        fb, {{"momenta", born_lab}, {"x1", x1_born}, {"x2", x2_born}}, conditions
    );
    auto r = fb.unstack(born.at("random"));
    r.push_back(rad.at("r_xi"));
    r.push_back(rad.at("r_y"));
    r.push_back(rad.at("r_phi"));
    NamedVector<Value> outputs{{"random", fb.stack(r)}};
    if (discrete_dim() > 0) {
        outputs.push_back("discrete", born.at("discrete"));
    }
    return {outputs, fb.mul(born.at("det"), rad.at("det"))};
}

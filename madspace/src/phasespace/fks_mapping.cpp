#include "madspace/phasespace/fks_mapping.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>

using namespace madspace;

namespace {

constexpr std::size_t INCOMING_COUNT = 2;

const PhaseSpaceMapping&
checked_born(const PhaseSpaceMapping& born, std::size_t j_fks) {
    if (born.sqrt_s_hat_condition()) {
        throw std::invalid_argument(
            "FKSMapping: a Born mapping with the sqrt_s_hat condition belongs to "
            "lepton beams, see FKSConfigurationMapping"
        );
    }
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

NamedVector<Type>
fks_input_types(const PhaseSpaceMapping& born, std::size_t extra = 3) {
    const auto& born_types = born.input_types();
    auto names = born_types.keys();
    NamedVector<Type> types;
    for (std::size_t k = 0; k < born_types.size(); ++k) {
        const auto& name = names.at(k);
        types.push_back(
            name,
            name == "random"
                ? batch_float_array(born.random_dim() + extra)
                : born_types.at(k)
        );
    }
    return types;
}

NamedVector<Type> fks_output_types(const PhaseSpaceMapping& born, bool lepton = false) {
    NamedVector<Type> types{
        {"momenta", batch_four_vec_array(born.particle_count() + 1)},
        {"x1", batch_float},
        {"x2", batch_float},
    };
    // with lepton PDFs 1 - x, which the momentum fractions close to 1 do not
    // resolve
    if (lepton) {
        types.push_back("one_minus_x1", batch_float);
        types.push_back("one_minus_x2", batch_float);
    }
    return types;
}

// the light-cone components (up to a factor) whose ratio is exp(2 y) for the
// rapidity y of the center-of-mass frame of the Born in the lab frame, from
// its momentum fractions (the arguments of fks_beam_boost)
std::pair<Value, Value> fks_born_frame(
    FunctionBuilder& fb,
    const PhaseSpaceMapping& born,
    Value x1,
    Value x2,
    const NamedVector<Value>& conditions
) {
    // the rapidity log(x1 / x2) / 2 of the Born in the beams' frame, which
    // moves with beam_rapidity in the lab. With mirror_index 1 the first leg
    // comes from the second beam, along -z: then x2 belongs to the parton
    // along +z. The products with 1 and 0 select exactly.
    double rapidity = born.beam_rapidity();
    double exp_plus = std::exp(rapidity), exp_minus = std::exp(-rapidity);
    if (born.leptonic()) {
        return {exp_plus, exp_minus};
    }
    Value x_plus = x1, x_minus = x2;
    if (born.mirror_beams()) {
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

// between the orientation of the beams (mirror_index) and the one of the
// emission, the first incoming parton along +z (an involution)
Value fks_standard_orientation(
    FunctionBuilder& fb,
    const PhaseSpaceMapping& born,
    Value momenta,
    const NamedVector<Value>& conditions
) {
    // the emission needs the first incoming parton along +z: with
    // mirror_index 1 it comes along -z, and the mirror (a rotation by pi about
    // the x axis) turns it around, in either direction
    if (!born.mirror_beams()) {
        return momenta;
    }
    return fb.mirror_momenta(momenta, conditions.at("mirror_index"));
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

    // the transverse masses: in the partonic center-of-mass frame sqrt(s_hat)
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
    double s_hat_min,
    FKSRadiation::InitialMap initial_map
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
        emitter_mass(born, i_fks, j_fks),
        initial_map,
        born.external_masses()
    ),
    _born_masses(born.external_masses()),
    _real_masses(real_masses(born, i_fks)) {}

std::pair<Value, Value> FKSMapping::born_frame(
    FunctionBuilder& fb, Value x1, Value x2, const NamedVector<Value>& conditions
) const {
    return fks_born_frame(fb, _born, x1, x2, conditions);
}

Value FKSMapping::standard_orientation(
    FunctionBuilder& fb, Value momenta, const NamedVector<Value>& conditions
) const {
    return fks_standard_orientation(fb, _born, momenta, conditions);
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
    // a batch also for a leptonic Born (x1 = x2 = 1), like every output of
    // PhaseSpaceMapping
    auto x1_born = born.at("x1"), x2_born = born.at("x2");
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
    // i per unit xi into the lab like a massless momentum
    auto hat_to_lab = [&](Value p_i_hat) {
        auto hat = standard_orientation(fb, fb.stack({p_i_hat}), conditions);
        if (boosted) {
            hat = fb.fks_beam_boost(hat, Value(std::vector<double>{0.}), plus, minus);
        }
        return fb.unstack(hat).at(0);
    };
    outputs.push_back("p_i_hat", hat_to_lab(real.at("p_i_hat")));
    for (const auto& name : _radiation.counter_events()) {
        to_lab(name + "_", real.at(name + "_det"), outputs);
        outputs.push_back(name + "_p_i_hat", hat_to_lab(real.at(name + "_p_i_hat")));
        outputs.push_back(name + "_xi", real.at(name + "_xi"));
        outputs.push_back(name + "_xi_max", real.at(name + "_xi_max"));
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
        {"p_i_hat", batch_four_vec},
    };
    for (const auto& name : _radiation.counter_events()) {
        types.push_back(name + "_momenta", batch_four_vec_array(n + 1));
        types.push_back(name + "_x1", batch_float);
        types.push_back(name + "_x2", batch_float);
        types.push_back(name + "_det", batch_float);
        types.push_back(name + "_p_i_hat", batch_four_vec);
        types.push_back(name + "_xi", batch_float);
        types.push_back(name + "_xi_max", batch_float);
    }
    return types;
}

namespace {

const Mapping& as_mapping(const FKSKinematics::Configurations& mapping) {
    return std::visit([](const auto& m) -> const Mapping& { return m; }, mapping);
}

const PhaseSpaceMapping& checked_configuration_born(
    const PhaseSpaceMapping& born,
    const std::vector<FKSConfigurationMapping::Configuration>& configurations,
    double sqrt_s_hat_floor,
    const std::optional<LeptonLuminosity>& luminosity
) {
    if (luminosity &&
        (!born.leptonic() || !born.sqrt_s_hat_condition() ||
         born.beam_rapidity() != 0. || born.mirror_beams())) {
        throw std::invalid_argument(
            "FKSConfigurationMapping: with a lepton luminosity the Born mapping "
            "must be leptonic, with the sqrt_s_hat condition, without a beam "
            "rapidity and without mirrored beams"
        );
    }
    if (luminosity && !luminosity->s_hat_min_condition()) {
        throw std::invalid_argument(
            "FKSConfigurationMapping: the lepton luminosity needs the s_hat_min "
            "condition (s_hat_min_condition)"
        );
    }
    if (!luminosity && born.sqrt_s_hat_condition()) {
        throw std::invalid_argument(
            "FKSConfigurationMapping: a Born mapping with the sqrt_s_hat "
            "condition needs a lepton luminosity"
        );
    }
    if (configurations.empty()) {
        throw std::invalid_argument(
            "FKSConfigurationMapping: needs at least one configuration"
        );
    }
    if (!born.cuts().empty()) {
        throw std::invalid_argument(
            "FKSConfigurationMapping: the Born mapping must not have cuts, they "
            "belong to the real emission and the counter-events"
        );
    }
    if (sqrt_s_hat_floor < 0.) {
        throw std::invalid_argument(
            "FKSConfigurationMapping: sqrt_s_hat_floor must not be negative"
        );
    }
    bool final_emitter = std::any_of(
        configurations.begin(), configurations.end(), [](const auto& config) {
            return config.second >= INCOMING_COUNT;
        }
    );
    // with a lepton luminosity the floor is that of the luminosity
    if (sqrt_s_hat_floor > 0. && final_emitter && !luminosity &&
        !born.sqrt_s_hat_min_condition()) {
        throw std::invalid_argument(
            "FKSConfigurationMapping: a floor on sqrt(s_hat) needs a Born mapping "
            "with the sqrt_s_hat_min condition (sqrt_s_hat_min_condition)"
        );
    }
    return born;
}

NamedVector<Type> configuration_conditions(const PhaseSpaceMapping& born) {
    NamedVector<Type> conditions;
    const auto& born_conditions = born.condition_types();
    auto names = born_conditions.keys();
    for (std::size_t k = 0; k < born_conditions.size(); ++k) {
        if (names.at(k) != "sqrt_s_hat_min" && names.at(k) != "sqrt_s_hat") {
            conditions.push_back(names.at(k), born_conditions.at(k));
        }
    }
    conditions.push_back("fks_index", batch_int);
    return conditions;
}

} // namespace

// what the emissions of an event need, in the Born c.m. frame
struct FKSConfigurationMapping::Context {
    Value fks_index;
    // the Born momenta, its incoming and final-state ones, its momentum
    // fractions and sqrt(s_hat)
    Value born;
    ValueVec incoming;
    ValueVec final_state;
    Value x1;
    Value x2;
    Value sqrt_s;
    Value r_phi;
    // per event, of its configuration: the Born emitter (final-state
    // emitters), the emitter mass (massive ones), s_hat_min and the beam
    // (incoming ones; for lepton beams s_hat_min is the lower end of the
    // energy squared of the Born)
    Value p_jbar;
    Value emitter_mass;
    Value s_hat_min;
    Value beam_sign;
    // the weight of the Born
    Value born_det;
};

namespace {

// (i_fks, j_fks) of FKS configurations, whose real processes without i have
// the masses of the Born mapping
std::vector<FKSConfigurationMapping::Configuration> configuration_pairs(
    const PhaseSpaceMapping& born, const std::vector<FKSConfiguration>& configurations
) {
    std::size_t n = check_fks_configurations(configurations, "FKSConfigurationMapping");
    if (n != born.particle_count() + 1) {
        throw std::invalid_argument(
            "FKSConfigurationMapping: the real processes must have one particle "
            "more than the Born mapping"
        );
    }
    auto born_masses = born.external_masses();
    std::vector<FKSConfigurationMapping::Configuration> pairs;
    for (const auto& config : configurations) {
        for (std::size_t k = 0; k < n; ++k) {
            if (k != config.i_fks &&
                config.masses.at(k) != born_masses.at(k < config.i_fks ? k : k - 1)) {
                throw std::invalid_argument(
                    "FKSConfigurationMapping: the masses of a configuration "
                    "differ from those of the Born mapping"
                );
            }
        }
        pairs.push_back({config.i_fks, config.j_fks});
    }
    return pairs;
}

} // namespace

FKSConfigurationMapping::FKSConfigurationMapping(
    const PhaseSpaceMapping& born,
    const std::vector<FKSConfiguration>& configurations,
    double s_hat_min,
    double sqrt_s_hat_floor,
    const std::optional<LeptonLuminosity>& luminosity,
    FKSRadiation::InitialMap initial_map
) :
    FKSConfigurationMapping(
        born,
        configuration_pairs(born, configurations),
        s_hat_min,
        sqrt_s_hat_floor,
        luminosity,
        initial_map
    ) {}

FKSConfigurationMapping::FKSConfigurationMapping(
    const PhaseSpaceMapping& born,
    const std::vector<Configuration>& configurations,
    double s_hat_min,
    double sqrt_s_hat_floor,
    const std::optional<LeptonLuminosity>& luminosity,
    FKSRadiation::InitialMap initial_map
) :
    Mapping(
        "FKSConfigurationMapping",
        fks_input_types(
            checked_configuration_born(
                born, configurations, sqrt_s_hat_floor, luminosity
            ),
            luminosity ? 5 : 3
        ),
        fks_output_types(born, luminosity.has_value()),
        configuration_conditions(born)
    ),
    _born(born),
    _configurations(configurations),
    _s_hat_min(s_hat_min),
    _sqrt_s_hat_floor(sqrt_s_hat_floor),
    _luminosity(luminosity),
    _initial_map(initial_map),
    _born_masses(born.external_masses()) {
    std::size_t n = born.particle_count();
    _layout_masses = _born_masses;
    _layout_masses.push_back(0.);
    // the final-state masses of the Born bound its energy from below
    double mass_sum = 0.;
    for (std::size_t b = INCOMING_COUNT; b < n; ++b) {
        mass_sum += std::max(_born_masses.at(b), 0.);
    }
    std::vector<EmitterKind> kind_of;
    for (auto [i_fks, j_fks] : configurations) {
        // FKSRadiation checks the positions, and the mass of the emitter is
        // that FKSMapping gives it. With lepton beams an incoming emitter
        // radiates without the projection of the event, which keeps the
        // momentum fractions.
        double mass = emitter_mass(born, i_fks, j_fks);
        bool fixed = luminosity && j_fks < INCOMING_COUNT;
        double radiation_s_hat_min = j_fks < INCOMING_COUNT && !luminosity
            ? std::max(s_hat_min, sqrt_s_hat_floor * sqrt_s_hat_floor)
            : luminosity
            ? 0.
            : s_hat_min;
        FKSRadiation(
            n, i_fks, j_fks, born.leptonic() && !fixed, radiation_s_hat_min, mass
        );
        auto kind = j_fks < INCOMING_COUNT ? incoming
            : mass > 0.
            ? massive_final
            : massless_final;
        kind_of.push_back(kind);
        _emitter_masses.push_back(mass);
        _born_floors.push_back(j_fks < INCOMING_COUNT ? 0. : sqrt_s_hat_floor);
        double born_s_min = j_fks < INCOMING_COUNT
            ? mass_sum * mass_sum
            : std::pow(std::max(sqrt_s_hat_floor, mass_sum), 2);
        if (luminosity && !(born_s_min < luminosity->s_lab())) {
            throw std::invalid_argument(
                "FKSConfigurationMapping: the lower end of the energy of the Born "
                "is above that of the beams"
            );
        }
        _born_s_min.push_back(born_s_min);
        _radiation_s_hat_min.push_back(fixed ? born_s_min : radiation_s_hat_min);
        _beam_signs.push_back(j_fks == 1 ? -1. : 1.);
        _collinear_exists.push_back(kind == massive_final ? 0. : 1.);

        auto born_index = [&](std::size_t r) { return r < i_fks ? r : r - 1; };
        // the Born emitter (final-state emitters) first; any order else
        std::size_t jb = j_fks < INCOMING_COUNT ? 0 : born_index(j_fks);
        std::vector<me_int_t> emitter_first{static_cast<me_int_t>(jb)};
        for (std::size_t b = 0; b < n; ++b) {
            if (b != jb) {
                emitter_first.push_back(b);
            }
        }
        _emitter_first.push_back(emitter_first);
        // [Born order, j, i] -> [Born order with j at jbar, i, (jbar)]
        std::vector<me_int_t> emission_into_place;
        for (std::size_t b = 0; b < n; ++b) {
            emission_into_place.push_back(b == jb ? n : b);
        }
        emission_into_place.push_back(n + 1);
        emission_into_place.push_back(jb);
        _emission_into_place.push_back(emission_into_place);
        // [Born order, jbar] -> [Born order with jbar in place, (j)]
        std::vector<me_int_t> jbar_into_place;
        for (std::size_t b = 0; b < n; ++b) {
            jbar_into_place.push_back(b == jb ? n : b);
        }
        jbar_into_place.push_back(jb);
        _jbar_into_place.push_back(jbar_into_place);
        // the layout (Born order, i last) -> the real process, and back
        std::vector<me_int_t> real_order, layout_order;
        for (std::size_t r = 0; r < n + 1; ++r) {
            real_order.push_back(r == i_fks ? n : born_index(r));
        }
        for (std::size_t b = 0; b < n; ++b) {
            layout_order.push_back(b < i_fks ? b : b + 1);
        }
        layout_order.push_back(i_fks);
        _real_order.push_back(real_order);
        _layout_order.push_back(layout_order);
        // the emitter j first in the layout
        std::vector<me_int_t> layout_emitter_first{static_cast<me_int_t>(jb)};
        for (std::size_t b = 0; b < n + 1; ++b) {
            if (b != jb) {
                layout_emitter_first.push_back(b);
            }
        }
        _layout_emitter_first.push_back(layout_emitter_first);
    }
    for (auto kind : {incoming, massless_final, massive_final}) {
        if (std::find(kind_of.begin(), kind_of.end(), kind) != kind_of.end()) {
            _kinds.push_back(kind);
        }
    }
    for (auto kind : kind_of) {
        _kind_slots.push_back(
            std::find(_kinds.begin(), _kinds.end(), kind) - _kinds.begin()
        );
    }
}

bool FKSConfigurationMapping::has(EmitterKind kind) const {
    return std::find(_kinds.begin(), _kinds.end(), kind) != _kinds.end();
}

NamedVector<Type> FKSConfigurationMapping::counter_event_types() const {
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
        {"p_i_hat", batch_four_vec},
    };
    for (const auto& name : {"soft", "collinear", "soft_collinear"}) {
        types.push_back(std::string(name) + "_momenta", batch_four_vec_array(n + 1));
        types.push_back(std::string(name) + "_x1", batch_float);
        types.push_back(std::string(name) + "_x2", batch_float);
        types.push_back(std::string(name) + "_det", batch_float);
        types.push_back(std::string(name) + "_p_i_hat", batch_four_vec);
        types.push_back(std::string(name) + "_xi", batch_float);
        types.push_back(std::string(name) + "_xi_max", batch_float);
    }
    if (_luminosity) {
        types.push_back("one_minus_x1", batch_float);
        types.push_back("one_minus_x2", batch_float);
        types.push_back("collinear_born_momenta", batch_four_vec_array(n));
    }
    return types;
}

NamedVector<Value> FKSConfigurationMapping::born_conditions(
    FunctionBuilder& fb,
    const NamedVector<Value>& conditions,
    const std::optional<Value>& sqrt_s
) const {
    NamedVector<Value> born_conditions;
    for (const auto& name : _born.condition_types().keys()) {
        if (name == "sqrt_s_hat") {
            born_conditions.push_back(name, sqrt_s.value());
            continue;
        }
        born_conditions.push_back(
            name,
            name == "sqrt_s_hat_min"
                ? fb.gather(conditions.at("fks_index"), _born_floors)
                : conditions.at(name)
        );
    }
    return born_conditions;
}

NamedVector<Value> FKSConfigurationMapping::fixed_born(
    FunctionBuilder& fb,
    const NamedVector<Value>& born_inputs,
    const NamedVector<Value>& conditions,
    Value sqrt_s
) const {
    return _born.build_forward(
        fb, born_inputs, born_conditions(fb, conditions, sqrt_s)
    );
}

Value FKSConfigurationMapping::pick(
    FunctionBuilder& fb, const ValueVec& by_kind, Value slot
) const {
    if (by_kind.size() == 1) {
        return by_kind.at(0);
    }
    return fb.gather(slot, fb.stack(by_kind));
}

Value FKSConfigurationMapping::pick_rows(
    FunctionBuilder& fb, const ValueVec& by_kind, std::size_t rows, Value fks_index
) const {
    if (by_kind.size() == 1) {
        return by_kind.at(0);
    }
    // the rows of all kinds one after the other, those of the kind of the
    // configuration first
    ValueVec all;
    for (const auto& momenta : by_kind) {
        auto unstacked = fb.unstack(momenta);
        all.insert(all.end(), unstacked.begin(), unstacked.end());
    }
    nested_vector2<me_int_t> table;
    for (auto slot : _kind_slots) {
        std::vector<me_int_t> order;
        for (std::size_t r = 0; r < rows; ++r) {
            order.push_back(slot * rows + r);
        }
        for (std::size_t k = 0; k < all.size(); ++k) {
            if (k / rows != static_cast<std::size_t>(slot)) {
                order.push_back(k);
            }
        }
        table.push_back(order);
    }
    auto picked = fb.unstack(fb.permute_momenta(fb.stack(all), table, fks_index));
    return fb.stack(ValueVec(picked.begin(), picked.begin() + rows));
}

FKSConfigurationMapping::Emission FKSConfigurationMapping::pick_emission(
    FunctionBuilder& fb,
    const std::vector<Emission>& by_kind,
    Value slot,
    Value fks_index
) const {
    auto field = [&](Value Emission::* member) {
        ValueVec values;
        for (const auto& emission : by_kind) {
            values.push_back(emission.*member);
        }
        return values;
    };
    ValueVec p_i_hat;
    for (const auto& value : field(&Emission::p_i_hat)) {
        p_i_hat.push_back(fb.stack({value}));
    }
    return {
        pick_rows(fb, field(&Emission::momenta), _born_masses.size() + 1, fks_index),
        pick(fb, field(&Emission::x1), slot),
        pick(fb, field(&Emission::x2), slot),
        pick(fb, field(&Emission::xi), slot),
        pick(fb, field(&Emission::y), slot),
        pick(fb, field(&Emission::xi_max), slot),
        pick(fb, field(&Emission::det_xi), slot),
        pick(fb, field(&Emission::rat), slot),
        fb.unstack(pick_rows(fb, p_i_hat, 1, fks_index)).at(0),
        pick(fb, field(&Emission::born_det), slot),
    };
}

FKSConfigurationMapping::Emission FKSConfigurationMapping::build_fixed_emission(
    FunctionBuilder& fb,
    const Context& context,
    Value xi_hat,
    Value y_var,
    const FixedBorn& born
) const {
    const auto& c = context;
    auto [p_i, xi, xi_max, sqrt_s_rec, det_xi, p_i_hat] = fb.fks_initial_fixed_radiate(
        xi_hat, y_var, c.r_phi, c.sqrt_s, c.s_hat_min, c.beam_sign
    );
    auto final_state = fb.stack(born.final_state);
    auto boosted = fb.unstack(
        _initial_map == FKSRadiation::asymmetric
            ? fb.fks_initial_fixed_light_cone_map(
                  final_state, Value(final_masses()), xi, y_var, c.r_phi, c.beam_sign
              )
            : fb.fks_initial_fixed_boost(final_state, p_i_hat, xi)
    );
    // the incoming momenta of the real emission, those of the Born at sqrt(s)
    ValueVec layout(c.incoming);
    layout.insert(layout.end(), boosted.begin(), boosted.end());
    layout.push_back(p_i);
    return {
        fb.stack(layout),
        c.x1,
        c.x2,
        xi,
        y_var,
        xi_max,
        det_xi,
        Value(1.),
        p_i_hat,
        born.det
    };
}

FKSConfigurationMapping::Emission FKSConfigurationMapping::build_emission(
    FunctionBuilder& fb,
    const Context& context,
    EmitterKind kind,
    Value xi_hat,
    Value y_var
) const {
    const auto& c = context;
    if (kind == incoming) {
        bool asymmetric = _initial_map == FKSRadiation::asymmetric;
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
                  xi_hat, y_var, c.r_phi, c.sqrt_s, c.x1, c.x2, c.s_hat_min, c.beam_sign
              )
            : fb.fks_initial_radiate(
                  xi_hat, y_var, c.r_phi, c.sqrt_s, c.x1, c.x2, c.s_hat_min, c.beam_sign
              );
        auto final_state = fb.stack(c.final_state);
        auto boosted = fb.unstack(
            asymmetric
                ? fb.fks_initial_light_cone_map(
                      final_state, Value(final_masses()), xi, y, c.r_phi, c.beam_sign
                  )
                : fb.fks_initial_recoil_boost(final_state, xi, y, c.r_phi, c.beam_sign)
        );
        ValueVec layout{p_in1, p_in2};
        layout.insert(layout.end(), boosted.begin(), boosted.end());
        layout.push_back(p_i);
        return {
            fb.stack(layout),
            x1_real,
            x2_real,
            xi,
            y,
            xi_max,
            det_xi,
            Value(1.),
            p_i_hat,
            c.born_det
        };
    }
    Value p_i, p_j, xi, xi_max, det_xi, rat(1.), p_i_hat;
    if (kind == massive_final) {
        auto [p_i_m, p_j_m, xi_m, xi_max_m, xi_norm, det_xi_m, p_i_hat_m] =
            fb.fks_final_massive_radiate(
                xi_hat, y_var, c.r_phi, c.sqrt_s, c.p_jbar, c.emitter_mass
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
            fb.fks_final_massless_radiate(xi_hat, y_var, c.r_phi, c.sqrt_s, c.p_jbar);
        p_i = p_i_m;
        p_j = p_j_m;
        xi = xi_m;
        xi_max = xi_max_m;
        det_xi = det_xi_m;
        p_i_hat = p_i_hat_m;
    }
    // the whole final state of the Born recoils, the Born emitter included,
    // which j then replaces (the boost acts on each momentum alone)
    auto boosted = fb.unstack(fb.fks_recoil_boost(
        fb.stack(c.final_state), fb.add(p_i, p_j), c.p_jbar, c.sqrt_s
    ));
    ValueVec rows(c.incoming);
    rows.insert(rows.end(), boosted.begin(), boosted.end());
    rows.push_back(p_j);
    rows.push_back(p_i);
    auto placed = fb.unstack(
        fb.permute_momenta(fb.stack(rows), _emission_into_place, c.fks_index)
    );
    auto layout = fb.stack(ValueVec(placed.begin(), placed.end() - 1));
    return {layout, c.x1, c.x2, xi, y_var, xi_max, det_xi, rat, p_i_hat, c.born_det};
}

NamedVector<Value> FKSConfigurationMapping::build_events(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions,
    bool with_counter_events
) const {
    bool lepton = _luminosity.has_value();
    bool boosted = lepton || !_born.leptonic() || _born.beam_rapidity() != 0.;
    auto fks_index = conditions.at("fks_index");
    auto r = fb.unstack(inputs.at("random"));
    std::size_t n_born = _born.random_dim();
    // the random numbers of the emission follow those of the Born and of the
    // luminosity
    std::size_t n_before = n_born + (lepton ? 2 : 0);
    NamedVector<Value> born_inputs{
        {"random", fb.stack(ValueVec(r.begin(), r.begin() + n_born))}
    };
    if (discrete_dim() > 0) {
        born_inputs.push_back("discrete", inputs.at("discrete"));
    }

    // the momentum fractions and the Born at the partonic energy: of the
    // luminosity and the Born mapping at its energy for lepton beams, of the
    // Born mapping else
    Value x1_born, x2_born, det_lumi(1.), omx1, omx2, s_born_min, sqrt_s_lumi;
    NamedVector<Value> born;
    if (lepton) {
        s_born_min = fb.gather(fks_index, _born_s_min);
        auto lumi = _luminosity->build_forward(
            fb, ValueVec{r.at(n_born), r.at(n_born + 1)}, ValueVec{s_born_min}
        );
        x1_born = lumi.at("x1");
        x2_born = lumi.at("x2");
        omx1 = lumi.at("one_minus_x1");
        omx2 = lumi.at("one_minus_x2");
        det_lumi = lumi.at("det");
        sqrt_s_lumi = fb.sqrt(lumi.at("s_hat"));
        born = fixed_born(fb, born_inputs, conditions, sqrt_s_lumi);
    } else {
        born = _born.build_forward(fb, born_inputs, born_conditions(fb, conditions));
        x1_born = born.at("x1");
        x2_born = born.at("x2");
    }
    auto det_born = born.at("det");

    Value plus = x1_born, minus = x2_born;
    if (!lepton) {
        std::tie(plus, minus) = fks_born_frame(fb, _born, x1_born, x2_born, conditions);
    }
    auto born_cm = born.at("momenta");
    if (boosted && !lepton) {
        born_cm = fb.fks_beam_boost(born_cm, Value(_born_masses), minus, plus);
    }
    born_cm = fks_standard_orientation(fb, _born, born_cm, conditions);

    Context c;
    c.fks_index = fks_index;
    c.born = born_cm;
    auto p_born = fb.unstack(born_cm);
    c.incoming = ValueVec(p_born.begin(), p_born.begin() + INCOMING_COUNT);
    c.final_state = ValueVec(p_born.begin() + INCOMING_COUNT, p_born.end());
    c.x1 = x1_born;
    c.x2 = x2_born;
    c.sqrt_s = lepton ? sqrt_s_lumi : fb.obs_sqrt_s(born_cm);
    c.r_phi = r.at(n_before + 2);
    c.born_det = det_born;
    if (has(massless_final) || has(massive_final)) {
        c.p_jbar =
            fb.unstack(fb.permute_momenta(born_cm, _emitter_first, fks_index)).at(0);
    }
    if (has(massive_final)) {
        c.emitter_mass = fb.gather(fks_index, _emitter_masses);
    }
    if (has(incoming)) {
        c.s_hat_min = fb.gather(fks_index, _radiation_s_hat_min);
        c.beam_sign = fb.gather(fks_index, _beam_signs);
    }
    Value slot;
    if (_kinds.size() > 1) {
        slot = fb.gather_int(fks_index, _kind_slots);
    }

    // xi_hat and y (y_hat for an incoming emitter) of each kind of emitter
    auto r_xi = r.at(n_before), r_y = r.at(n_before + 1);
    ValueVec xi_hat_k, y_k, det_sample_k;
    std::optional<std::tuple<Value, Value, Value>> plain_sample;
    for (auto kind : _kinds) {
        if (kind == massive_final) {
            auto [xi_hat, y, det] = fb.fks_final_massive_sample(
                r_xi, r_y, c.sqrt_s, c.p_jbar, c.emitter_mass
            );
            xi_hat_k.push_back(xi_hat);
            y_k.push_back(y);
            det_sample_k.push_back(det);
        } else {
            if (!plain_sample) {
                auto [xi_hat, y, det] = fb.fks_xi_y_sample(r_xi, r_y);
                plain_sample = std::make_tuple(xi_hat, y, det);
            }
            xi_hat_k.push_back(std::get<0>(*plain_sample));
            y_k.push_back(std::get<1>(*plain_sample));
            det_sample_k.push_back(std::get<2>(*plain_sample));
        }
    }
    auto xi_hat = pick(fb, xi_hat_k, slot);
    auto det_sample = pick(fb, det_sample_k, slot);

    // with lepton beams an incoming emitter radiates off the Born of the
    // energy it leaves (that of the event, also for the collinear
    // counter-event) and off the Born at sqrt(s) (the soft ones)
    bool fixed = lepton && has(incoming);
    FixedBorn born_at_s{c.final_state, det_born}, born_at_rec;
    if (fixed) {
        std::size_t k =
            std::find(_kinds.begin(), _kinds.end(), incoming) - _kinds.begin();
        [[maybe_unused]] auto [p_i, xi, xi_max, sqrt_s_rec, det_xi, p_i_hat] =
            fb.fks_initial_fixed_radiate(
                xi_hat_k.at(k), y_k.at(k), c.r_phi, c.sqrt_s, c.s_hat_min, c.beam_sign
            );
        auto born_rec = fixed_born(fb, born_inputs, conditions, sqrt_s_rec);
        auto p_rec = fb.unstack(born_rec.at("momenta"));
        born_at_rec = {
            ValueVec(p_rec.begin() + INCOMING_COUNT, p_rec.end()), born_rec.at("det")
        };
    }
    auto emission = [&](EmitterKind kind, Value xi_hat_kind, Value y_kind, bool soft) {
        if (fixed && kind == incoming) {
            return build_fixed_emission(
                fb, c, xi_hat_kind, y_kind, soft ? born_at_s : born_at_rec
            );
        }
        return build_emission(fb, c, kind, xi_hat_kind, y_kind);
    };

    std::vector<Emission> event_k;
    for (auto [kind, xi_hat_kind, y_kind] : zip(_kinds, xi_hat_k, y_k)) {
        event_k.push_back(emission(kind, xi_hat_kind, y_kind, false));
    }
    auto event = pick_emission(fb, event_k, slot, fks_index);

    // into the lab frame, in the order of the real process, with the weights
    // times the weight of the Born (and of the luminosity of lepton beams)
    auto to_lab =
        [&](const std::string& prefix,
            const Emission& emission,
            Value det,
            NamedVector<Value>& out) {
            auto momenta =
                fks_standard_orientation(fb, _born, emission.momenta, conditions);
            if (boosted) {
                momenta =
                    fb.fks_beam_boost(momenta, Value(_layout_masses), plus, minus);
            }
            out.push_back(
                prefix + "momenta", fb.permute_momenta(momenta, _real_order, fks_index)
            );
            out.push_back(prefix + "x1", emission.x1);
            out.push_back(prefix + "x2", emission.x2);
            auto born_weight =
                lepton ? fb.mul(det_lumi, emission.born_det) : Value(det_born);
            out.push_back(
                prefix + (prefix.empty() ? "weight" : "det"), fb.mul(born_weight, det)
            );
        };
    NamedVector<Value> outputs;
    auto weight = fb.mul(fb.mul(event.xi, event.det_xi), det_sample);
    to_lab("", event, weight, outputs);
    if (!with_counter_events) {
        if (lepton) {
            outputs.push_back("one_minus_x1", omx1);
            outputs.push_back("one_minus_x2", omx2);
        }
        return outputs;
    }
    auto born_lab = born.at("momenta");
    if (lepton) {
        born_lab = fb.fks_beam_boost(born_lab, Value(_born_masses), plus, minus);
    }
    outputs.push_back("born_momenta", born_lab);
    outputs.push_back("born_x1", x1_born);
    outputs.push_back("born_x2", x2_born);
    outputs.push_back("born_weight", lepton ? fb.mul(det_lumi, det_born) : det_born);
    outputs.push_back("xi", event.xi);
    outputs.push_back("xi_hat", xi_hat);
    outputs.push_back("y", event.y);
    outputs.push_back("xi_max", event.xi_max);
    // i per unit xi into the lab like a massless momentum
    auto hat_to_lab = [&](Value p_i_hat) {
        auto hat = fks_standard_orientation(fb, _born, fb.stack({p_i_hat}), conditions);
        if (boosted) {
            hat = fb.fks_beam_boost(hat, Value(std::vector<double>{0.}), plus, minus);
        }
        return fb.unstack(hat).at(0);
    };
    outputs.push_back("p_i_hat", hat_to_lab(event.p_i_hat));

    Value collinear_exists(1.);
    if (has(massive_final)) {
        collinear_exists = fb.gather(fks_index, _collinear_exists);
    }
    std::optional<Emission> collinear_event;
    for (std::string name : {"soft", "collinear", "soft_collinear"}) {
        bool soft = name != "collinear", collinear = name != "soft";
        std::vector<Emission> counter_k;
        for (auto [kind, xi_hat_kind, y_kind, event_kind] :
             zip(_kinds, xi_hat_k, y_k, event_k)) {
            // a massive emitter has no collinear counter-events: they keep the
            // layout with the event, and det = 0
            counter_k.push_back(
                collinear && kind == massive_final
                    ? event_kind
                    : emission(
                          kind,
                          soft ? Value(0.) : xi_hat_kind,
                          collinear ? Value(1.) : y_kind,
                          soft
                      )
            );
        }
        auto counter = pick_emission(fb, counter_k, slot, fks_index);
        auto det = fb.fks_counter_event_det(
            fb.mul(counter.det_xi, det_sample),
            counter.xi,
            counter.y,
            xi_hat,
            event.rat,
            soft ? 1. : 0.,
            collinear ? 1. : 0.,
            collinear ? collinear_exists : Value(1.)
        );
        to_lab(name + "_", counter, det, outputs);
        outputs.push_back(name + "_p_i_hat", hat_to_lab(counter.p_i_hat));
        outputs.push_back(name + "_xi", counter.xi);
        outputs.push_back(name + "_xi_max", counter.xi_max);
        if (name == "collinear") {
            collinear_event = counter;
        }
    }
    if (lepton) {
        outputs.push_back("one_minus_x1", omx1);
        outputs.push_back("one_minus_x2", omx2);
        // the Born of the collinear limit: for an incoming emitter the
        // collinear counter-event without i and with j - i = (1 - xi) j, else
        // the Born
        Value collinear_born = born_lab;
        if (fixed) {
            // the candidates in the c.m. frame one after the other, j from
            // the first beam, from the second, and the Born, and per
            // configuration its own first
            std::size_t n = _born_masses.size();
            auto rows = fb.unstack(collinear_event->momenta);
            auto p_i = rows.back();
            ValueVec finals(rows.begin() + INCOMING_COUNT, rows.end() - 1);
            ValueVec all{fb.sub(rows.at(0), p_i), rows.at(1)};
            all.insert(all.end(), finals.begin(), finals.end());
            all.push_back(rows.at(0));
            all.push_back(fb.sub(rows.at(1), p_i));
            all.insert(all.end(), finals.begin(), finals.end());
            all.insert(all.end(), p_born.begin(), p_born.end());
            nested_vector2<me_int_t> table;
            for (auto [i_fks, j_fks] : _configurations) {
                std::size_t first = j_fks < INCOMING_COUNT ? j_fks * n : 2 * n;
                std::vector<me_int_t> order;
                for (std::size_t k = 0; k < n; ++k) {
                    order.push_back(first + k);
                }
                for (std::size_t k = 0; k < all.size(); ++k) {
                    if (k < first || k >= first + n) {
                        order.push_back(k);
                    }
                }
                table.push_back(order);
            }
            auto picked =
                fb.unstack(fb.permute_momenta(fb.stack(all), table, fks_index));
            collinear_born = fb.fks_beam_boost(
                fb.stack(ValueVec(picked.begin(), picked.begin() + n)),
                Value(_born_masses),
                plus,
                minus
            );
        }
        outputs.push_back("collinear_born_momenta", collinear_born);
    }
    return outputs;
}

Mapping::Result FKSConfigurationMapping::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    auto events = build_events(fb, inputs, conditions, false);
    NamedVector<Value> outputs{
        {"momenta", events.at("momenta")},
        {"x1", events.at("x1")},
        {"x2", events.at("x2")}
    };
    if (_luminosity) {
        outputs.push_back("one_minus_x1", events.at("one_minus_x1"));
        outputs.push_back("one_minus_x2", events.at("one_minus_x2"));
    }
    return {outputs, events.at("weight")};
}

NamedVector<Value> FKSConfigurationMapping::build_counter_events(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    return build_events(fb, inputs, conditions, true);
}

Mapping::Result FKSConfigurationMapping::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    bool lepton = _luminosity.has_value();
    bool boosted = lepton || !_born.leptonic() || _born.beam_rapidity() != 0.;
    std::size_t n = _born_masses.size();
    auto fks_index = conditions.at("fks_index");
    auto layout = fb.permute_momenta(inputs.at("momenta"), _layout_order, fks_index);
    auto x1 = inputs.at("x1"), x2 = inputs.at("x2");
    Value slot;
    if (_kinds.size() > 1) {
        slot = fb.gather_int(fks_index, _kind_slots);
    }
    if (lepton) {
        return build_lepton_inverse(
            fb,
            layout,
            x1,
            x2,
            inputs.at("one_minus_x1"),
            inputs.at("one_minus_x2"),
            slot,
            conditions
        );
    }

    // into the center-of-mass frame of the Born: that of the incoming momenta
    // for a final-state emitter, of the recoil against i for an incoming one
    ValueVec plus_k, minus_k;
    auto p_lab = fb.unstack(layout);
    for (auto kind : _kinds) {
        if (kind == incoming && _initial_map == FKSRadiation::asymmetric) {
            auto [x1_born, x2_born] = fb.fks_initial_asymmetric_born_fractions(
                p_lab.at(0),
                p_lab.at(1),
                p_lab.at(n),
                x1,
                x2,
                fb.gather(fks_index, _beam_signs)
            );
            auto [plus, minus] =
                fks_born_frame(fb, _born, x1_born, x2_born, conditions);
            plus_k.push_back(plus);
            minus_k.push_back(minus);
        } else if (kind == incoming) {
            auto [plus, minus] =
                fb.fks_recoil_light_cone(p_lab.at(0), p_lab.at(1), p_lab.at(n));
            plus_k.push_back(plus);
            minus_k.push_back(minus);
        } else {
            auto [plus, minus] = fks_born_frame(fb, _born, x1, x2, conditions);
            plus_k.push_back(plus);
            minus_k.push_back(minus);
        }
    }
    auto real_cm = layout;
    if (boosted) {
        real_cm = fb.fks_beam_boost(
            layout,
            Value(_layout_masses),
            pick(fb, minus_k, slot),
            pick(fb, plus_k, slot)
        );
    }
    real_cm = fks_standard_orientation(fb, _born, real_cm, conditions);
    auto p = fb.unstack(real_cm);
    auto p_i = p.at(n);
    ValueVec final_state(p.begin() + INCOMING_COUNT, p.end() - 1);

    Value sqrt_s, p_j, emitter_mass, s_hat_min, beam_sign;
    if (has(massless_final) || has(massive_final)) {
        sqrt_s = fb.obs_sqrt_s(real_cm);
        p_j = fb.unstack(fb.permute_momenta(real_cm, _layout_emitter_first, fks_index))
                  .at(0);
    }
    if (has(massive_final)) {
        emitter_mass = fb.gather(fks_index, _emitter_masses);
    }
    if (has(incoming)) {
        s_hat_min = fb.gather(fks_index, _radiation_s_hat_min);
        beam_sign = fb.gather(fks_index, _beam_signs);
    }

    // the Born in its c.m. frame, its momentum fractions, the random numbers
    // and the inverse weight of the emission, for each kind of emitter
    ValueVec born_k, x1_k, x2_k, r_xi_k, r_y_k, r_phi_k, det_k;
    for (auto kind : _kinds) {
        if (kind == incoming) {
            bool asymmetric = _initial_map == FKSRadiation::asymmetric;
            auto
                [sqrt_s_born,
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
                      p.at(0), p.at(1), p_i, x1, x2, s_hat_min, beam_sign
                  )
                : fb.fks_initial_cluster(
                      p.at(0), p.at(1), p_i, x1, x2, s_hat_min, beam_sign
                  );
            auto [r_xi, r_y, det_sample_inv] =
                fb.fks_xi_y_sample_inverse(xi_hat, y_hat);
            auto final_stack = fb.stack(final_state);
            auto boosted_final = fb.unstack(
                asymmetric
                    ? fb.fks_initial_light_cone_map_inverse(
                          final_stack, Value(final_masses()), xi, y, r_phi, beam_sign
                      )
                    : fb.fks_initial_recoil_boost_inverse(
                          final_stack, xi, y, r_phi, beam_sign
                      )
            );
            auto [p_in1, p_in2] = fb.com_p_in(sqrt_s_born);
            ValueVec born{p_in1, p_in2};
            born.insert(born.end(), boosted_final.begin(), boosted_final.end());
            born_k.push_back(fb.stack(born));
            x1_k.push_back(x1_born);
            x2_k.push_back(x2_born);
            r_xi_k.push_back(r_xi);
            r_y_k.push_back(r_y);
            r_phi_k.push_back(r_phi);
            det_k.push_back(fb.div(fb.mul(det_xi_inv, det_sample_inv), xi));
            continue;
        }
        Value p_jbar, r_xi, r_y, r_phi, xi, det_xi_inv, det_sample_inv;
        if (kind == massive_final) {
            auto [p_jbar_m, xi_hat, y, r_phi_m, xi_m, xi_max, xi_norm, det_m] =
                fb.fks_final_massive_cluster(p_i, p_j, sqrt_s, emitter_mass);
            auto [r_xi_m, r_y_m, det] = fb.fks_final_massive_sample_inverse(
                xi_hat, y, sqrt_s, p_jbar_m, emitter_mass
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
        // the whole final state recoils back, j included, which the Born
        // emitter then replaces
        auto boosted_final = fb.unstack(fb.fks_recoil_boost_inverse(
            fb.stack(final_state), fb.add(p_i, p_j), p_jbar, sqrt_s
        ));
        ValueVec rows{p.at(0), p.at(1)};
        rows.insert(rows.end(), boosted_final.begin(), boosted_final.end());
        rows.push_back(p_jbar);
        auto placed =
            fb.unstack(fb.permute_momenta(fb.stack(rows), _jbar_into_place, fks_index));
        born_k.push_back(fb.stack(ValueVec(placed.begin(), placed.end() - 1)));
        x1_k.push_back(x1);
        x2_k.push_back(x2);
        r_xi_k.push_back(r_xi);
        r_y_k.push_back(r_y);
        r_phi_k.push_back(r_phi);
        det_k.push_back(fb.div(fb.mul(det_xi_inv, det_sample_inv), xi));
    }
    auto born_cm = pick_rows(fb, born_k, n, fks_index);
    auto x1_born = pick(fb, x1_k, slot), x2_born = pick(fb, x2_k, slot);

    auto born_lab = fks_standard_orientation(fb, _born, born_cm, conditions);
    if (boosted) {
        auto [born_plus, born_minus] =
            fks_born_frame(fb, _born, x1_born, x2_born, conditions);
        born_lab =
            fb.fks_beam_boost(born_lab, Value(_born_masses), born_plus, born_minus);
    }
    auto born = _born.build_inverse(
        fb,
        {{"momenta", born_lab}, {"x1", x1_born}, {"x2", x2_born}},
        born_conditions(fb, conditions)
    );
    auto r = fb.unstack(born.at("random"));
    r.push_back(pick(fb, r_xi_k, slot));
    r.push_back(pick(fb, r_y_k, slot));
    r.push_back(pick(fb, r_phi_k, slot));
    NamedVector<Value> outputs{{"random", fb.stack(r)}};
    if (discrete_dim() > 0) {
        outputs.push_back("discrete", born.at("discrete"));
    }
    return {outputs, fb.mul(born.at("det"), pick(fb, det_k, slot))};
}

Mapping::Result FKSConfigurationMapping::build_lepton_inverse(
    FunctionBuilder& fb,
    Value layout,
    Value x1,
    Value x2,
    Value omx1,
    Value omx2,
    Value slot,
    const NamedVector<Value>& conditions
) const {
    std::size_t n = _born_masses.size();
    auto fks_index = conditions.at("fks_index");
    // into the partonic c.m. frame, which keeps the momentum fractions
    auto real_cm = fb.fks_beam_boost(layout, Value(_layout_masses), x2, x1);
    auto p = fb.unstack(real_cm);
    auto p_i = p.at(n);
    ValueVec final_state(p.begin() + INCOMING_COUNT, p.end() - 1);
    auto s_born_min = fb.gather(fks_index, _born_s_min);
    auto s_lab = _luminosity->s_lab();
    auto s_hat = fb.mul(fb.mul(x1, x2), s_lab);
    auto sqrt_s = fb.sqrt(s_hat);

    Value p_j, emitter_mass, beam_sign;
    if (has(massless_final) || has(massive_final)) {
        p_j = fb.unstack(fb.permute_momenta(real_cm, _layout_emitter_first, fks_index))
                  .at(0);
    }
    if (has(massive_final)) {
        emitter_mass = fb.gather(fks_index, _emitter_masses);
    }
    if (has(incoming)) {
        beam_sign = fb.gather(fks_index, _beam_signs);
    }

    // the Born in its c.m. frame, its energy, the random numbers and the
    // inverse weight of the emission, for each kind of emitter
    ValueVec born_k, sqrt_s_k, r_xi_k, r_y_k, r_phi_k, det_k;
    for (auto kind : _kinds) {
        if (kind == incoming) {
            auto [xi_hat, y, r_phi, xi, sqrt_s_rec, det_xi_inv] =
                fb.fks_initial_fixed_cluster(p_i, sqrt_s, s_born_min, beam_sign);
            auto [r_xi, r_y, det_sample_inv] = fb.fks_xi_y_sample_inverse(xi_hat, y);
            // the direction of i, which the boost takes
            auto final_stack = fb.stack(final_state);
            auto rest = fb.unstack(
                _initial_map == FKSRadiation::asymmetric
                    ? fb.fks_initial_fixed_light_cone_map_inverse(
                          final_stack, Value(final_masses()), xi, y, r_phi, beam_sign
                      )
                    : fb.fks_initial_fixed_boost_inverse(final_stack, p_i, xi)
            );
            auto [p_in1, p_in2] = fb.com_p_in(sqrt_s_rec);
            ValueVec born{p_in1, p_in2};
            born.insert(born.end(), rest.begin(), rest.end());
            born_k.push_back(fb.stack(born));
            sqrt_s_k.push_back(sqrt_s_rec);
            r_xi_k.push_back(r_xi);
            r_y_k.push_back(r_y);
            r_phi_k.push_back(r_phi);
            det_k.push_back(fb.div(fb.mul(det_xi_inv, det_sample_inv), xi));
            continue;
        }
        Value p_jbar, r_xi, r_y, r_phi, xi, det_xi_inv, det_sample_inv;
        if (kind == massive_final) {
            auto [p_jbar_m, xi_hat, y, r_phi_m, xi_m, xi_max, xi_norm, det_m] =
                fb.fks_final_massive_cluster(p_i, p_j, sqrt_s, emitter_mass);
            auto [r_xi_m, r_y_m, det] = fb.fks_final_massive_sample_inverse(
                xi_hat, y, sqrt_s, p_jbar_m, emitter_mass
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
        auto boosted_final = fb.unstack(fb.fks_recoil_boost_inverse(
            fb.stack(final_state), fb.add(p_i, p_j), p_jbar, sqrt_s
        ));
        ValueVec rows{p.at(0), p.at(1)};
        rows.insert(rows.end(), boosted_final.begin(), boosted_final.end());
        rows.push_back(p_jbar);
        auto placed =
            fb.unstack(fb.permute_momenta(fb.stack(rows), _jbar_into_place, fks_index));
        born_k.push_back(fb.stack(ValueVec(placed.begin(), placed.end() - 1)));
        sqrt_s_k.push_back(sqrt_s);
        r_xi_k.push_back(r_xi);
        r_y_k.push_back(r_y);
        r_phi_k.push_back(r_phi);
        det_k.push_back(fb.div(fb.mul(det_xi_inv, det_sample_inv), xi));
    }
    auto born_cm = pick_rows(fb, born_k, n, fks_index);
    auto born = _born.build_inverse(
        fb,
        {{"momenta", born_cm}, {"x1", Value(1.)}, {"x2", Value(1.)}},
        born_conditions(fb, conditions, pick(fb, sqrt_s_k, slot))
    );
    auto lumi = _luminosity->build_inverse(
        fb, ValueVec{x1, x2, s_hat, omx1, omx2}, ValueVec{s_born_min}
    );
    auto r = fb.unstack(born.at("random"));
    r.push_back(lumi.at("r1"));
    r.push_back(lumi.at("r2"));
    r.push_back(pick(fb, r_xi_k, slot));
    r.push_back(pick(fb, r_y_k, slot));
    r.push_back(pick(fb, r_phi_k, slot));
    NamedVector<Value> outputs{{"random", fb.stack(r)}};
    if (discrete_dim() > 0) {
        outputs.push_back("discrete", born.at("discrete"));
    }
    return {
        outputs, fb.mul(fb.mul(born.at("det"), lumi.at("det")), pick(fb, det_k, slot))
    };
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

    // into the center-of-mass frame of the Born: that of the incoming momenta
    // for a final-state emitter, of the recoil against i for an incoming one
    Value plus, minus;
    if (_radiation.initial_state() &&
        _radiation.initial_map() == FKSRadiation::asymmetric) {
        auto p = fb.unstack(momenta);
        auto [x1_born, x2_born] = fb.fks_initial_asymmetric_born_fractions(
            p.at(0), p.at(1), p.at(i_fks()), x1, x2, j_fks() == 0 ? 1. : -1.
        );
        std::tie(plus, minus) = born_frame(fb, x1_born, x2_born, conditions);
    } else if (_radiation.initial_state()) {
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

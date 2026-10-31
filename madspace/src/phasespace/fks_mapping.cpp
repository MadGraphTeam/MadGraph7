#include "madspace/phasespace/fks_mapping.hpp"

#include <cmath>
#include <stdexcept>

using namespace madspace;

namespace {

constexpr std::size_t INCOMING_COUNT = 2;

const PhaseSpaceMapping& checked_born(const PhaseSpaceMapping& born) {
    if (born.mirror_beams()) {
        throw std::invalid_argument(
            "FKSMapping: the Born mapping must not mirror the beams, the emission "
            "needs the first incoming parton along +z"
        );
    }
    if (!born.cuts().empty()) {
        throw std::invalid_argument(
            "FKSMapping: the Born mapping must not have cuts, they belong to the "
            "real emission and the counter-events"
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

FKSMapping::FKSMapping(
    const PhaseSpaceMapping& born,
    std::size_t i_fks,
    std::size_t j_fks,
    double s_hat_min
) :
    Mapping(
        "FKSMapping",
        fks_input_types(checked_born(born)),
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

std::pair<Value, Value>
FKSMapping::born_frame(FunctionBuilder& fb, Value x1, Value x2) const {
    // the rapidity log(x1 / x2) / 2 of the Born in the beams' frame, which
    // moves with beam_rapidity in the lab
    double rapidity = _born.beam_rapidity();
    double exp_plus = std::exp(rapidity), exp_minus = std::exp(-rapidity);
    if (_born.leptonic()) {
        return {exp_plus, exp_minus};
    }
    if (rapidity == 0.) {
        return {x1, x2};
    }
    return {fb.mul(x1, exp_plus), fb.mul(x2, exp_minus)};
}

Mapping::Result FKSMapping::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
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

    auto [plus, minus] = born_frame(fb, x1_born, x2_born);
    auto born_cm = born.at("momenta");
    if (boosted) {
        born_cm = fb.fks_beam_boost(born_cm, Value(_born_masses), minus, plus);
    }
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
    auto real = _radiation.build_forward(fb, rad_inputs);
    auto momenta = real.at("momenta");
    if (boosted) {
        momenta = fb.fks_beam_boost(momenta, Value(_real_masses), plus, minus);
    }
    auto x1 = _born.leptonic() ? x1_born : real.at("x1");
    auto x2 = _born.leptonic() ? x2_born : real.at("x2");
    auto det = fb.mul(born.at("det"), real.at("det"));
    return {{{"momenta", momenta}, {"x1", x1}, {"x2", x2}}, det};
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
        std::tie(plus, minus) = born_frame(fb, x1, x2);
    }
    auto real_cm = momenta;
    if (boosted) {
        real_cm = fb.fks_beam_boost(momenta, Value(_real_masses), minus, plus);
    }
    NamedVector<Value> rad_inputs{{"momenta", real_cm}};
    if (!_born.leptonic()) {
        rad_inputs.push_back("x1", x1);
        rad_inputs.push_back("x2", x2);
    }
    auto rad = _radiation.build_inverse(fb, rad_inputs);
    auto x1_born = _born.leptonic() ? x1 : rad.at("x1");
    auto x2_born = _born.leptonic() ? x2 : rad.at("x2");

    auto born_lab = rad.at("born_momenta");
    if (boosted) {
        auto [born_plus, born_minus] = born_frame(fb, x1_born, x2_born);
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

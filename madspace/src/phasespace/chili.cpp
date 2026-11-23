#include "madspace/phasespace/chili.hpp"

#include <stdexcept>

using namespace madspace;

ChiliMapping::ChiliMapping(
    std::size_t n_particles,
    const std::vector<double>& y_max,
    const std::vector<double>& pt_min,
    bool fixed_beams
) :
    Mapping(
        "ChiliMapping",
        [&] {
            if (fixed_beams && n_particles < 2) {
                throw std::invalid_argument(
                    "ChiliMapping: fixed beams need at least two outgoing particles"
                );
            }
            NamedVector<Type> input_types;
            for (std::size_t i = 0; i < 3 * n_particles - (fixed_beams ? 4 : 2); ++i) {
                input_types.push_back(std::format("random{}", i), batch_float);
            }
            return input_types;
        }(),
        [&] {
            NamedVector<Type> output_types;
            for (std::size_t i = 0; i < n_particles + 2; ++i) {
                output_types.push_back(std::format("momentum{}", i), batch_four_vec);
            }
            return output_types;
        }(),
        [&] {
            NamedVector<Type> cond_types{{"com_energy", batch_float}};
            for (std::size_t i = 0; i < n_particles; ++i) {
                cond_types.push_back(std::format("mass{}", i), batch_float);
            }
            return cond_types;
        }()
    ),
    _n_particles(n_particles),
    _y_max(y_max),
    _pt_min(pt_min),
    _fixed_beams(fixed_beams) {}

Mapping::Result ChiliMapping::build_forward_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    Value e_cm = conditions.at(0);
    ValueVec m_out(conditions.begin() + 1, conditions.end());
    auto forward = _fixed_beams
        ? &FunctionBuilder::chili_forward_fixed
        : &FunctionBuilder::chili_forward;
    auto [p_ext, det] = (fb.*forward)(
        fb.stack(inputs.values()), e_cm, fb.stack(m_out), _pt_min, _y_max
    );
    auto outputs = fb.unstack(p_ext);
    return {{output_types().keys(), outputs}, det};
}

Mapping::Result ChiliMapping::build_inverse_impl(
    FunctionBuilder& fb,
    const NamedVector<Value>& inputs,
    const NamedVector<Value>& conditions
) const {
    Value e_cm = conditions.at(0);
    ValueVec m_out(conditions.begin() + 1, conditions.end());
    auto inverse = _fixed_beams
        ? &FunctionBuilder::chili_inverse_fixed
        : &FunctionBuilder::chili_inverse;
    auto [r, det] = (fb.*inverse)(
        fb.stack(inputs.values()), e_cm, fb.stack(m_out), _pt_min, _y_max
    );
    ValueVec r_vec = fb.unstack(r);
    ValueVec outputs;
    outputs.insert(outputs.end(), r_vec.begin(), r_vec.end());
    return {{input_types().keys(), fb.unstack(r)}, det};
}

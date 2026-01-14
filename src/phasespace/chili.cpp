#include "madevent/phasespace/chili.h"

#include <ranges>

using namespace madevent;

ChiliMapping::ChiliMapping(
    std::size_t _n_particles,
    const std::vector<double>& _y_max,
    const std::vector<double>& _pt_min
) :
    Mapping(
        "ChiliMapping",
        TypeVec(4 * _n_particles - 1, batch_float),
        TypeVec(_n_particles + 2, batch_four_vec),
        {}
    ),
    n_particles(_n_particles),
    y_max(_y_max),
    pt_min(_pt_min) {}

Mapping::Result ChiliMapping::build_forward_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    ValueVec r(inputs.begin(), inputs.begin() + 3 * n_particles - 2);
    Value e_cm = inputs.at(3 * n_particles - 2);
    ValueVec m_out(inputs.begin() + 3 * n_particles - 1, inputs.end());
    auto [p_ext, det] =
        fb.chili_forward(fb.stack(r), e_cm, fb.stack(m_out), pt_min, y_max);
    auto outputs = fb.unstack(p_ext);
    return {outputs, det};
}

Mapping::Result ChiliMapping::build_inverse_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    auto [r, e_cm, m_out, det] = fb.chili_inverse(fb.stack(inputs), pt_min, y_max);
    ValueVec r_vec = fb.unstack(r);
    ValueVec m_out_vec = fb.unstack(m_out);
    ValueVec outputs;
    outputs.insert(outputs.end(), r_vec.begin(), r_vec.end());
    outputs.push_back(e_cm);
    outputs.insert(outputs.end(), m_out_vec.begin(), m_out_vec.end());
    return {outputs, det};
}

#include "madspace/phasespace/chili.h"

#include <ranges>

using namespace madspace;

ChiliMapping::ChiliMapping(
    std::size_t n_particles,
    const std::vector<double>& y_max,
    const std::vector<double>& pt_min
) :
    Mapping(
        "ChiliMapping",
        TypeVec(3 * n_particles - 2, batch_float),
        TypeVec(n_particles + 2, batch_four_vec),
        TypeVec(n_particles + 1, batch_float)
    ),
    _n_particles(n_particles),
    _y_max(y_max),
    _pt_min(pt_min) {}

Mapping::Result ChiliMapping::build_forward_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    Value e_cm = conditions.at(0);
    ValueVec m_out(conditions.begin() + 1, conditions.end());
    auto [p_ext, det] =
        fb.chili_forward(fb.stack(inputs), e_cm, fb.stack(m_out), _pt_min, _y_max);
    auto outputs = fb.unstack(p_ext);
    return {outputs, det};
}

Mapping::Result ChiliMapping::build_inverse_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    Value e_cm = conditions.at(0);
    ValueVec m_out(conditions.begin() + 1, conditions.end());
    auto [r, det] =
        fb.chili_inverse(fb.stack(inputs), e_cm, fb.stack(m_out), _pt_min, _y_max);
    ValueVec r_vec = fb.unstack(r);
    ValueVec outputs;
    outputs.insert(outputs.end(), r_vec.begin(), r_vec.end());
    return {fb.unstack(r), det};
}

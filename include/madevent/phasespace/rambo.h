#pragma once

#include "madevent/phasespace/base.h"

namespace madevent {

class FastRamboMapping : public Mapping {
public:
    FastRamboMapping(std::size_t n_particles, bool massless, bool com = true);

    std::size_t random_dim() const { return 3 * _n_particles - 4; }

private:
    Result build_forward_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;

    std::size_t _n_particles;
    bool _massless;
    double _com;
};

} // namespace madevent

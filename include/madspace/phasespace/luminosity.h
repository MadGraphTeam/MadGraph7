#pragma once

#include "madspace/phasespace/base.h"
#include "madspace/phasespace/invariants.h"

namespace madspace {

class Luminosity : public Mapping {
public:
    Luminosity(
        double s_lab,
        double s_hat_min,
        double s_hat_max = 0,
        double invariant_power = 1,
        double mass = 0,
        double width = 0
    ) :
        Mapping(
            "Luminosity",
            {batch_float, batch_float},
            {batch_float, batch_float, batch_float},
            {}
        ),
        _s_lab(s_lab),
        _s_hat_min(s_hat_min),
        _s_hat_max(s_hat_max == 0 ? s_lab : s_hat_max),
        _invariant(invariant_power, mass, width) {}

private:
    Result build_forward_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;

    double _s_lab, _s_hat_min, _s_hat_max;
    Invariant _invariant;
};

} // namespace madspace

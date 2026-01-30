#pragma once

#include "madspace/phasespace/base.h"

namespace madspace {

class Invariant : public Mapping {
public:
    Invariant(double power = 0, double mass = 0, double width = 0) :
        Mapping("Invariant", {batch_float}, {batch_float}, {batch_float, batch_float}),
        _power(power),
        _mass(mass),
        _width(width) {}

private:
    Result build_forward_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
    ) const override;

    double _power, _mass, _width;
};

} // namespace madspace

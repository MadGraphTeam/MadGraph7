#include "madspace/phasespace/rambo.h"

#include <cmath>

#include "madspace/constants.h"

using namespace madspace;

FastRamboMapping::FastRamboMapping(std::size_t n_particles, bool massless, bool com) :
    Mapping(
        "FastRamboMapping",
        [&] {
            TypeVec input_types(3 * n_particles - 4, batch_float);
            if (!com) {
                input_types.push_back(batch_four_vec);
            }
            return input_types;
        }(),
        TypeVec(n_particles, batch_four_vec),
        TypeVec(massless ? 1 : n_particles + 1, batch_float)
    ),
    _n_particles(n_particles),
    _massless(massless),
    _com(com) {
    if (n_particles < 3 || n_particles > 12) {
        throw std::invalid_argument("The number of particles must be between 3 and 12");
    }
}

Mapping::Result FastRamboMapping::build_forward_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    Value r = fb.stack(ValueVec(inputs.begin(), inputs.begin() + random_dim()));
    Value e_cm = conditions.at(0);

    std::array<Value, 2> output;
    if (_massless) {
        if (_com) {
            output = fb.fast_rambo_massless_com(r, e_cm);
        } else {
            Value p0 = inputs.back();
            output = fb.fast_rambo_massless(r, e_cm, p0);
        }
    } else {
        if (_com) {
            ValueVec masses(conditions.begin() + 1, conditions.end());
            output = fb.fast_rambo_massive_com(r, e_cm, fb.stack(masses));
        } else {
            ValueVec masses(conditions.begin() + 1, conditions.end());
            Value p0 = inputs.back();
            output = fb.fast_rambo_massive(r, e_cm, fb.stack(masses), p0);
        }
    }
    return {fb.unstack(output[0]), output[1]};
}

Mapping::Result FastRamboMapping::build_inverse_impl(
    FunctionBuilder& fb, const ValueVec& inputs, const ValueVec& conditions
) const {
    throw std::logic_error("inverse mapping not implemented");
}

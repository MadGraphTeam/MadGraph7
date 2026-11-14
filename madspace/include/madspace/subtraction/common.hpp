#pragma once

#include <utility>

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/phasespace.hpp"

namespace madspace {

/**
 * The light-cone components (up to a common factor) whose ratio is
 * @f$e^{2y}@f$ for the rapidity @f$y@f$ in the lab frame of the
 * center-of-mass frame of the momentum fractions @p x1 and @p x2 of @p born,
 * with its beam rapidity and, with `mirror_beams`, its `mirror_index`
 * condition. The instruction `boost_beam_light_cone` with the arguments `minus`,
 * `plus` boosts the lab frame into that frame, with `plus`, `minus` back.
 *
 * @param fb          The function builder.
 * @param born        The Born mapping.
 * @param x1, x2      The momentum fractions of the first and second incoming
 *                    parton, those of the Born or of a real emission.
 * @param conditions  The conditions of the Born mapping.
 */
std::pair<Value, Value> born_frame_light_cone(
    FunctionBuilder& fb,
    const PhaseSpaceMapping& born,
    Value x1,
    Value x2,
    const NamedVector<Value>& conditions
);

} // namespace madspace

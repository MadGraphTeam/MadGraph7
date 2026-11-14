#include "madspace/subtraction/common.hpp"

#include <cmath>

using namespace madspace;

std::pair<Value, Value> madspace::born_frame_light_cone(
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
    // the momentum fractions of lepton beams or of a decay are one
    if (born.leptonic() || born.is_decay()) {
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

#pragma once

#include "../kinematics.hpp"

namespace madspace {
namespace kernels {

// The parton luminosity of one kinematics of the FKS integrand, that of the
// parton_lum_*.f files of MadFKS.

// The parton luminosity of one kinematics, f_1(x_1) f_2(x_2) of the flavor of
// the event among those of its FKS configuration, or their sum for a negative
// flavor index (PD of parton_lum_*.f, without the conversion to pb), each
// times its factor, from the densities x f(x) of the two beams. flavors holds
// for each configuration the positions in pdf1 and pdf2 of its flavors, in
// pairs, -2 for a flavor without density and -1 after the last, and factors
// the factor of each. A flavor index beyond the flavors of the configuration
// gives zero.
template <typename T>
KERNELSPEC void kernel_fks_luminosity(
    FIn<T, 1> pdf1,
    FIn<T, 1> pdf2,
    FIn<T, 0> x1,
    FIn<T, 0> x2,
    IIn<T, 0> index,
    IIn<T, 0> flavor,
    ISIn<T, 2> flavors,
    FSIn<T, 2> factors,
    FOut<T, 0> luminosity
) {
    IVal<T> picked = flavor;
    FVal<T> f1 = FVal<T>(x1), f2 = FVal<T>(x2);
    for_each_value(IVal<T>(index), [&](std::size_t c, BVal<T> lanes) {
        auto pairs = flavors[c];
        auto weights = factors[c];
        FVal<T> sum = 0.;
        for (std::size_t k = 0; k + 1 < pairs.size(); k += 2) {
            int a = single_index(pairs[k]), b = single_index(pairs[k + 1]);
            if (a == -1) {
                break;
            }
            if (a < 0 || b < 0) {
                // a flavor without density
                continue;
            }
            // the flavor of the event, or all of them
            auto mine = (picked < 0) | (picked == IVal<T>(int(k / 2)));
            FVal<T> term = single_value(weights[k / 2]) * (FVal<T>(pdf1[a]) / f1) *
                (FVal<T>(pdf2[b]) / f2);
            sum = sum + where(mine, term, FVal<T>(0.));
        }
        masked_store(luminosity, lanes, sum);
    });
}

} // namespace kernels
} // namespace madspace

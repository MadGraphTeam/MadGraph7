#pragma once

#include "../../kinematics.hpp"

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
    IIn<T, 2> flavors,
    FIn<T, 2> factors,
    FOut<T, 0> luminosity
) {
    std::size_t c = single_index(index);
    auto pairs = flavors[c];
    auto weights = factors[c];
    int picked = single_index(flavor);
    FVal<T> sum = 0.;
    for (std::size_t k = 0; k + 1 < pairs.size(); k += 2) {
        int a = single_index(pairs[k]), b = single_index(pairs[k + 1]);
        if (a == -1) {
            break;
        }
        if ((picked >= 0 && static_cast<int>(k / 2) != picked) || a < 0 || b < 0) {
            // another flavor, or one without density
            continue;
        }
        sum = sum +
            FVal<T>(weights[k / 2]) * (FVal<T>(pdf1[a]) / FVal<T>(x1)) *
                (FVal<T>(pdf2[b]) / FVal<T>(x2));
    }
    luminosity = sum;
}

} // namespace kernels
} // namespace madspace

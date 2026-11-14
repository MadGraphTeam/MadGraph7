#pragma once

#include "../kinematics.hpp"

namespace madspace {
namespace kernels {

// Functions that the kernels of more than one subtraction scheme share.

// The dilogarithm of a real argument, its real part for x > 1, without
// branches so that the kernels that take it vectorize. Li2(y) for -1 <= y <=
// 1/2 is the series in u = -log(1 - y) with the Bernoulli numbers B_n,
// Li2(y) = sum_n B_n u^(n + 1) / (n + 1)! ('t Hooft and Veltman, Nucl. Phys.
// B153 (1979) 365), which for |u| <= log 2 reaches double precision with the
// terms up to u^19. The other arguments map onto that range by Li2(x) =
// pi^2 / 6 - log(x) log|1 - x| - Li2(1 - x) for 1/2 < x <= 2 and by Li2(x) =
// -pi^2 / 6 - log^2(-x) / 2 - Li2(1 / x) for x < -1, pi^2 / 3 - log^2(x) / 2 -
// Li2(1 / x) for x > 2, the map of each event a select. u is then -log(x) or
// log|x| - log|1 - x|, so that the two logarithms of the maps are all it
// takes, log|1 - x| from log1p(-x) below 1 and from log1p(x - 2) above, which
// keep their digits near x = 0 and x = 1.
template <typename T>
KERNELSPEC FVal<T> dilog(FVal<T> x) {
    constexpr double pi2_6 = 1.6449340668482264365, pi2_3 = 2. * pi2_6;
    // B_n / (n + 1)! of the even n from 2 to 18
    constexpr double b[9] = {
        2.7777777777777777778e-2,
        -2.7777777777777777778e-4,
        4.7241118669690098262e-6,
        -9.1857730746619635509e-8,
        1.8978869988970999072e-9,
        -4.0647616451442255268e-11,
        8.9216910204564525552e-13,
        -1.9939295860721075687e-14,
        4.5189800296199181917e-16,
    };
    FVal<T> l_x = log(fabs(x));
    FVal<T> l_omx = log1p(where(x < 1., -x, x - 2.));
    // Li2(x) = r + Li2(y) on the direct range and r - Li2(y) on the others,
    // y = 1 - exp(-u)
    auto direct = (x >= -1.) & (x <= 0.5);
    auto reflect = (x > 0.5) & (x <= 2.);
    FVal<T> u = where(direct, -l_omx, where(reflect, -l_x, l_x - l_omx));
    FVal<T> r = where(
        direct,
        FVal<T>(0.),
        where(
            reflect,
            pi2_6 - l_x * l_omx,
            where(x < 0., FVal<T>(-pi2_6), FVal<T>(pi2_3)) - 0.5 * l_x * l_x
        )
    );
    FVal<T> u2 = u * u, sum = b[8];
    for (int k = 7; k >= 0; --k) {
        sum = b[k] + u2 * sum;
    }
    FVal<T> li2 = u - 0.25 * u2 + u * u2 * sum;
    // at x = 1 the reflection takes 0 log 0
    return where(x == 1., FVal<T>(pi2_6), r + where(direct, li2, -li2));
}

} // namespace kernels
} // namespace madspace

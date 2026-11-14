#pragma once

#include "../kinematics.hpp"

namespace madspace {
namespace kernels {

// Functions that the kernels of more than one subtraction scheme share.

// The dilogarithm of a real argument as DDILOG of CERNLIB (fks_singular.f),
// with its Chebyshev series
template <typename T>
KERNELSPEC FVal<T> dilog(FVal<T> x) {
    constexpr double pi = 3.14159265358979323846;
    constexpr double pi3 = pi * pi / 3., pi6 = pi * pi / 6., pi12 = pi * pi / 12.;
    constexpr double c[20] = {
        0.42996693560813697,
        0.40975987533077105,
        -0.01858843665014592,
        0.00145751084062268,
        -0.00014304184442340,
        0.00001588415541880,
        -0.00000190784959387,
        0.00000024195180854,
        -0.00000003193341274,
        0.00000000434545063,
        -0.00000000060578480,
        0.00000000008612098,
        -0.00000000001244332,
        0.00000000000182256,
        -0.00000000000027007,
        0.00000000000004042,
        -0.00000000000000610,
        0.00000000000000093,
        -0.00000000000000014,
        0.00000000000000002,
    };
    if (x == 1.) {
        return pi6;
    }
    if (x == -1.) {
        return -pi12;
    }
    FVal<T> t = -x, y, s, a;
    if (t <= -2.) {
        y = -1. / (1. + t);
        s = 1.;
        FVal<T> l1 = log(-t), l2 = log(1. + 1. / t);
        a = -pi3 + 0.5 * (l1 * l1 - l2 * l2);
    } else if (t < -1.) {
        y = -1. - t;
        s = -1.;
        a = log(-t);
        a = -pi6 + a * (a + log(1. + 1. / t));
    } else if (t <= -0.5) {
        y = -(1. + t) / t;
        s = 1.;
        a = log(-t);
        a = -pi6 + a * (-0.5 * a + log(1. + t));
    } else if (t < 0.) {
        y = -t / (1. + t);
        s = -1.;
        FVal<T> l = log(1. + t);
        a = 0.5 * l * l;
    } else if (t <= 1.) {
        y = t;
        s = 1.;
        a = 0.;
    } else {
        y = 1. / t;
        s = -1.;
        FVal<T> l = log(t);
        a = pi6 + 0.5 * l * l;
    }
    FVal<T> h = y + y - 1., alpha = h + h, b0 = 0., b1 = 0., b2 = 0.;
    for (int k = 19; k >= 0; --k) {
        b0 = c[k] + alpha * b1 - b2;
        b2 = b1;
        b1 = b0;
    }
    return -(s * (b0 - h * b2) + a);
}

} // namespace kernels
} // namespace madspace

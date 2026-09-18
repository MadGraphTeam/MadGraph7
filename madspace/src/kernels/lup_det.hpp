#pragma once

#include "definitions.hpp"

namespace madspace {
namespace kernels {

// Local tolerance constant, intentionally tighter than the module-level EPS.
// Chosen so the relative tolerance scale * LU_TOL sits above eps_mach but
// below the noise floor at which the polynomial expansion blows up.
inline constexpr double LU_TOL = 1e-14;

// Variadic max-of-|.| over an arbitrary list of FVal<T>.
template <typename T, typename... Args>
KERNELSPEC FVal<T> max_abs(FVal<T> first, Args... rest) {
    if constexpr (sizeof...(rest) == 0) {
        return fabs(first);
    } else {
        return max(fabs(first), max_abs<T>(rest...));
    }
}

// det of a symmetric 3x3 matrix via LU with partial pivoting.
//
//     [ a11  a12  a13 ]
//     [ a12  a22  a23 ]
//     [ a13  a23  a33 ]
//
// Falls back lane-wise to `polynomial_fallback` if any pivot is below
// LU_TOL * max(|entries|).
template <typename T>
KERNELSPEC FVal<T> lup_det3(
    FVal<T> a11,
    FVal<T> a12,
    FVal<T> a13,
    FVal<T> a22,
    FVal<T> a23,
    FVal<T> a33,
    FVal<T> polynomial_fallback
) {
    // Row-major working copy.
    auto r00 = a11, r01 = a12, r02 = a13;
    auto r10 = a12, r11 = a22, r12 = a23;
    auto r20 = a13, r21 = a23, r22 = a33;

    auto scale = max_abs<T>(a11, a12, a13, a22, a23, a33);
    auto tol = LU_TOL * scale;
    auto sign = FVal<T>(1.0);
    auto degenerate = (scale == 0.);

    // ---- Step 0: pivot from column 0, rows {0,1,2} ----
    {
        auto m0 = fabs(r00), m1 = fabs(r10), m2 = fabs(r20);
        auto best = IVal<T>(0);
        auto best_mag = m0;
        auto pick1 = (m1 > best_mag);
        best = where(pick1, IVal<T>(1), best);
        best_mag = where(pick1, m1, best_mag);
        auto pick2 = (m2 > best_mag);
        best = where(pick2, IVal<T>(2), best);
        best_mag = where(pick2, m2, best_mag);

        auto sel1 = (best == IVal<T>(1));
        auto sel2 = (best == IVal<T>(2));
        auto any_swap = sel1 | sel2;

        auto n00 = where(sel1, r10, where(sel2, r20, r00));
        auto n01 = where(sel1, r11, where(sel2, r21, r01));
        auto n02 = where(sel1, r12, where(sel2, r22, r02));
        auto n10 = where(sel1, r00, r10);
        auto n11 = where(sel1, r01, r11);
        auto n12 = where(sel1, r02, r12);
        auto n20 = where(sel2, r00, r20);
        auto n21 = where(sel2, r01, r21);
        auto n22 = where(sel2, r02, r22);
        r00 = n00;
        r01 = n01;
        r02 = n02;
        r10 = n10;
        r11 = n11;
        r12 = n12;
        r20 = n20;
        r21 = n21;
        r22 = n22;

        sign = where(any_swap, -sign, sign);
        degenerate = degenerate | (best_mag < tol);

        auto pivot_safe = where(degenerate, FVal<T>(1.0), r00);
        auto f10 = r10 / pivot_safe;
        auto f20 = r20 / pivot_safe;
        r10 = f10;
        r11 = r11 - f10 * r01;
        r12 = r12 - f10 * r02;
        r20 = f20;
        r21 = r21 - f20 * r01;
        r22 = r22 - f20 * r02;
    }

    // ---- Step 1: pivot from column 1, rows {1, 2} ----
    {
        auto m1 = fabs(r11), m2 = fabs(r21);
        auto sel2 = (m2 > m1);
        auto pivot_mag = where(sel2, m2, m1);

        auto n11 = where(sel2, r21, r11);
        auto n12 = where(sel2, r22, r12);
        auto n21 = where(sel2, r11, r21);
        auto n22 = where(sel2, r12, r22);
        r11 = n11;
        r12 = n12;
        r21 = n21;
        r22 = n22;
        // Multipliers in column 0 follow the row swap (preserves L, not
        // strictly needed for the determinant but kept for correctness).
        auto n10 = where(sel2, r20, r10);
        auto n20 = where(sel2, r10, r20);
        r10 = n10;
        r20 = n20;

        sign = where(sel2, -sign, sign);
        degenerate = degenerate | (pivot_mag < tol);

        auto pivot_safe = where(degenerate, FVal<T>(1.0), r11);
        auto f21 = r21 / pivot_safe;
        r21 = f21;
        r22 = r22 - f21 * r12;
    }

    // ---- Final pivot check: the trailing diagonal entry is also a pivot ----
    degenerate = degenerate | (fabs(r22) < tol);

    auto lu_det = sign * r00 * r11 * r22;
    return where(degenerate, polynomial_fallback, lu_det);
}

// det of a general (non-symmetric) 3x3 matrix via LU with partial pivoting.
//
//     [ a11  a12  a13 ]
//     [ a21  a22  a23 ]
//     [ a31  a32  a33 ]
//
// Same fallback policy as lup_det3. Needed for bk_V, whose matrix has
// a21 = a12 but a13 != a31 and a23 != a32 (and a22 = a33 = 0).
template <typename T>
KERNELSPEC FVal<T> lup_det3_general(
    FVal<T> a11,
    FVal<T> a12,
    FVal<T> a13,
    FVal<T> a21,
    FVal<T> a22,
    FVal<T> a23,
    FVal<T> a31,
    FVal<T> a32,
    FVal<T> a33,
    FVal<T> polynomial_fallback
) {
    auto r00 = a11, r01 = a12, r02 = a13;
    auto r10 = a21, r11 = a22, r12 = a23;
    auto r20 = a31, r21 = a32, r22 = a33;

    auto scale = max_abs<T>(a11, a12, a13, a21, a22, a23, a31, a32, a33);
    auto tol = LU_TOL * scale;
    auto sign = FVal<T>(1.0);
    auto degenerate = (scale == 0.);

    // ---- Step 0: pivot from column 0, rows {0,1,2} ----
    {
        auto m0 = fabs(r00), m1 = fabs(r10), m2 = fabs(r20);
        auto best = IVal<T>(0);
        auto best_mag = m0;
        auto pick1 = (m1 > best_mag);
        best = where(pick1, IVal<T>(1), best);
        best_mag = where(pick1, m1, best_mag);
        auto pick2 = (m2 > best_mag);
        best = where(pick2, IVal<T>(2), best);
        best_mag = where(pick2, m2, best_mag);

        auto sel1 = (best == IVal<T>(1));
        auto sel2 = (best == IVal<T>(2));
        auto any_swap = sel1 | sel2;

        auto n00 = where(sel1, r10, where(sel2, r20, r00));
        auto n01 = where(sel1, r11, where(sel2, r21, r01));
        auto n02 = where(sel1, r12, where(sel2, r22, r02));
        auto n10 = where(sel1, r00, r10);
        auto n11 = where(sel1, r01, r11);
        auto n12 = where(sel1, r02, r12);
        auto n20 = where(sel2, r00, r20);
        auto n21 = where(sel2, r01, r21);
        auto n22 = where(sel2, r02, r22);
        r00 = n00;
        r01 = n01;
        r02 = n02;
        r10 = n10;
        r11 = n11;
        r12 = n12;
        r20 = n20;
        r21 = n21;
        r22 = n22;

        sign = where(any_swap, -sign, sign);
        degenerate = degenerate | (best_mag < tol);

        auto pivot_safe = where(degenerate, FVal<T>(1.0), r00);
        auto f10 = r10 / pivot_safe;
        auto f20 = r20 / pivot_safe;
        r10 = f10;
        r11 = r11 - f10 * r01;
        r12 = r12 - f10 * r02;
        r20 = f20;
        r21 = r21 - f20 * r01;
        r22 = r22 - f20 * r02;
    }

    // ---- Step 1: pivot from column 1, rows {1, 2} ----
    {
        auto m1 = fabs(r11), m2 = fabs(r21);
        auto sel2 = (m2 > m1);
        auto pivot_mag = where(sel2, m2, m1);

        auto n11 = where(sel2, r21, r11);
        auto n12 = where(sel2, r22, r12);
        auto n21 = where(sel2, r11, r21);
        auto n22 = where(sel2, r12, r22);
        r11 = n11;
        r12 = n12;
        r21 = n21;
        r22 = n22;
        auto n10 = where(sel2, r20, r10);
        auto n20 = where(sel2, r10, r20);
        r10 = n10;
        r20 = n20;

        sign = where(sel2, -sign, sign);
        degenerate = degenerate | (pivot_mag < tol);

        auto pivot_safe = where(degenerate, FVal<T>(1.0), r11);
        auto f21 = r21 / pivot_safe;
        r21 = f21;
        r22 = r22 - f21 * r12;
    }

    degenerate = degenerate | (fabs(r22) < tol);

    auto lu_det = sign * r00 * r11 * r22;
    return where(degenerate, polynomial_fallback, lu_det);
}

} // namespace kernels
} // namespace madspace

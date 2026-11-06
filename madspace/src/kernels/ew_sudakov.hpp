#pragma once

#include "kinematics.hpp"

namespace madspace {
namespace kernels {

// The Sudakov approximation of the EW corrections of ewsudakov_wrapper.f and
// ewsudakov_functions.f of MadGraph (arXiv:hep-ph/0010201,
// arXiv:2110.03714). The coefficients of the matrix elements are linear
// combinations of the logarithms L = a log^2(s / M_W^2), l = a log(s / M_W^2)
// and, for every pair (i, j) with j < i, F_ij and G_ij, whose factors come in
// the tables (see EWSudakov).

template <typename T>
KERNELSPEC void kernel_ew_sudakov(
    FIn<T, 2> momenta,
    FIn<T, 1> diagonal,
    FIn<T, 1> lo2,
    FIn<T, 2> derivatives,
    FIn<T, 2> interference,
    FIn<T, 0> alpha_s,
    FIn<T, 0> ren_scale,
    FIn<T, 1> signs,
    IIn<T, 2> modes,
    FIn<T, 3> diagonal_coefficients,
    FIn<T, 3> interference_coefficients,
    FIn<T, 1> par_coefficients,
    FIn<T, 1> qcd_coefficients,
    FIn<T, 1> options,
    FOut<T, 1> sudakov
) {
    constexpr double pi = 3.14159265358979323846;
    FVal<T> mw2 = options[0], mz2 = options[1], mt2 = options[2], a = options[3];
    std::size_t n = momenta.size();
    std::size_t pairs = n * (n - 1) / 2;
    std::size_t helicities = diagonal.size(), terms = interference.size();
    FVal<T> log_wz = log(mw2 / mz2);

    // the parameter renormalisation per unit l and the LO2 summed over the
    // helicities, the same in every mode
    FVal<T> par = 0., lo2_sum = 0.;
    for (std::size_t h = 0; h < helicities; ++h) {
        for (std::size_t k = 0; k < 6; ++k) {
            par = par + FVal<T>(par_coefficients[k]) * FVal<T>(derivatives[h][k]);
        }
        lo2_sum = lo2_sum + FVal<T>(lo2[h]);
    }

    for (std::size_t m = 0; m < sudakov.size(); ++m) {
        bool s_to_rij = int(modes[m][0]) != 0, rij_ge_mw = int(modes[m][1]) != 0;
        // the invariants of the pairs as sdk_get_invariants, (p_i + p_j)^2
        // with -p_j for one incoming
        auto invariant = [&](std::size_t i, std::size_t j) {
            auto p_i = load_mom<T>(momenta[i]), p_j = load_mom<T>(momenta[j]);
            FVal<T> sign = FVal<T>(signs[i]) * FVal<T>(signs[j]);
            FourMom<T> q{
                p_i[0] + sign * p_j[0],
                p_i[1] + sign * p_j[1],
                p_i[2] + sign * p_j[2],
                p_i[3] + sign * p_j[3]
            };
            FVal<T> r = lsquare<T>(q);
            if (rij_ge_mw) {
                r = where(fabs(r) < mw2, where(r < 0., -mw2, mw2), r);
            }
            return r;
        };
        FVal<T> s = invariant(1, 0);
        FVal<T> log_s = log(s / mw2);
        FVal<T> big_l = a * log_s * log_s, small_l = a * log_s;

        // the logarithms, real and imaginary parts: L, l, then F and G of
        // every pair
        std::size_t size = 2 + 2 * pairs;
        constexpr std::size_t max_size = 2 + 2 * 45;
        FVal<T> basis_re[max_size], basis_im[max_size];
        basis_re[0] = big_l;
        basis_im[0] = 0.;
        basis_re[1] = small_l;
        basis_im[1] = 0.;
        std::size_t k = 2;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < i; ++j, k += 2) {
                FVal<T> r = invariant(i, j);
                FVal<T> l_r = log(fabs(r) / s);
                // the imaginary part of the logarithm, -pi for r > 0
                FVal<T> im = where(r > 0., FVal<T>(-pi), FVal<T>(0.));
                basis_re[k] = 2. * small_l * l_r;
                basis_im[k] = 2. * small_l * im;
                basis_re[k + 1] = 0.;
                basis_im[k + 1] = 0.;
                if (s_to_rij) {
                    basis_re[k] = basis_re[k] + a * l_r * l_r;
                    basis_im[k] = basis_im[k] + 2. * a * l_r * im;
                    basis_re[k + 1] = 2. * a * l_r * log_wz;
                }
            }
        }

        // 2 Re of the coefficients times the matrix elements
        FVal<T> total = 0.;
        for (std::size_t h = 0; h < helicities; ++h) {
            FVal<T> re = 0.;
            for (std::size_t b = 0; b < size; ++b) {
                FVal<T> c_re = diagonal_coefficients[m][h][2 * b];
                FVal<T> c_im = diagonal_coefficients[m][h][2 * b + 1];
                re = re + c_re * basis_re[b] - c_im * basis_im[b];
            }
            total = total + 2. * re * FVal<T>(diagonal[h]);
        }
        for (std::size_t t = 0; t < terms; ++t) {
            FVal<T> re = 0., im = 0.;
            for (std::size_t b = 0; b < size; ++b) {
                FVal<T> c_re = interference_coefficients[m][t][2 * b];
                FVal<T> c_im = interference_coefficients[m][t][2 * b + 1];
                re = re + c_re * basis_re[b] - c_im * basis_im[b];
                im = im + c_re * basis_im[b] + c_im * basis_re[b];
            }
            total = total +
                2. *
                    (re * FVal<T>(interference[t][0]) -
                     im * FVal<T>(interference[t][1]));
        }
        // the parameter renormalisation and the QCD part on top of the LO2
        FVal<T> mu_r = ren_scale;
        FVal<T> qcd = FVal<T>(alpha_s) / (2. * pi) *
            (FVal<T>(qcd_coefficients[0]) * log(mu_r * mu_r / mt2) +
             FVal<T>(qcd_coefficients[1]) * log(s / mt2));
        sudakov[m] = total + small_l * par + 2. * qcd * lo2_sum;
    }
}

} // namespace kernels
} // namespace madspace

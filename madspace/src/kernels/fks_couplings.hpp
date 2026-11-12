#pragma once

#include "kinematics.hpp"

namespace madspace {
namespace kernels {

// The scales and the parton luminosity of one kinematics of the FKS integrand,
// those of setscales.f and of the parton_lum_*.f files of MadFKS.

// The renormalization and factorization scales of set_ren_scale and
// set_fac_scale of setscales.f. The reference scale of scale_global_reference
// (dynamical_scale_choice 1: sum of the transverse energies E pT / |p| of the
// outgoing particles in the partonic center-of-mass frame, 2: sum of their
// transverse masses sqrt((E + pz) (E - pz)), 3 or -1: half of it, -2: the fixed
// mur_ref_fixed) is raised to the floor (minscaleR and minscaleF, 2 GeV). A
// fixed scale replaces it, and each scale is multiplied by its factor
// (mur_over_ref, muf1_over_ref, muf2_over_ref), fixed or not. For a decay (one
// incoming particle) the reference scale is the mass of the decaying
// particle, as setscales.f takes it for mu_R and the Ellis-Sexton scale.
template <typename T>
KERNELSPEC void kernel_fks_scale(
    FIn<T, 2> p,
    FIn<T, 1> options,
    IIn<T, 0> incoming,
    FOut<T, 0> ren_scale,
    FOut<T, 0> fact_scale1,
    FOut<T, 0> fact_scale2
) {
    int choice = static_cast<int>(FVal<T>(options[0]));
    bool fixed_ren = FVal<T>(options[1]) != 0., fixed_fac = FVal<T>(options[2]) != 0.;
    FVal<T> mur_ref_fixed = options[3], muf1_ref_fixed = options[4],
            muf2_ref_fixed = options[5], mur_over_ref = options[6],
            muf1_over_ref = options[7], muf2_over_ref = options[8], floor = options[9];
    std::size_t n = p.size();
    FVal<T> reference = 0.;
    if (single_index(incoming) == 1) {
        auto q = load_mom<T>(p[0]);
        reference = sqrt(max(lsquare<T>(q), 0.));
    } else if (choice == 1) {
        // the boost to the partonic center-of-mass frame
        FVal<T> e_tot = FVal<T>(p[0][0]) + FVal<T>(p[1][0]),
                pz_tot = FVal<T>(p[0][3]) + FVal<T>(p[1][3]);
        FVal<T> rs = sqrt((e_tot + pz_tot) * (e_tot - pz_tot));
        FVal<T> gamma = e_tot / rs, beta_gamma = pz_tot / rs;
        for (std::size_t i = 2; i < n; ++i) {
            auto q = load_mom<T>(p[i]);
            FVal<T> e = gamma * q[0] - beta_gamma * q[3];
            FVal<T> pz = gamma * q[3] - beta_gamma * q[0];
            FVal<T> pt = sqrt(q[1] * q[1] + q[2] * q[2]);
            if (pt > 0.) {
                reference = reference + e * pt / sqrt(pt * pt + pz * pz);
            }
        }
    } else if (choice == 2 || choice == 3 || choice == -1) {
        for (std::size_t i = 2; i < n; ++i) {
            auto q = load_mom<T>(p[i]);
            reference = reference + sqrt(max((q[0] + q[3]) * (q[0] - q[3]), 0.));
        }
        if (choice != 2) {
            reference = reference / 2.;
        }
    } else {
        reference = mur_ref_fixed;
    }
    FVal<T> dynamical = max(floor, reference);
    ren_scale = mur_over_ref * (fixed_ren ? mur_ref_fixed : dynamical);
    fact_scale1 = muf1_over_ref * (fixed_fac ? muf1_ref_fixed : dynamical);
    fact_scale2 = muf2_over_ref * (fixed_fac ? muf2_ref_fixed : dynamical);
}

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

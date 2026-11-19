#pragma once

#include "definitions.hpp"

// The Effective Vector boson Approximation (EVA): the densities of a W, Z or
// photon of helicity -1, 0 or +1 in an electron or muon, at leading log (LLA),
// full leading power (LP) or next-to-leading power (NLP). A line-by-line port
// of madevent's Template/LO/Source/PDF/ElectroweakFlux.f (R. Ruiz,
// arXiv:2111.02442, Bigaran & Ruiz arXiv:2502.07878) and of the checks of
// ElectroweakFluxDriver.f; the couplings of a (vector, parent lepton, beam)
// combination are computed on the host by EvaFlux::parameters, one row of
// eva_param_count numbers laid out as below. See @ref EvaFlux.

namespace madspace {
namespace kernels {

// layout of a parameter row (EvaFlux::parameters)
inline constexpr int eva_kind = 0;      // 0: no density, 1: W/Z, 2: photon
inline constexpr int eva_gg2 = 1;       // squared universal coupling
inline constexpr int eva_gl2 = 2;       // squared left chiral coupling of the parent
inline constexpr int eva_gr2 = 3;       // squared right chiral coupling of the parent
inline constexpr int eva_flpol = 4;     // left-handed fraction of the parent
inline constexpr int eva_mv2 = 5;       // vector mass^2 (photon: parent mass^2)
inline constexpr int eva_mu2min = 6;    // floor of mu^2, before the (1-x) of ievo
inline constexpr int eva_xmin = 7;      // densities vanish below (eva_xcut)
inline constexpr int eva_ebeam = 8;     // parent energy
inline constexpr int eva_order = 9;     // 0 LLA, 1 LP, 2 NLP (photon: 0)
inline constexpr int eva_evolution = 10; // 0: evolution in q^2, 1: in pt^2
inline constexpr int eva_param_count = 11;
// a helicity outside -1, 0, 1 asks for the sum over the vector's helicities
inline constexpr int eva_helicity_sum = 2;

// eva_fL_to_vp (with x the momentum fraction, mu2 the evolution scale squared)
template <typename F>
KERNELSPEC F
eva_fl_to_vp(F gg2, F gl2, F mv2, F x, F mu2, F ebeam, int ievo, int order) {
    const F four_pi_sq = 39.47841760435743;
    F coup2 = gg2 * gl2 / four_pi_sq;
    F split = (1. - x) / 2. / x;
    F ev2 = (x * ebeam) * (x * ebeam);
    F mv_o_ev = mv2 / ev2;
    F mu_o2ev = mu2 / ev2 / 2.;
    F mv_o_mu = mv2 / mu2;
    F mu_o_mumv = 1. / (1. + mv_o_mu);
    F mumv_o_mv = (mu2 / mv2) + 1.;
    F xxlog = log(mumv_o_mv);
    F xlp1 = 1. - x;
    F xlp2 = xlp1;
    F nlp1 = (2. - x) * mv_o_ev;
    F nlp2 = nlp1 + (2. - x) * mu_o2ev;
    F tmp;
    if (order == 2) {
        tmp = xxlog * (xlp1 + nlp1) - mu_o_mumv * (xlp2 + nlp2);
    } else if (order == 1) {
        tmp = xxlog * xlp1 - mu_o_mumv * xlp2;
    } else {
        xxlog = ievo == 0 ? log(mu2 / mv2) : log(mu2 / mv2) - log(1. - x);
        tmp = xlp1 * xxlog;
    }
    return coup2 * split * tmp;
}

// eva_fL_to_vm
template <typename F>
KERNELSPEC F
eva_fl_to_vm(F gg2, F gl2, F mv2, F x, F mu2, F ebeam, int ievo, int order) {
    const F four_pi_sq = 39.47841760435743;
    F coup2 = gg2 * gl2 / four_pi_sq;
    F split = 1. / 2. / x;
    F ev2 = (x * ebeam) * (x * ebeam);
    F mv_o_ev = mv2 / ev2;
    F mu_o2ev = mu2 / ev2 / 2.;
    F mv_o_mu = mv2 / mu2;
    F mu_o_mumv = 1. / (1. + mv_o_mu);
    F mumv_o_mv = (mu2 / mv2) + 1.;
    F xxlog = log(mumv_o_mv);
    F xlp1 = 1.;
    F xlp2 = xlp1;
    F nlp1 = (2. - x) * mv_o_ev;
    F nlp2 = nlp1 + (2. - x) * mu_o2ev;
    F tmp;
    if (order == 2) {
        tmp = xxlog * (xlp1 + nlp1) - mu_o_mumv * (xlp2 + nlp2);
    } else if (order == 1) {
        tmp = xxlog * xlp1 - mu_o_mumv * xlp2;
    } else {
        xxlog = ievo == 0 ? log(mu2 / mv2) : log(mu2 / mv2) - log(1. - x);
        tmp = xlp1 * xxlog;
    }
    return coup2 * split * tmp;
}

// eva_fL_to_v0. At NLP the helicity-flip term is subtracted as madevent does,
// outside of the coupling and splitting factor (its hflip already carry them).
template <typename F>
KERNELSPEC F
eva_fl_to_v0(F gg2, F gl2, F mv2, F x, F mu2, F ebeam, int ievo, int order) {
    const F four_pi_sq = 39.47841760435743;
    F coup2 = gg2 * gl2 / four_pi_sq;
    F split = (1. - x) / x;
    F ev2 = (x * ebeam) * (x * ebeam);
    F mv_o_ev = mv2 / ev2;
    F mv_o_mu = mv2 / mu2;
    F mu_o_mumv = 1. / (1. + mv_o_mu);
    F tmp1, tmp2;
    if (order == 2) {
        F hflip = eva_fl_to_vm<F>(gg2, gl2, mv2, x, mu2, ebeam, ievo, 1) +
            eva_fl_to_vp<F>(gg2, gl2, mv2, x, mu2, ebeam, ievo, 1);
        tmp1 = mu_o_mumv;
        tmp2 = 0.5 * mv_o_ev * hflip;
    } else if (order == 1) {
        tmp1 = mu_o_mumv;
        tmp2 = 0.;
    } else {
        tmp1 = 1.;
        tmp2 = 0.;
    }
    return coup2 * split * tmp1 - tmp2;
}

// f(x) (not x f(x)) of helicity `helicity` (-1, 0, 1, or eva_helicity_sum)
// from the parameter row `p` (eva_get_pdf_by_PID with its checks). `p` is
// anything indexable by an int giving an F.
template <typename F, typename P>
KERNELSPEC F eva_density_value(F x, F q, int helicity, const P& p) {
    int kind = static_cast<int>(p[eva_kind]);
    if (kind == 0 || !(x >= 1e-8 && x <= 1. - 1e-8)) {
        return 0.;
    }
    if (kind == 2 && helicity == 0) {
        return 0.; // a photon has no longitudinal polarisation
    }
    if (helicity < -1 || helicity > 1) {
        F sum = eva_density_value<F, P>(x, q, -1, p) +
            eva_density_value<F, P>(x, q, 1, p);
        if (kind == 1) {
            sum = sum + eva_density_value<F, P>(x, q, 0, p);
        }
        return sum;
    }
    if (x < static_cast<F>(p[eva_xmin])) {
        return 0.;
    }
    int ievo = static_cast<int>(p[eva_evolution]);
    int order = static_cast<int>(p[eva_order]);
    F mu2 = q * q;
    F mu2min = p[eva_mu2min];
    if (ievo != 0) {
        mu2min = (1. - x) * mu2min;
    }
    if (mu2 < mu2min) {
        mu2 = mu2min;
    }
    F gg2 = p[eva_gg2], gl2 = p[eva_gl2], gr2 = p[eva_gr2];
    F fl = p[eva_flpol], mv2 = p[eva_mv2], ebeam = p[eva_ebeam];
    // eva_fX_to_vX: fLpol f_L + (1 - fLpol) f_R, with f_R > v_+- = f_L > v_-+
    if (helicity == 1) {
        return fl * eva_fl_to_vp<F>(gg2, gl2, mv2, x, mu2, ebeam, ievo, order) +
            (1. - fl) * eva_fl_to_vm<F>(gg2, gr2, mv2, x, mu2, ebeam, ievo, order);
    } else if (helicity == -1) {
        return fl * eva_fl_to_vm<F>(gg2, gl2, mv2, x, mu2, ebeam, ievo, order) +
            (1. - fl) * eva_fl_to_vp<F>(gg2, gr2, mv2, x, mu2, ebeam, ievo, order);
    } else {
        return fl * eva_fl_to_v0<F>(gg2, gl2, mv2, x, mu2, ebeam, ievo, order) +
            (1. - fl) * eva_fl_to_v0<F>(gg2, gr2, mv2, x, mu2, ebeam, ievo, order);
    }
}

// x f_helicity(x, q) of the configuration `config` (a row of `params`), zero
// for a configuration < 0
template <typename T>
KERNELSPEC void kernel_eva_density(
    FIn<T, 0> x,
    FIn<T, 0> q,
    IIn<T, 0> helicity,
    IIn<T, 0> config,
    FIn<T, 2> params,
    FOut<T, 0> pdf
) {
    using F = FVal<T>;
    int cfg = static_cast<int>(config);
    if (cfg < 0 || cfg >= static_cast<int>(params.size())) {
        pdf = 0.;
        return;
    }
    auto row = params[cfg];
    F x_val = x;
    pdf = x_val *
        eva_density_value<F>(x_val, static_cast<F>(q), static_cast<int>(helicity), row);
}

// The density of one incoming leg of an event, after its matrix element picked
// a helicity row: x n f_lambda(x, q), with lambda the leg's helicity in row
// `helicity` of the event's flavour `flavor`, n the leg's number of helicity
// states (the matrix element's averaging factor) and f the density of the beam
// the leg comes from (`mirror` = 1: the other one). A row out of range gives
// the helicity-summed density x sum_lambda f_lambda without n.
template <typename T>
KERNELSPEC void kernel_eva_leg_density(
    FIn<T, 0> x,
    FIn<T, 0> q,
    IIn<T, 0> flavor,
    IIn<T, 0> helicity,
    IIn<T, 0> mirror,
    IIn<T, 1> row_offsets,
    IIn<T, 1> row_counts,
    IIn<T, 1> leg_helicities,
    IIn<T, 1> configs,
    FIn<T, 1> helicity_counts,
    FIn<T, 2> params,
    FOut<T, 0> pdf
) {
    using F = FVal<T>;
    int flav = static_cast<int>(flavor);
    if (flav < 0 || flav >= static_cast<int>(row_counts.size())) {
        pdf = 0.;
        return;
    }
    int cfg = static_cast<int>(configs[2 * flav + (static_cast<int>(mirror) != 0)]);
    if (cfg < 0 || cfg >= static_cast<int>(params.size())) {
        pdf = 0.;
        return;
    }
    auto row = params[cfg];
    int hel_row = static_cast<int>(helicity);
    F x_val = x;
    F q_val = q;
    if (hel_row < 0 || hel_row >= static_cast<int>(row_counts[flav])) {
        pdf = x_val * eva_density_value<F>(x_val, q_val, eva_helicity_sum, row);
        return;
    }
    int offset = static_cast<int>(row_offsets[flav]);
    int lambda = static_cast<int>(leg_helicities[offset + hel_row]);
    F count = helicity_counts[flav];
    pdf = x_val * count * eva_density_value<F>(x_val, q_val, lambda, row);
}

} // namespace kernels
} // namespace madspace

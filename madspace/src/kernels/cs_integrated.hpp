#pragma once

#include "cs.hpp"
#include "fks_subtraction.hpp"

namespace madspace {
namespace kernels {

// The integrated Catani-Seymour dipoles (arXiv:hep-ph/9605323, Sec. 10;
// masses: arXiv:hep-ph/0201036, Sec. 6) in conventional dimensional
// regularization, with the normalization (4 pi)^eps / Gamma(1 - eps) of the
// virtual of MadLoop. The alpha terms are what the gates of the dipoles of
// cs.hpp remove from them at alpha = 1, in four dimensions, per unit
// <T_j . T_k> / T_j^2 and alpha_s / (2 pi).

constexpr double CS_PI2 = 3.14159265358979323846 * 3.14159265358979323846;

// V^(S)(s, m_j, m_k) of CDST eqs. 6.20 and 6.21, its coefficients of eps^-2,
// eps^-1 and eps^0
template <typename T>
KERNELSPEC void cs_v_singular(FVal<T> s, FVal<T> mj, FVal<T> mk, FVal<T> (&v_s)[3]) {
    FVal<T> mj2 = mj * mj, mk2 = mk * mk, q2 = s + mj2 + mk2;
    if (mj == 0. && mk == 0.) {
        v_s[0] = 1.;
        v_s[1] = 0.;
        v_s[2] = 0.;
    } else if (mj > 0. && mk > 0.) {
        // the velocity v_jk and 1 - v_jk, rho^2 = (1 - v) / (1 + v) and the
        // rho_n^2 of eq. 5.30
        FVal<T> d = 4. * mj2 * mk2 / (s * s);
        FVal<T> v = sqrt(1. - d), omv = d / (1. + v);
        FVal<T> l_rho2 = log(omv / (1. + v));
        FVal<T> l_rhoj2 = log((omv + 2. * mj2 / s) / (1. + v + 2. * mj2 / s));
        FVal<T> l_rhok2 = log((omv + 2. * mk2 / s) / (1. + v + 2. * mk2 / s));
        v_s[0] = 0.;
        v_s[1] = 0.5 * l_rho2 / v;
        v_s[2] = (-0.25 * (l_rhoj2 * l_rhoj2 + l_rhok2 * l_rhok2) - CS_PI2 / 6.) / v +
            0.5 * l_rho2 * log(q2 / s) / v;
    } else {
        FVal<T> m2 = mj > 0. ? mj2 : mk2;
        FVal<T> l_ms = log(m2 / s), l_sq = log(s / q2);
        v_s[0] = 0.5;
        v_s[1] = 0.5 * l_ms;
        v_s[2] = -0.25 * l_ms * l_ms - CS_PI2 / 12. - 0.5 * l_ms * l_sq -
            0.5 * log(m2 / q2) * l_sq;
    }
}

// V^(NS)_j(s, m_j, m_k; kappa) of CDST eqs. 6.22 to 6.26 for a quark or a
// massless gluon j with the gamma_j / T_j^2 of j, without the splitting of
// a gluon into heavy quarks (they decouple from the virtual)
template <typename T>
KERNELSPEC FVal<T> cs_v_nonsingular(
    bool gluon,
    FVal<T> s,
    FVal<T> mj,
    FVal<T> mk,
    FVal<T> gamma_over_t2,
    FVal<T> kappa,
    FVal<T> n_f
) {
    FVal<T> mj2 = mj * mj, mk2 = mk * mk, q2 = s + mj2 + mk2, q = sqrt(q2);
    FVal<T> l_sq = log(s / q2);
    if (mk > 0. && mj == 0.) {
        FVal<T> value =
            gamma_over_t2 * (l_sq - 2. * log((q - mk) / q) - 2. * mk / (q + mk)) +
            CS_PI2 / 6. - fks_dilog<T>(s / q2);
        if (gluon) {
            value = value +
                (kappa - 2. / 3.) * mk2 / s * (2. * CS_TR * n_f / CS_CA - 1.) *
                    log(2. * mk / (q + mk));
        }
        return value;
    }
    if (gluon || mj == 0.) {
        return 0.;
    }
    if (mk == 0.) {
        return gamma_over_t2 * l_sq + CS_PI2 / 6. - fks_dilog<T>(s / q2) - 2. * l_sq -
            mj2 / s * log(mj2 / q2);
    }
    FVal<T> d = 4. * mj2 * mk2 / (s * s);
    FVal<T> v = sqrt(1. - d), omv = d / (1. + v);
    FVal<T> rho2 = omv / (1. + v);
    FVal<T> rhoj2 = (omv + 2. * mj2 / s) / (1. + v + 2. * mj2 / s);
    FVal<T> rhok2 = (omv + 2. * mk2 / s) / (1. + v + 2. * mk2 / s);
    FVal<T> qmk = q - mk;
    return gamma_over_t2 * l_sq +
        (log(rho2) * log(1. + rho2) + 2. * fks_dilog<T>(rho2) -
         fks_dilog<T>(1. - rhoj2) - fks_dilog<T>(1. - rhok2) - CS_PI2 / 6.) /
        v +
        log(qmk / q) - 2. * log((qmk * qmk - mj2) / q2) - 2. * mj2 / s * log(mj / qmk) -
        mk / qmk + 2. * mk * (2. * mk - q) / s + CS_PI2 / 2.;
}

// What the gate y < alpha y_+ of the final-final dipoles of the emitter j
// with the spectator k removes from their integral at alpha = 1: the
// dipoles over alpha y_+ < y < y_+ in four dimensions, per unit <T_j . T_k> /
// T_j^2 and alpha_s / (2 pi), of q -> q g for a quark j and of g -> g g (with
// the symmetry factor 1/2) and n_f g -> q q~ for a gluon j. The z integral is
// done in closed form, the y integral on 32 Gauss-Legendre nodes in t with
// y = alpha y_+ (1 / alpha)^(1 - (1 - t)^2), which keeps the sqrt(y_+ - y) of
// the velocity of a massive spectator out of the integrand.
template <typename T>
KERNELSPEC FVal<T> cs_removed_final_final(
    bool gluon,
    FVal<T> s,
    FVal<T> mj,
    FVal<T> mk,
    FVal<T> alpha,
    FVal<T> kappa,
    FVal<T> n_f
) {
    constexpr double nodes[16] = {
        0.52415383284386918,
        0.57223598079139826,
        0.61964368112606849,
        0.66593430114106389,
        0.71067563806531764,
        0.75344995446611462,
        0.79385787862038115,
        0.83152213346510762,
        0.86609105937014486,
        0.89724189798397114,
        0.92468380686628504,
        0.94816057788302599,
        0.96745303796886983,
        0.98238112779375319,
        0.99280575577263419,
        0.99863193092474078,
    };
    constexpr double weights[16] = {
        0.04827004425736383,
        0.047819360039637354,
        0.046922199540402255,
        0.04558693934788189,
        0.043826046502201892,
        0.041655962113473353,
        0.039096947893535114,
        0.036172897054424169,
        0.032911111388180841,
        0.029342046739267783,
        0.025499029631188046,
        0.021417949011113418,
        0.017136931456510882,
        0.012696032654631012,
        0.0081371973654528716,
        0.0035093050047352529,
    };
    if (alpha >= 1.) {
        return 0.;
    }
    FVal<T> q2 = s + mj * mj + mk * mk;
    FVal<T> muj2 = mj * mj / q2, muk2 = mk * mk / q2, muk = sqrt(muk2);
    FVal<T> av = 1. - muj2 - muk2;
    // the velocity of the Born and the upper end of y (CDST eqs. 5.12, 5.14)
    FVal<T> vel_born = sqrt(max(av * av - 4. * muj2 * muk2, 0.)) / av;
    FVal<T> y_plus = 1. - 2. * muk * (1. - muk) / av;
    FVal<T> l_alpha = -log(alpha);
    FVal<T> total = 0.;
    for (int n = 0; n < 32; ++n) {
        double t = n < 16 ? 1. - nodes[15 - n] : nodes[n - 16];
        double w = n < 16 ? weights[15 - n] : weights[n - 16];
        double u = 1. - t;
        FVal<T> y = alpha * y_plus * exp(l_alpha * (1. - u * u));
        FVal<T> jac = y * l_alpha * 2. * u * w;
        FVal<T> omy = 1. - y;
        FVal<T> vel =
            sqrt(max((2. * muk2 + av * omy) * (2. * muk2 + av * omy) - 4. * muk2, 0.)) /
            (av * omy);
        FVal<T> z0 = gluon ? FVal<T>(0.5) : av * y / (2. * (muj2 + av * y));
        FVal<T> zp = z0 * (1. + vel), zm = z0 * (1. - vel);
        FVal<T> dz = 2. * z0, z1 = 2. * z0 * z0;
        // int dz_i / (1 - z_j (1 - y)) and int dz_i / (1 - z_i (1 - y))
        FVal<T> l_j = log1p((zp - zm) * omy / (y + zm * omy)) / omy;
        FVal<T> value;
        if (gluon) {
            FVal<T> l_i = log1p((zp - zm) * omy / (1. - zp * omy)) / omy;
            FVal<T> zz = z1 - z0 * z0 * z0 * (2. + 2. * vel * vel / 3.);
            FVal<T> zpm = z0 * z0 * (1. - vel * vel);
            value = CS_CA * (l_i + l_j + zz - ((1. - kappa) * zpm + 2.) * dz) +
                n_f * CS_TR * (dz - 2. * (zz - (1. - kappa) * zpm * dz));
        } else {
            value = CS_CF *
                (2. * l_j - vel_born * (2. * dz - z1 + dz * 2. * muj2 / (av * y)));
        }
        total = total + jac * omy * value / (vel_born * y);
    }
    return total;
}

template <typename T>
KERNELSPEC void kernel_cs_insertion(
    FIn<T, 2> momenta,
    FIn<T, 3> color_correlated,
    FIn<T, 0> alpha_s,
    FIn<T, 0> ren_scale,
    FIn<T, 1> casimir,
    FIn<T, 1> gamma,
    FIn<T, 1> k_constant,
    FIn<T, 1> masses,
    FIn<T, 1> options,
    IIn<T, 0> incoming,
    FOut<T, 1> insertion,
    FOut<T, 2> poles
) {
    FVal<T> kappa = options[0], alpha_ff = options[1], n_f = options[2];
    std::size_t n_in = single_index(incoming);
    FVal<T> mu2 = FVal<T>(ren_scale) * FVal<T>(ren_scale);
    FVal<T> a = FVal<T>(alpha_s) / (2. * 3.14159265358979323846);
    for (std::size_t b = 0; b < insertion.size(); ++b) {
        insertion[b] = 0.;
        poles[b][0] = 0.;
        poles[b][1] = 0.;
    }
    for (std::size_t j = 0; j < momenta.size(); ++j) {
        FVal<T> t2 = casimir[j];
        if (t2 == 0.) {
            continue;
        }
        bool gluon = t2 == CS_CA;
        FVal<T> mj = masses[j], g = gamma[j], k = k_constant[j];
        // Gamma_j of CDST eqs. 6.27 to 6.29 (no heavy quarks for a gluon)
        FVal<T> gamma_1 = mj > 0. ? FVal<T>(CS_CF) : g;
        FVal<T> gamma_0 =
            mj > 0. ? CS_CF * (0.5 * log(mj * mj / mu2) - 2.) : FVal<T>(0.);
        auto p_j = load_mom<T>(momenta[j]);
        for (std::size_t m = 0; m < momenta.size(); ++m) {
            if (m == j || FVal<T>(casimir[m]) == 0.) {
                continue;
            }
            FVal<T> mk = masses[m];
            FVal<T> s = 2. * ldot<T>(p_j, load_mom<T>(momenta[m]));
            FVal<T> l = log(mu2 / s);
            FVal<T> v_s[3];
            cs_v_singular<T>(s, mj, mk, v_s);
            // an incoming emitter has kappa = 2/3 (CDST)
            FVal<T> v_ns = cs_v_nonsingular<T>(
                gluon, s, mj, mk, g / t2, j < n_in ? FVal<T>(2. / 3.) : kappa, n_f
            );
            FVal<T> removed = j >= n_in && m >= n_in
                ? cs_removed_final_final<T>(gluon, s, mj, mk, alpha_ff, kappa, n_f)
                : FVal<T>(0.);
            FVal<T> double_pole = -v_s[0];
            FVal<T> single_pole = -(v_s[1] + v_s[0] * l) - gamma_1 / t2;
            FVal<T> finite =
                -(v_s[2] + v_s[1] * l + 0.5 * v_s[0] * l * l + v_ns - CS_PI2 / 3.) -
                (gamma_0 + g * l + g + k - removed) / t2;
            for (std::size_t b = 0; b < insertion.size(); ++b) {
                FVal<T> cc = color_correlated[b][j][m];
                insertion[b] = insertion[b] + a * cc * finite;
                poles[b][0] = poles[b][0] + a * cc * double_pole;
                poles[b][1] = poles[b][1] + a * cc * single_pole;
            }
        }
    }
}

} // namespace kernels
} // namespace madspace

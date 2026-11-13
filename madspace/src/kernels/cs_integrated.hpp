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

// the kinds of a splitting of a parton a of the hadron into the parton a' of
// the Born: q -> q, g -> g, g -> q (a' a quark) and q -> g (a' a gluon), the
// order of P^{aa'} of CS
constexpr int CS_QQ = 0, CS_GG = 1, CS_GQ = 2, CS_QG = 3;

// P^{aa'}(x) without its plus distribution and delta(1 - x) (CS eq. 10.12)
template <typename T>
KERNELSPEC FVal<T> cs_p_regular(int kind, FVal<T> x) {
    FVal<T> omx = 1. - x;
    switch (kind) {
    case CS_QQ:
        return -CS_CF * (1. + x);
    case CS_GG:
        return 2. * CS_CA * (omx / x - 1. + x * omx);
    case CS_GQ:
        return CS_TR * (x * x + omx * omx);
    default:
        return CS_CF * (1. + omx * omx) / x;
    }
}

// P^{aa'}(x) at x < 1 (the four-dimensional Altarelli-Parisi kernel)
template <typename T>
KERNELSPEC FVal<T> cs_p_full(int kind, FVal<T> x) {
    FVal<T> omx = 1. - x;
    switch (kind) {
    case CS_QQ:
        return CS_CF * (1. + x * x) / omx;
    case CS_GG:
        return 2. * CS_CA * (x / omx + omx / x + x * omx);
    case CS_GQ:
        return CS_TR * (x * x + omx * omx);
    default:
        return CS_CF * (1. + omx * omx) / x;
    }
}

// What alpha_fi removes from the final-state emitter j (a quark of mass m or a
// massless gluon) with the incoming spectator: the dipoles over 1 - x > alpha,
// at x per unit <T_j . T_a> / T_j^2, with the invariant s = 2 p_j p_a of the
// real emission
template <typename T>
KERNELSPEC FVal<T> cs_removed_final_initial(
    bool gluon, FVal<T> x, FVal<T> s, FVal<T> m, FVal<T> alpha, FVal<T> gamma_g
) {
    FVal<T> omx = 1. - x;
    if (!(omx > alpha)) {
        return 0.;
    }
    if (gluon) {
        return (2. * CS_CA * log((1. + omx) / omx) - gamma_g) / omx;
    }
    FVal<T> mu2 = m * m / s, zp = omx / (omx + mu2);
    return CS_CF / omx *
        (2. * log((omx + zp) / omx) - 2. * zp + 0.5 * zp * zp - 2. * mu2 * zp / omx);
}

// What alpha_if removes from the incoming emitter a with the final-state
// spectator j of mass m: the dipoles over u > alpha, at x per unit <T_j . T_a>
// / T_a^2, with s = 2 p_j p_a of the real emission
template <typename T>
KERNELSPEC FVal<T>
cs_removed_initial_final(int kind, FVal<T> x, FVal<T> s, FVal<T> m, FVal<T> alpha) {
    FVal<T> omx = 1. - x, mu2 = m * m / s, zp = omx / (omx + mu2);
    if (!(zp > alpha)) {
        return 0.;
    }
    FVal<T> l_u = log(zp / alpha);
    FVal<T> e = log(zp * (omx + alpha) / (alpha * (omx + zp))) / omx;
    FVal<T> l_m =
        m > 0. ? mu2 * log((1. - alpha) * (omx + mu2) / mu2) / x : FVal<T>(0.);
    switch (kind) {
    case CS_QQ:
        return CS_CF * (2. * e - (1. + x) * l_u);
    case CS_GQ:
        return CS_TR * (x * x + omx * omx) * l_u;
    case CS_QG:
        return CS_CF * ((x + 2. * omx / x) * l_u - 2. * l_m);
    default:
        return 2. * CS_CA * (e + (-1. + x * omx + omx / x) * l_u - l_m);
    }
}

// The convolution of a distribution in x with the luminosity g(x) of a beam,
// int_0^1 dx [term](x) g(x), estimated at one point z = eta + (1 - eta) r of
// weight jac = 1 - eta, with g(x) = 0 below the momentum fraction eta of the
// Born: weight of g(z) and of g(1)
template <typename T>
struct CSConvolution {
    FVal<T> z, jac, l_eta, at_z, at_one;
    // f(x) regular at x = 1
    KERNELSPEC void regular(FVal<T> coef, FVal<T> f_z) {
        at_z = at_z + coef * f_z * jac;
    }
    // [f(x)]_+ with f_z = f(z) inside the convolution, f_end = f(z) at the
    // kinematics of the end point and integral = int_0^eta f
    KERNELSPEC void plus(FVal<T> coef, FVal<T> f_z, FVal<T> f_end, FVal<T> integral) {
        at_z = at_z + coef * f_z * jac;
        at_one = at_one - coef * (f_end * jac + integral);
    }
    // [1 / (1 - x)]_+ h(x) with h_z = h(z) and h_one = h(1)
    KERNELSPEC void plus_times(FVal<T> coef, FVal<T> h_z, FVal<T> h_one) {
        at_z = at_z + coef * h_z / (1. - z) * jac;
        at_one = at_one + coef * h_one * (l_eta - jac / (1. - z));
    }
    // delta(1 - x) h
    KERNELSPEC void delta(FVal<T> coef, FVal<T> h) { at_one = at_one + coef * h; }
};

// The terms of a massive final-state quark j of CDST eqs. 6.55-6.62, -<T_j .
// T_a'> K_j^{a,a'}(x; s, m) and the mass terms of the collinear part, per
// coefficient coef = <T_j . T_a'> B, with s = 2 p_j p~_a' of the Born (s / z
// inside the convolution)
template <typename T>
KERNELSPEC void cs_massive_spectator(
    CSConvolution<T>& conv,
    int kind,
    FVal<T> coef,
    FVal<T> t2,
    FVal<T> gamma_a,
    FVal<T> s,
    FVal<T> m
) {
    FVal<T> z = conv.z, omz = 1. - z, m2 = m * m;
    FVal<T> s_z = s / z, mu2 = m2 / s, mu2_z = m2 / s_z;
    bool same = kind == CS_QQ || kind == CS_GG;
    if (same) {
        FVal<T> l_omz = log(omz), l_eta = conv.l_eta, u = conv.jac;
        conv.plus(-2. * coef, l_omz / omz, l_omz / omz, -0.5 * l_eta * l_eta);
        conv.regular(-coef, -2. * log(2. - z) / omz);
        // [(1 - x) / (2 (1 - x + mu^2)^2) - 2 (1 + log(1 - x + mu^2)) / (1 - x)]_+
        FVal<T> f_z = omz / (2. * (omz + mu2_z) * (omz + mu2_z)) -
            2. * (1. + log(omz + mu2_z)) / omz;
        FVal<T> f_end =
            omz / (2. * (omz + mu2) * (omz + mu2)) - 2. * (1. + log(omz + mu2)) / omz;
        FVal<T> integral =
            0.5 * (log((1. + mu2) / (u + mu2)) + mu2 / (1. + mu2) - mu2 / (u + mu2)) +
            2. * l_eta +
            2. * (log(mu2) * l_eta + fks_dilog<T>(-1. / mu2) - fks_dilog<T>(-u / mu2));
        conv.plus(-coef, f_z, f_end, integral);
        conv.plus_times(-2. * coef, log(2. + mu2_z - z), log(1. + mu2));
        conv.plus_times(-2. * coef, log((2. - z) / (2. - z + mu2_z)), -log(1. + mu2));
        conv.delta(-coef, -1.5 + mu2 * log(mu2 / (1. + mu2)) + 0.5 * mu2 / (1. + mu2));
        FVal<T> rq = sqrt(s + m2);
        conv.delta(
            -coef * gamma_a / t2,
            log((s - 2. * m * rq + 2. * m2) / s) + 2. * m / (rq + m)
        );
    }
    if (kind == CS_QG || kind == CS_GG) {
        FVal<T> factor = kind == CS_GG ? CS_CA / CS_CF : 1.;
        conv.regular(
            -coef * factor, 2. * CS_CF / CS_CA * mu2_z / z * log(mu2_z / (omz + mu2_z))
        );
    }
    conv.regular(
        -coef / t2, cs_p_regular<T>(kind, z) * log(omz * s_z / (omz * s_z + m2))
    );
}

template <typename T>
KERNELSPEC void kernel_cs_collinear(
    FIn<T, 2> momenta,
    FIn<T, 1> born,
    FIn<T, 3> color_correlated,
    FIn<T, 0> alpha_s,
    FIn<T, 0> fact_scale1,
    FIn<T, 0> fact_scale2,
    FIn<T, 0> x1,
    FIn<T, 0> x2,
    FIn<T, 0> r,
    FIn<T, 1> casimir,
    FIn<T, 1> gamma,
    FIn<T, 1> masses,
    FIn<T, 1> options,
    FOut<T, 1> z,
    FOut<T, 3> weights
) {
    constexpr double pi2 = CS_PI2;
    FVal<T> alpha_fi = options[0], alpha_if = options[1], alpha_ii = options[2];
    FVal<T> n_f = options[3];
    FVal<T> gamma_g = 11. / 6. * CS_CA - 2. / 3. * CS_TR * n_f;
    FVal<T> a = FVal<T>(alpha_s) / (2. * 3.14159265358979323846);
    std::size_t n = momenta.size();
    for (std::size_t l = 0; l < 2; ++l) {
        FVal<T> eta = l == 0 ? FVal<T>(x1) : FVal<T>(x2);
        FVal<T> mu_f = l == 0 ? FVal<T>(fact_scale1) : FVal<T>(fact_scale2);
        FVal<T> rv = r, zl = eta + (1. - eta) * rv;
        z[l] = zl;
        FVal<T> t2 = casimir[l];
        for (std::size_t b = 0; b < weights.size(); ++b) {
            for (std::size_t c = 0; c < 3; ++c) {
                weights[b][l][c] = 0.;
            }
        }
        if (t2 == 0.) {
            continue;
        }
        bool gluon = t2 == CS_CA;
        int same = gluon ? CS_GG : CS_QQ, other = gluon ? CS_QG : CS_GQ;
        FVal<T> omz = 1. - zl, l_omz = log(omz), l_eta = log(1. - eta);
        FVal<T> l_ratio = log(omz / zl);
        auto p_l = load_mom<T>(momenta[l]);
        std::size_t o = 1 - l;
        for (std::size_t b = 0; b < weights.size(); ++b) {
            CSConvolution<T> conv_same{zl, 1. - eta, l_eta, 0., 0.};
            CSConvolution<T> conv_other{zl, 1. - eta, l_eta, 0., 0.};
            FVal<T> born_b = born[b];
            // P: (1 / T_a'^2) sum_j <T_j . T_a'> log(mu_F^2 / s_ja') P^{aa'}
            FVal<T> coef_p = 0.;
            for (std::size_t j = 0; j < n; ++j) {
                if (j == l || FVal<T>(casimir[j]) == 0.) {
                    continue;
                }
                FVal<T> s = 2. * ldot<T>(p_l, load_mom<T>(momenta[j]));
                coef_p =
                    coef_p + FVal<T>(color_correlated[b][j][l]) * log(mu_f * mu_f / s);
            }
            coef_p = coef_p / t2;
            conv_same.regular(coef_p, cs_p_regular<T>(same, zl));
            conv_same.plus(2. * coef_p * t2, 1. / omz, 1. / omz, -l_eta);
            conv_same.delta(coef_p, gluon ? gamma_g : FVal<T>(1.5 * CS_CF));
            conv_other.regular(coef_p, cs_p_regular<T>(other, zl));
            // Kbar of CS eqs. 10.27-10.28 times the Born
            FVal<T> int_ratio =
                -0.5 * l_eta * l_eta - fks_dilog<T>(1. - eta) + pi2 / 6.;
            if (gluon) {
                conv_same.regular(born_b, cs_p_regular<T>(CS_GG, zl) * l_ratio);
                conv_same.plus(
                    2. * CS_CA * born_b, l_ratio / omz, l_ratio / omz, int_ratio
                );
                conv_same.delta(
                    -born_b, (50. / 9. - pi2) * CS_CA - 16. / 9. * CS_TR * n_f
                );
                conv_other.regular(
                    born_b, cs_p_regular<T>(CS_QG, zl) * l_ratio + CS_CF * zl
                );
            } else {
                conv_same.regular(born_b, CS_CF * (-(1. + zl) * l_ratio + omz));
                conv_same.plus(
                    2. * CS_CF * born_b, l_ratio / omz, l_ratio / omz, int_ratio
                );
                conv_same.delta(-born_b, CS_CF * (5. - pi2));
                conv_other.regular(
                    born_b, cs_p_regular<T>(CS_GQ, zl) * l_ratio + 2. * CS_TR * zl * omz
                );
            }
            // Ktilde of CS eq. 10.29 with the other incoming parton, and what
            // alpha_ii removes
            if (FVal<T>(casimir[o]) != 0.) {
                FVal<T> coef = -FVal<T>(color_correlated[b][o][l]) / t2;
                conv_same.regular(coef, cs_p_regular<T>(same, zl) * l_omz);
                conv_same.plus(
                    2. * t2 * coef, l_omz / omz, l_omz / omz, -0.5 * l_eta * l_eta
                );
                conv_same.delta(coef, -t2 * pi2 / 3.);
                conv_other.regular(coef, cs_p_regular<T>(other, zl) * l_omz);
                if (omz > alpha_ii) {
                    FVal<T> l_alpha = log(omz / alpha_ii);
                    conv_same.regular(-coef, cs_p_full<T>(same, zl) * l_alpha);
                    conv_other.regular(-coef, cs_p_full<T>(other, zl) * l_alpha);
                }
            }
            // the final-state partons j as emitters with the spectator a' and as
            // spectators of a'
            for (std::size_t j = 2; j < n; ++j) {
                FVal<T> cj = casimir[j];
                if (cj == 0.) {
                    continue;
                }
                FVal<T> cc = color_correlated[b][j][l];
                FVal<T> mj = masses[j];
                FVal<T> s = 2. * ldot<T>(p_l, load_mom<T>(momenta[j]));
                if (mj == 0.) {
                    FVal<T> coef = cc * FVal<T>(gamma[j]) / cj;
                    conv_same.plus(coef, 1. / omz, 1. / omz, -l_eta);
                    conv_same.delta(coef, 1.);
                } else {
                    cs_massive_spectator<T>(conv_same, same, cc, t2, gamma[l], s, mj);
                    cs_massive_spectator<T>(conv_other, other, cc, t2, gamma[l], s, mj);
                }
                conv_same.regular(
                    cc / cj,
                    cs_removed_final_initial<T>(
                        cj == CS_CA, zl, s / zl, mj, alpha_fi, gamma_g
                    )
                );
                conv_same.regular(
                    cc / t2, cs_removed_initial_final<T>(same, zl, s / zl, mj, alpha_if)
                );
                conv_other.regular(
                    cc / t2,
                    cs_removed_initial_final<T>(other, zl, s / zl, mj, alpha_if)
                );
            }
            weights[b][l][0] = a * conv_same.at_z;
            weights[b][l][1] = a * conv_other.at_z;
            weights[b][l][2] = a * (conv_same.at_one + conv_other.at_one);
        }
    }
}

} // namespace kernels
} // namespace madspace

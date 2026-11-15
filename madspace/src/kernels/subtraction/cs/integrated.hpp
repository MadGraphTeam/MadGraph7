#pragma once

#include "../common.hpp"
#include "mapping.hpp"

namespace madspace {
namespace kernels {

// The integrated Catani-Seymour dipoles (arXiv:hep-ph/9605323, Sec. 10;
// masses: arXiv:hep-ph/0201036, Sec. 6) in conventional dimensional
// regularization, with the normalization (4 pi)^eps / Gamma(1 - eps) of the
// virtual of MadLoop. The alpha terms are what the gates of the dipoles of
// mapping.hpp remove from them at alpha = 1, in four dimensions, per unit
// <T_j . T_k> / T_j^2 and alpha_s / (2 pi).
//
// The kernels vectorize: the masses, couplings and options are the same for
// all events, so that whether a mass vanishes is a flag of the call (the mass
// of the recoil system of a decay, which varies, is massive unless it is a
// single massless particle), and what depends on the event selects with where.

constexpr double CS_PI2 = 3.14159265358979323846 * 3.14159265358979323846;

// V^(S)(s, m_j, m_k) of CDST eqs. 6.20 and 6.21, its coefficients of eps^-2,
// eps^-1 and eps^0
template <typename T>
KERNELSPEC void cs_v_singular(
    FVal<T> s, FVal<T> mj, FVal<T> mk, bool massive_j, bool massive_k, FVal<T> (&v_s)[3]
) {
    FVal<T> mj2 = mj * mj, mk2 = mk * mk, q2 = s + mj2 + mk2;
    if (!massive_j && !massive_k) {
        v_s[0] = 1.;
        v_s[1] = 0.;
        v_s[2] = 0.;
    } else if (massive_j && massive_k) {
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
        FVal<T> m2 = massive_j ? mj2 : mk2;
        FVal<T> l_ms = log(m2 / s), l_sq = log(s / q2);
        v_s[0] = 0.5;
        v_s[1] = 0.5 * l_ms;
        v_s[2] = -0.25 * l_ms * l_ms - CS_PI2 / 12. - 0.5 * l_ms * l_sq -
            0.5 * log(m2 / q2) * l_sq;
    }
}

// V^(NS)_j(s, m_j, m_k; kappa) of CDST eqs. 6.22 to 6.26 for a quark or a
// massless gluon j, without the splitting of a gluon into heavy quarks (they
// decouple from the virtual), in two parts: the soft part per unit <T_j .
// T_k>, and the collinear part per unit <T_j . T_k> / T_j^2 with gamma_j of j
// and for a gluon its self coupling C_A and its coupling to the light quark
// pairs T_R n_f (the photon of QED has none and sum_f N_c Q_f^2), so that
// the photon, whose T_j^2 is zero, keeps the collinear part
template <typename T>
struct CSVNonsingular {
    FVal<T> soft, collinear;
};

template <typename T>
KERNELSPEC CSVNonsingular<T> cs_v_nonsingular(
    bool boson,
    FVal<T> s,
    FVal<T> mj,
    FVal<T> mk,
    bool massive_j,
    bool massive_k,
    double gamma_j,
    double kappa,
    double self_coupling,
    double pair_splitting
) {
    FVal<T> mj2 = mj * mj, mk2 = mk * mk, q2 = s + mj2 + mk2, q = sqrt(q2);
    FVal<T> l_sq = log(s / q2);
    if (massive_k && !massive_j) {
        FVal<T> collinear =
            gamma_j * (l_sq - 2. * log((q - mk) / q) - 2. * mk / (q + mk));
        if (boson) {
            collinear = collinear +
                (kappa - 2. / 3.) * mk2 / s * (2. * pair_splitting - self_coupling) *
                    log(2. * mk / (q + mk));
        }
        return {CS_PI2 / 6. - dilog<T>(s / q2), collinear};
    }
    if (boson || !massive_j) {
        return {FVal<T>(0.), FVal<T>(0.)};
    }
    if (!massive_k) {
        return {
            CS_PI2 / 6. - dilog<T>(s / q2) - 2. * l_sq - mj2 / s * log(mj2 / q2),
            gamma_j * l_sq
        };
    }
    FVal<T> d = 4. * mj2 * mk2 / (s * s);
    FVal<T> v = sqrt(1. - d), omv = d / (1. + v);
    FVal<T> rho2 = omv / (1. + v);
    FVal<T> rhoj2 = (omv + 2. * mj2 / s) / (1. + v + 2. * mj2 / s);
    FVal<T> rhok2 = (omv + 2. * mk2 / s) / (1. + v + 2. * mk2 / s);
    FVal<T> qmk = q - mk;
    return {
        (log(rho2) * log(1. + rho2) + 2. * dilog<T>(rho2) - dilog<T>(1. - rhoj2) -
         dilog<T>(1. - rhok2) - CS_PI2 / 6.) /
                v +
            log(qmk / q) - 2. * log((qmk * qmk - mj2) / q2) -
            2. * mj2 / s * log(mj / qmk) - mk / qmk + 2. * mk * (2. * mk - q) / s +
            CS_PI2 / 2.,
        gamma_j * l_sq
    };
}

// An antiderivative of log|u + c| / u for real c that stays well conditioned
// for |c| << |u|: 1/2 log^2|u| + Li2(-c/u) for |u| >= |c| (c = 0 among them)
// and log|c| log|u| - Li2(-u/c) - 1/2 log^2|c| - K otherwise, with the
// constant K = pi^2/6 for u/c > 0 and -pi^2/3 for u/c < 0 that joins the two
// (u does not cross zero). Each dilogarithm takes zero on the other side.
template <typename T>
KERNELSPEC FVal<T> cs_log_over_primitive(FVal<T> u, FVal<T> c) {
    FVal<T> l_u = log(fabs(u));
    auto outer = fabs(u) >= fabs(c);
    FVal<T> c_in = where(outer, FVal<T>(1.), c);
    FVal<T> l_c = log(fabs(c_in));
    FVal<T> k = where(u / c_in > 0., FVal<T>(CS_PI2 / 6.), FVal<T>(-CS_PI2 / 3.));
    FVal<T> big = 0.5 * l_u * l_u + dilog<T>(where(outer, -c / u, FVal<T>(0.)));
    FVal<T> small = l_c * l_u - dilog<T>(where(outer, FVal<T>(0.), -u / c_in)) -
        0.5 * l_c * l_c - k;
    return where(outer, big, small);
}

// int_u0^u1 du log|u + c| / u
template <typename T>
KERNELSPEC FVal<T> cs_log_over(FVal<T> u0, FVal<T> u1, FVal<T> c) {
    return cs_log_over_primitive<T>(u1, c) - cs_log_over_primitive<T>(u0, c);
}

// The integrals over alpha y_+ < y < y_+ of the final-final dipoles of a
// massless emitted parton (CDST eqs. 5.11-5.14 with m_i = 0, m_ij = m_j), from
// the masses mu_n^2 = m_n^2 / Q^2: J = int dy (2 / y) log R(y) with R = (y +
// z_+ (1 - y)) / (y + z_- (1 - y)), which the z integral of the eikonal terms
// leaves, and S = int dy D (1 - y) v(y) / y. With D = 1 - mu_j^2 - mu_k^2 and
// v^2 = 1 - 4 mu_k^2 y / (D (1 - y)^2) the velocity of the dipole, (1 - y) D
// + 2 mu_k^2 = mu_k (t + 1 / t) makes v rational for a massive spectator: R =
// (t - mu_k) / (t (1 - mu_k t)), D (1 - y) v = mu_k (t - 1 / t) and y = -mu_k
// (t - t_+) (t - t_-) / (D t) with t_+- = (1 - mu_j^2 + mu_k^2 +- sqrt(lambda(1,
// mu_j^2, mu_k^2))) / (2 mu_k), and y from alpha y_+ to y_+ is t from t_alpha
// to 1. int dy / y log R then is a sum of int dt log|t - a| / (t - b), of
// logarithms and dilogarithms, and S = mu_k int dt (t^2 - 1)^2 / (t^2 (t -
// t_+) (t - t_-)) one of logarithms.
template <typename T>
struct CSAlphaIntegrals {
    FVal<T> j_log, s_over_y;
};

template <typename T>
KERNELSPEC CSAlphaIntegrals<T> cs_alpha_integrals(
    FVal<T> muj2, FVal<T> muk2, bool massive_j, bool massive_k, double alpha
) {
    FVal<T> muk = sqrt(muk2);
    FVal<T> av = 1. - muj2 - muk2;
    FVal<T> lam = sqrt(max(av * av - 4. * muj2 * muk2, 0.));
    FVal<T> omy_plus = 2. * muk * (1. - muk) / av, y_plus = 1. - omy_plus;
    FVal<T> y_alpha = alpha * y_plus;
    double l_alpha = log(alpha);
    if (!massive_k) {
        // R = 1 / (mu_j^2 + D y), v = 1 and y_+ = 1
        FVal<T> j_log = !massive_j
            ? FVal<T>(l_alpha * l_alpha)
            : -2. * (-log(av) * l_alpha + cs_log_over<T>(y_alpha, y_plus, muj2 / av));
        return {j_log, av * (-l_alpha - y_plus * (1. - alpha))};
    }
    FVal<T> b_one = 1. - muj2 + muk2, b_alpha = b_one - y_alpha * av;
    FVal<T> s_alpha = sqrt(max(b_alpha * b_alpha - 4. * muk2, 0.));
    FVal<T> t_plus = (b_one + lam) / (2. * muk), t_minus = 1. / t_plus;
    FVal<T> t_alpha = (b_alpha + s_alpha) / (2. * muk);
    // t_+ - t_alpha, t_+ - 1 / mu_k and t_- - mu_k without cancellations
    FVal<T> delta =
        y_alpha * av * (1. + (b_one + b_alpha) / (lam + s_alpha)) / (2. * muk);
    FVal<T> c_plus = -2. * muj2 / (muk * (lam + 1. + muj2 - muk2));
    FVal<T> c_minus = 2. * muj2 * muk / (av + lam);
    // log R = log(t - mu_k) - log t - log(1 / mu_k - t) - log mu_k and
    // d log y / dt = 1 / (t - t_+) + 1 / (t - t_-) - 1 / t
    FVal<T> logs[3] = {muk, 0., 1. / muk};
    double log_signs[3] = {1., -1., -1.};
    FVal<T> poles[3] = {t_plus, t_minus, 0.};
    double pole_signs[3] = {1., 1., -1.};
    FVal<T> u_alpha[3] = {-delta, t_alpha - t_minus, t_alpha};
    FVal<T> u_one[3] = {1. - t_plus, 1. - t_minus, 1.};
    FVal<T> j_log = log(muk) * l_alpha;
    for (int b = 0; b < 3; ++b) {
        for (int a = 0; a < 3; ++a) {
            FVal<T> c = b == 0 && a == 2 ? c_plus
                : b == 1 && a == 0
                ? c_minus
                : poles[b] - logs[a];
            j_log = j_log +
                log_signs[a] * pole_signs[b] * cs_log_over<T>(u_alpha[b], u_one[b], c);
        }
    }
    // (t^2 - 1)^2 / (t^2 (t - t_+) (t - t_-)) = 1 + (t_+ + t_-) / t + 1 / t^2 +
    // (t_+ - t_-) (1 / (t - t_+) - 1 / (t - t_-))
    FVal<T> t_plus_one = (b_one + lam - 2. * muk) / (2. * muk);
    FVal<T> s_t = (1. - t_alpha) * (1. + 1. / t_alpha) -
        (t_plus + t_minus) * log(t_alpha) +
        (t_plus - t_minus) *
            (log(t_plus_one / delta) - log(t_plus_one / t_plus / (t_alpha - t_minus)));
    return {2. * j_log, muk * s_t};
}

// What the gate y < alpha y_+ of the final-final dipoles of the emitter j
// with the spectator k removes from their integral at alpha = 1: the
// dipoles over alpha y_+ < y < y_+ in four dimensions, per unit <T_j . T_k> /
// T_j^2 and alpha_s / (2 pi), of q -> q g for a quark j and of g -> g g (with
// the symmetry factor 1/2) and n_f g -> q q~ for a gluon j, in closed form.
// The z integral of the kernels leaves (2 / y) log R(y) for both emitters
// (J of cs_alpha_integrals), and rational functions of y and of v^2 = 1 -
// 4 mu_k^2 y / (D (1 - y)^2).
template <typename T>
KERNELSPEC FVal<T> cs_removed_final_final(
    bool boson,
    FVal<T> s,
    FVal<T> mj,
    FVal<T> mk,
    bool massive_j,
    bool massive_k,
    double alpha,
    double kappa,
    double casimir,
    double self_coupling,
    double pair_splitting
) {
    if (alpha >= 1.) {
        return 0.;
    }
    FVal<T> q2 = s + mj * mj + mk * mk;
    FVal<T> muj2 = mj * mj / q2, muk2 = mk * mk / q2, muk = sqrt(muk2);
    FVal<T> av = 1. - muj2 - muk2;
    FVal<T> lam = sqrt(max(av * av - 4. * muj2 * muk2, 0.));
    // the velocity of the Born and the ends of y (CDST eqs. 5.12, 5.14)
    FVal<T> vel_born = lam / av;
    FVal<T> omy_plus = 2. * muk * (1. - muk) / av, y_plus = 1. - omy_plus;
    FVal<T> y_alpha = alpha * y_plus;
    double l_alpha = log(alpha);
    FVal<T> j_log =
        cs_alpha_integrals<T>(muj2, muk2, massive_j, massive_k, alpha).j_log;
    // int dy (1 - y) / y
    FVal<T> a_0 = -l_alpha - y_plus * (1. - alpha);
    if (boson) {
        // with (1 - y) (1 - v^2) / y = 4 mu_k^2 / (D (1 - y)), the self
        // coupling C_A and the light pairs T_R n_f of a gluon
        double gamma_g = 11. / 6. * self_coupling - 2. / 3. * pair_splitting;
        double c_v = self_coupling * (1. / 12. - 0.25 * (1. - kappa)) +
            pair_splitting * (0.5 * (1. - kappa) - 1. / 6.);
        FVal<T> x_1 = !massive_k
            ? FVal<T>(0.)
            : FVal<T>(4. * muk2 / av * log((1. - y_alpha) / omy_plus));
        return self_coupling * j_log - gamma_g * a_0 + c_v * x_1;
    }
    // int dy (1 - y) z_1 / y with z_1 = 2 z_0^2 = D^2 y^2 / (2 (mu_j^2 + D y)^2)
    // in u = mu_j^2 + D y
    FVal<T> e = 1. - muk2, u_plus = muj2 + av * y_plus, u_alpha = muj2 + av * y_alpha;
    FVal<T> z_1 =
        (u_alpha - u_plus + (e + muj2) * log(u_plus / u_alpha) +
         e * muj2 * (1. / u_plus - 1. / u_alpha)) /
        (2. * av);
    return casimir * (j_log / vel_born - 2. * a_0 + z_1);
}

// The integral of the dipole of a final-state quark j with the decaying
// particle a (mass m_a) as the spectator (cs_final_decay_dipole), over the
// emission at fixed Born in d = 4 - 2 eps dimensions, relative to the pair term
// of CDST with the recoil system K (mass m_k) as the spectator. Per alpha_s /
// (2 pi) (4 pi)^eps / Gamma(1 - eps) (mu^2 / s)^eps with s = m_a^2 - m_j^2 -
// m_k^2 it is
//
//   (s / 2 + 2 m_j^2) H_j - (s / 2) H_k - I_a,
//
// with H_x the integral of 1 / (p_a.q p_x.q) for the emitted gluon q and I_a
// that of m_a^2 / (p_a.q)^2. The eikonal term of the final-final kernel is
// exactly s / (p_a.q p_j.q), and the pair term of CDST (V^(S) of its eq. 6.20)
// has the mean of its integral and of that of s / (p_a.q p_k.q) of the other
// ordering, which only the sum over both orderings of the I operator takes:
// for one ordering the integral differs by s (H_j - H_k) / 2 (checked against
// the integral of the dipole in d dimensions over its variables). The soft
// terms of a add 2 m_j^2 H_j - I_a. All are integrals of functions of the
// invariants, so over the phase space of a -> q j K over that of a -> j K. In
// the rest frame of a, at the energy E of q, j and K share s' = m_a^2 - 2 m_a E
// isotropically in their rest frame, and each is an integral over E of E^(-1 -
// 2 eps) times the ratio rho(E) of the phase spaces of j K at s' and at m_a^2,
// times sqrt(s') / m_a^2 <1 / (E'_x - |p'_x| c)> for H_x, with the average over
// the angle in d - 1 dimensions. With X = s / (4 E_max^2), E_max = (m_a^2 - (m_j
// + m_k)^2) / (2 m_a),
//
//   I_a = -1 / eps - log X - 2 + 2 J_a,
//   H_x = (l_x / sqrt(lambda)) (-1 / eps - log X - 2 - eta_x) + 2 T_x /
//         sqrt(lambda)                                  (m_x > 0),
//   s H_x = 1 / eps^2 + log X / eps + log^2 X / 2 - 5 pi^2 / 6 + 2 Li2(1 -
//         m_y^2 / m_a^2)                                (m_x = 0),
//
// with l_x = log((1 + beta_x) / (1 - beta_x)) of the velocity beta_x of x in
// the rest frame of a, eta_x the term of order eps of the average of 1 / (1 -
// beta_x c) over its value, J_a = int_0^E_max dE (rho(E) - 1) / E and T_x the
// same for H_x over its value at E = 0 in four dimensions (m_y the mass of the
// other one). In s' = m_x^2 + m_y^2 + m_x m_y (x + 1 / x) the square root of
// lambda(s', m_x^2, m_y^2) = m_x m_y (1 / x - x) is rational, dE / E = (-1 / x +
// 1 / (x - x_a) + 1 / (x - 1 / x_a)) dx from x_a (s' = m_a^2) to 1, and J_a and
// T_x are logarithms and dilogarithms, here in forms that keep their digits
// for light masses. A massless x averages 1 / (1 - c) to -(1 - 2 eps) / (2 eps)
// at every E.
template <typename T>
KERNELSPEC FVal<T> cs_decay_eta(FVal<T> beta) {
    // the series below beta = 0.1, the closed form above
    auto small = beta < 0.1;
    FVal<T> series = 0.;
    {
        constexpr double c[8] = {
            2. / 9.,
            94. / 675.,
            10154. / 99225.,
            120926. / 1488375.,
            36490618. / 540280125.,
            554974302418. / 9587270818125.,
            1458381727046. / 28761812454375.,
            511730469371542. / 11334768817246875.,
        };
        FVal<T> b2 = beta * beta;
        for (int n = 7; n >= 0; --n) {
            series = (series + c[n]) * b2;
        }
    }
    FVal<T> b = where(small, FVal<T>(0.5), beta);
    FVal<T> l = log((1. + b) / (1. - b));
    FVal<T> closed = 2. * log(2.) - 2. - log((1. - b) * (1. + b)) - 0.5 * l +
        (CS_PI2 / 3. - 2. * dilog<T>((1. - b) / (1. + b))) / l + 2. * log(b);
    return where(small, series, closed);
}

// int_0^{1 - x_a} du log(1 + u / c) / (u + q) for q != 0 and c - q = c_q > 0
template <typename T>
KERNELSPEC FVal<T> cs_decay_log_over(FVal<T> xa, FVal<T> c, FVal<T> q, FVal<T> c_q) {
    FVal<T> u1 = 1. - xa;
    return log1p(-q / c) * log1p(u1 / q) - dilog<T>(-(u1 + q) / c_q) +
        dilog<T>(-q / c_q);
}

// (eps^-2, eps^-1, eps^0) of s H_x for the emitter x of mass m_x and the other
// particle of mass m_y (see above)
template <typename T>
KERNELSPEC Triplet<FVal<T>, FVal<T>, FVal<T>>
cs_decay_eikonal(double ma, FVal<T> mx, FVal<T> my, bool massive_x, bool massive_y) {
    double ma2 = ma * ma;
    FVal<T> mx2 = mx * mx, my2 = my * my;
    FVal<T> s = ma2 - mx2 - my2;
    FVal<T> rl = sqrt(max(s * s - 4. * mx2 * my2, 0.));
    FVal<T> e_max = (ma - mx - my) * (ma + mx + my) / (2. * ma);
    FVal<T> l_x = log(s / (4. * e_max * e_max));
    if (!massive_x) {
        return {
            1., l_x, 0.5 * l_x * l_x - 5. * CS_PI2 / 6. + 2. * dilog<T>(1. - my2 / ma2)
        };
    }
    FVal<T> beta = rl / (ma2 + mx2 - my2);
    FVal<T> l_beta = log((1. + beta) / (1. - beta));
    // T_x with the logs of x + 1 / rho, x and x + rho (rho = m_x / m_y) and the
    // poles of dE / E, each log taken relative to its value at x_a
    FVal<T> t;
    if (!massive_y) {
        t = -dilog<T>(1. - mx2 / ma2);
    } else {
        FVal<T> xa = 2. * mx * my / (s + rl);
        FVal<T> rho = mx / my, u1 = 1. - xa;
        FVal<T> q_b = -(1. - xa) * (1. + xa) / xa;
        FVal<T> cs[3] = {xa + 1. / rho, xa, xa + rho};
        double signs[3] = {1., -1., -1.};
        t = 0.;
        for (int n = 0; n < 3; ++n) {
            FVal<T> c = cs[n], r = xa - c;
            FVal<T> f_0 =
                n == 1 ? 0.5 * log(xa) * log(xa) : cs_decay_log_over<T>(xa, c, xa, -r);
            FVal<T> f_a = -dilog<T>(-u1 / c);
            FVal<T> f_b = cs_decay_log_over<T>(xa, c, q_b, 1. / xa - r);
            t = t + signs[n] * (-f_0 + f_a + f_b);
        }
    }
    FVal<T> ratio = s / rl;
    return {
        0.,
        -ratio * l_beta,
        ratio * (l_beta * (-l_x - 2. - cs_decay_eta<T>(beta)) + 2. * t)
    };
}

// (eps^-2, eps^-1, eps^0) of the integral of the decay dipole of the quark j
// relative to the pair term of CDST, (s / 2 + 2 m_j^2) H_j - (s / 2) H_k - I_a
template <typename T>
KERNELSPEC Triplet<FVal<T>, FVal<T>, FVal<T>>
cs_decay_soft(double ma, FVal<T> mj, FVal<T> mk, bool massive_j, bool massive_k) {
    double ma2 = ma * ma;
    FVal<T> mj2 = mj * mj, mk2 = mk * mk;
    FVal<T> s = ma2 - mj2 - mk2;
    FVal<T> rl = sqrt(max(s * s - 4. * mj2 * mk2, 0.));
    FVal<T> w = s + rl;
    FVal<T> e_max = (ma - mj - mk) * (ma + mj + mk) / (2. * ma);
    // J_a
    FVal<T> xa = 2. * mj * mk / w;
    FVal<T> kk = 4. * ma2 * mk2 / (ma2 - mj2 + mk2 + rl);
    FVal<T> kj = 4. * ma2 * mj2 / (ma2 + mj2 - mk2 + rl);
    FVal<T> t = (mj2 - mk2) * log((w + 2. * mk2) / (w + 2. * mj2));
    if (massive_k) {
        t = t + 0.5 * kk * log(2. * mk2 / w);
    }
    if (massive_j) {
        t = t + 0.5 * kj * log(2. * mj2 / w);
    }
    FVal<T> i_a = -log(s / (4. * e_max * e_max)) - 2. + 2. * (2. * log1p(xa) + t / rl);
    auto h_j = cs_decay_eikonal<T>(ma, mj, mk, massive_j, massive_k);
    auto h_k = cs_decay_eikonal<T>(ma, mk, mj, massive_k, massive_j);
    FVal<T> c_j = 0.5 + 2. * mj2 / s;
    return {
        c_j * h_j.first - 0.5 * h_k.first,
        c_j * h_j.second - 0.5 * h_k.second + 1.,
        c_j * h_j.third - 0.5 * h_k.third - i_a
    };
}

// What the gate y < alpha y_+ removes from the soft terms of the decaying
// particle in the dipole of the quark j (cs_decay_soft), in four dimensions per
// unit <T_j . T_a> and alpha_s / (2 pi): 2 / sqrt(lambda(1, mu_j^2, mu_k^2))
// (mu_j^2 J - S) with J and S of cs_alpha_integrals
template <typename T>
KERNELSPEC FVal<T> cs_removed_decay_soft(
    FVal<T> s, FVal<T> mj, FVal<T> mk, bool massive_j, bool massive_k, double alpha
) {
    if (alpha >= 1.) {
        return 0.;
    }
    FVal<T> q2 = s + mj * mj + mk * mk;
    FVal<T> muj2 = mj * mj / q2, muk2 = mk * mk / q2;
    FVal<T> av = 1. - muj2 - muk2;
    FVal<T> lam = sqrt(max(av * av - 4. * muj2 * muk2, 0.));
    auto integrals = cs_alpha_integrals<T>(muj2, muk2, massive_j, massive_k, alpha);
    return 2. / lam * (muj2 * integrals.j_log - integrals.s_over_y);
}

template <typename T>
KERNELSPEC void kernel_cs_insertion(
    FIn<T, 2> momenta,
    FIn<T, 3> color_correlated,
    FIn<T, 1> born,
    FIn<T, 0> alpha_s,
    FIn<T, 0> ren_scale,
    FSIn<T, 1> casimir,
    FSIn<T, 1> gamma,
    FSIn<T, 1> k_constant,
    FSIn<T, 1> self_coupling,
    FSIn<T, 1> pair_splitting,
    FSIn<T, 1> charges,
    FSIn<T, 1> spectator_weights,
    FSIn<T, 1> masses,
    FSIn<T, 1> options,
    ISIn<T, 0> incoming,
    FOut<T, 1> insertion,
    FOut<T, 2> poles
) {
    double kappa = single_value(options[0]), alpha_ff = single_value(options[1]);
    bool qed = single_value(options[3]) != 0.;
    std::size_t n_in = single_index(incoming);
    std::size_t n = momenta.size();
    FVal<T> mu2 = FVal<T>(ren_scale) * FVal<T>(ren_scale);
    FVal<T> a = FVal<T>(alpha_s) / (2. * 3.14159265358979323846);
    for (std::size_t b = 0; b < insertion.size(); ++b) {
        insertion[b] = 0.;
        poles[b][0] = 0.;
        poles[b][1] = 0.;
    }
    // a decaying particle emits no collinear parton, its soft terms belong to
    // the dipoles of the final-state partons with it as the spectator
    bool decay = n_in == 1;
    for (std::size_t j = decay ? 1 : 0; j < n; ++j) {
        double t2 = single_value(casimir[j]);
        double ca = single_value(self_coupling[j]),
               trnf = single_value(pair_splitting[j]);
        // a gluon, or a photon, which has no T_j^2 but splits into pairs
        bool boson = ca != 0. || trnf != 0.;
        if (t2 == 0. && !boson) {
            continue;
        }
        double mj = single_value(masses[j]), g = single_value(gamma[j]);
        double k = single_value(k_constant[j]);
        bool massive_j = mj > 0.;
        // Gamma_j of CDST eqs. 6.27 to 6.29 (no heavy quarks for a gluon)
        double gamma_1 = massive_j ? t2 : g;
        FVal<T> gamma_0 =
            massive_j ? FVal<T>(t2 * (0.5 * log(mj * mj / mu2) - 2.)) : FVal<T>(0.);
        auto p_j = load_mom<T>(momenta[j]);
        for (std::size_t m = 0; m < n; ++m) {
            if (m == j || single_value(casimir[m]) == 0.) {
                continue;
            }
            FVal<T> mk = single_value(masses[m]);
            bool massive_k = single_value(masses[m]) > 0.;
            FVal<T> s;
            bool spectator_decay = decay && m == 0;
            if (spectator_decay) {
                // the final-final dipole with the recoil system K~ = p_a - p_j
                // of the other final-state particles as the spectator, massive
                // unless it is a single massless particle
                FVal<T> pjpa = ldot<T>(p_j, load_mom<T>(momenta[0]));
                double ma = single_value(masses[0]);
                s = 2. * (pjpa - mj * mj);
                massive_k = n > 3 || single_value(masses[j == 1 ? 2 : 1]) > 0.;
                mk = massive_k
                    ? FVal<T>(sqrt(max(ma * ma - 2. * pjpa + mj * mj, 0.)))
                    : FVal<T>(0.);
            } else {
                // (s of a massless pair close to collinear rounds to zero or
                // below at points that the cuts remove, where the Born
                // vanishes)
                s = max(2. * ldot<T>(p_j, load_mom<T>(momenta[m])), 1e-300);
            }
            FVal<T> l = log(mu2 / s);
            FVal<T> v_s[3];
            cs_v_singular<T>(s, mj, mk, massive_j, massive_k, v_s);
            // an incoming emitter has kappa = 2/3 (CDST)
            auto v_ns = cs_v_nonsingular<T>(
                boson,
                s,
                mj,
                mk,
                massive_j,
                massive_k,
                g,
                j < n_in ? 2. / 3. : kappa,
                ca,
                trnf
            );
            FVal<T> removed = (j >= n_in && m >= n_in) || spectator_decay
                ? cs_removed_final_final<T>(
                      boson,
                      s,
                      mj,
                      mk,
                      massive_j,
                      massive_k,
                      alpha_ff,
                      kappa,
                      t2,
                      ca,
                      trnf
                  )
                : FVal<T>(0.);
            // the parts per unit <T_j . T_k> (the soft ones) and per unit
            // <T_j . T_k> / T_j^2 (the collinear ones)
            FVal<T> double_pole = -v_s[0];
            FVal<T> single_pole = -(v_s[1] + v_s[0] * l);
            FVal<T> finite =
                -(v_s[2] + v_s[1] * l + 0.5 * v_s[0] * l * l + v_ns.soft - CS_PI2 / 3.);
            FVal<T> single_pole_collinear = -gamma_1;
            FVal<T> finite_collinear =
                -(gamma_0 + g * l + g + k + v_ns.collinear - removed);
            if (spectator_decay) {
                // the rest of the integral of the decay dipole, per unit
                // <T_j . T_a>, with (mu^2 / s)^eps expanded
                double ma = single_value(masses[0]);
                auto rest = cs_decay_soft<T>(ma, mj, mk, massive_j, massive_k);
                double_pole = double_pole - rest.first;
                single_pole = single_pole - rest.second - rest.first * l;
                finite = finite - rest.third - rest.second * l -
                    0.5 * rest.first * l * l +
                    cs_removed_decay_soft<T>(s, mj, mk, massive_j, massive_k, alpha_ff);
            }
            // in QED the charges of j and m, and the spectators of a photon,
            // which has no charge, each with its weight
            double qq = single_value(charges[j]) * single_value(charges[m]);
            double weight_m = single_value(spectator_weights[m]);
            for (std::size_t b = 0; b < insertion.size(); ++b) {
                FVal<T> born_b = born[b];
                FVal<T> cc =
                    qed ? FVal<T>(qq * born_b) : FVal<T>(color_correlated[b][j][m]);
                FVal<T> ncc = t2 != 0. ? FVal<T>(cc / t2) : FVal<T>(-weight_m * born_b);
                insertion[b] =
                    insertion[b] + a * (cc * finite + ncc * finite_collinear);
                poles[b][0] = poles[b][0] + a * cc * double_pole;
                poles[b][1] =
                    poles[b][1] + a * (cc * single_pole + ncc * single_pole_collinear);
            }
        }
    }
}

// the kinds of a splitting of a parton a of the hadron into the parton a' of
// the Born: q -> q, g -> g, g -> q (a' a quark) and q -> g (a' a gluon), the
// order of P^{aa'} of CS. In QED a charged fermion takes the place of the
// quark and the photon that of the gluon, each kernel with its coefficient c
// (C_F, C_A, T_R and C_F of QCD, Q_f^2, none, N_c Q_f^2 and one per unit Q_f^2
// of the luminosity of the fermions in QED)
constexpr int CS_QQ = 0, CS_GG = 1, CS_GQ = 2, CS_QG = 3;

// P^{aa'}(x) without its plus distribution and delta(1 - x) (CS eq. 10.12),
// with its coefficient c
template <typename T>
KERNELSPEC FVal<T> cs_p_regular(int kind, double c, FVal<T> x) {
    FVal<T> omx = 1. - x;
    switch (kind) {
    case CS_QQ:
        return -c * (1. + x);
    case CS_GG:
        return 2. * c * (omx / x - 1. + x * omx);
    case CS_GQ:
        return c * (x * x + omx * omx);
    default:
        return c * (1. + omx * omx) / x;
    }
}

// P^{aa'}(x) at x < 1 (the four-dimensional Altarelli-Parisi kernel)
template <typename T>
KERNELSPEC FVal<T> cs_p_full(int kind, double c, FVal<T> x) {
    FVal<T> omx = 1. - x;
    switch (kind) {
    case CS_QQ:
        return c * (1. + x * x) / omx;
    case CS_GG:
        return 2. * c * (x / omx + omx / x + x * omx);
    case CS_GQ:
        return c * (x * x + omx * omx);
    default:
        return c * (1. + omx * omx) / x;
    }
}

// What alpha_fi removes from the final-state emitter j (a quark of mass m and
// Casimir t2, or a massless gluon with self coupling ca and gamma_j) with the
// incoming spectator: the dipoles over 1 - x > alpha, at x per unit <T_j .
// T_a> / T_j^2, with the invariant s = 2 p_j p_a of the real emission
template <typename T>
KERNELSPEC FVal<T> cs_removed_final_initial(
    bool boson,
    FVal<T> x,
    FVal<T> s,
    double m,
    double alpha,
    double t2,
    double ca,
    double gamma_j
) {
    FVal<T> omx = 1. - x;
    FVal<T> out;
    if (boson) {
        out = (2. * ca * log((1. + omx) / omx) - gamma_j) / omx;
    } else {
        FVal<T> mu2 = m * m / s, zp = omx / (omx + mu2);
        out = t2 / omx *
            (2. * log((omx + zp) / omx) - 2. * zp + 0.5 * zp * zp -
             2. * mu2 * zp / omx);
    }
    return where(omx > alpha, out, FVal<T>(0.));
}

// What alpha_if removes from the incoming emitter a with the final-state
// spectator j of mass m: the dipoles over u > alpha, at x per unit <T_j . T_a>
// / T_a^2, with s = 2 p_j p_a of the real emission, the kernel with its
// coefficient c
template <typename T>
KERNELSPEC FVal<T> cs_removed_initial_final(
    int kind, double c, FVal<T> x, FVal<T> s, double m, double alpha
) {
    FVal<T> omx = 1. - x, mu2 = m * m / s, zp = omx / (omx + mu2);
    FVal<T> l_u = log(zp / alpha);
    FVal<T> e = log(zp * (omx + alpha) / (alpha * (omx + zp))) / omx;
    FVal<T> l_m =
        m > 0. ? FVal<T>(mu2 * log((1. - alpha) * (omx + mu2) / mu2) / x) : FVal<T>(0.);
    FVal<T> out;
    switch (kind) {
    case CS_QQ:
        out = c * (2. * e - (1. + x) * l_u);
        break;
    case CS_GQ:
        out = c * (x * x + omx * omx) * l_u;
        break;
    case CS_QG:
        out = c * ((x + 2. * omx / x) * l_u - 2. * l_m);
        break;
    default:
        out = 2. * c * (e + (-1. + x * omx + omx / x) * l_u - l_m);
    }
    return where(zp > alpha, out, FVal<T>(0.));
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
// coefficient coef = <T_j . T_a'> B and per normalized coefficient norm =
// <T_j . T_a'> B / T_a'^2 (the same with the weight of j for a photon a'),
// with s = 2 p_j p~_a' of the Born (s / z inside the convolution), the kernel
// with its coefficient c and the self coupling ca of a'
template <typename T>
KERNELSPEC void cs_massive_spectator(
    CSConvolution<T>& conv,
    int kind,
    double c,
    FVal<T> coef,
    FVal<T> norm,
    double ca,
    double gamma_a,
    FVal<T> s,
    double m
) {
    FVal<T> z = conv.z, omz = 1. - z;
    double m2 = m * m;
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
            2. * (log(mu2) * l_eta + dilog<T>(-1. / mu2) - dilog<T>(-u / mu2));
        conv.plus(-coef, f_z, f_end, integral);
        conv.plus_times(-2. * coef, log(2. + mu2_z - z), log(1. + mu2));
        conv.plus_times(-2. * coef, log((2. - z) / (2. - z + mu2_z)), -log(1. + mu2));
        conv.delta(-coef, -1.5 + mu2 * log(mu2 / (1. + mu2)) + 0.5 * mu2 / (1. + mu2));
        FVal<T> rq = sqrt(s + m2);
        conv.delta(
            -norm * gamma_a, log((s - 2. * m * rq + 2. * m2) / s) + 2. * m / (rq + m)
        );
    }
    if (kind == CS_QG || kind == CS_GG) {
        // 2 C_F mu^2 / z log(mu^2 / (1 - z + mu^2)) for q -> g, 2 C_A for g -> g,
        // per unit <T_j . T_a'> / T_a'^2
        double factor = kind == CS_GG ? ca : c;
        conv.regular(-norm, 2. * factor * mu2_z / z * log(mu2_z / (omz + mu2_z)));
    }
    conv.regular(
        -norm, cs_p_regular<T>(kind, c, z) * log(omz * s_z / (omz * s_z + m2))
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
    FSIn<T, 1> casimir,
    FSIn<T, 1> gamma,
    FSIn<T, 1> self_coupling,
    FSIn<T, 1> pair_splitting,
    FSIn<T, 1> other_coefficient,
    FSIn<T, 1> charges,
    FSIn<T, 1> spectator_weights,
    FSIn<T, 1> masses,
    FSIn<T, 1> options,
    FOut<T, 1> z,
    FOut<T, 3> weights
) {
    constexpr double pi2 = CS_PI2;
    double alpha_fi = single_value(options[0]), alpha_if = single_value(options[1]);
    double alpha_ii = single_value(options[2]);
    bool qed = single_value(options[4]) != 0.;
    FVal<T> a = FVal<T>(alpha_s) / (2. * 3.14159265358979323846);
    std::size_t n = momenta.size();
    // <T_j . T_k> B of QCD, or Q_j Q_k B of the charges in QED, and the same
    // over T_j^2, with the weight of k for a photon j
    auto correlated = [&](std::size_t b, std::size_t j, std::size_t k) -> FVal<T> {
        return qed
            ? FVal<T>(
                  single_value(charges[j]) * single_value(charges[k]) * FVal<T>(born[b])
              )
            : FVal<T>(color_correlated[b][j][k]);
    };
    auto normalized = [&](std::size_t b, std::size_t j, std::size_t k) -> FVal<T> {
        double t2 = single_value(casimir[j]);
        return t2 != 0.
            ? FVal<T>(correlated(b, j, k) / t2)
            : FVal<T>(-single_value(spectator_weights[k]) * FVal<T>(born[b]));
    };
    for (std::size_t l = 0; l < 2; ++l) {
        FVal<T> eta = l == 0 ? FVal<T>(x1) : FVal<T>(x2);
        FVal<T> mu_f = l == 0 ? FVal<T>(fact_scale1) : FVal<T>(fact_scale2);
        FVal<T> rv = r, zl = eta + (1. - eta) * rv;
        z[l] = zl;
        double t2 = single_value(casimir[l]);
        double ca = single_value(self_coupling[l]),
               trnf = single_value(pair_splitting[l]);
        double c_other = single_value(other_coefficient[l]);
        double gamma_l = single_value(gamma[l]);
        for (std::size_t b = 0; b < weights.size(); ++b) {
            for (std::size_t c = 0; c < 3; ++c) {
                weights[b][l][c] = 0.;
            }
        }
        // a gluon, or a photon, which has no T^2 but takes the fermions
        bool boson = ca != 0. || trnf != 0.;
        if (t2 == 0. && !boson) {
            continue;
        }
        int same = boson ? CS_GG : CS_QQ, other = boson ? CS_QG : CS_GQ;
        // the coefficients of the kernels of a' from the same parton (the
        // self coupling of a boson, T^2 of a fermion) and from the other one
        double c_same = boson ? ca : t2;
        FVal<T> omz = 1. - zl, l_omz = log(omz), l_eta = log(1. - eta);
        FVal<T> l_ratio = log(omz / zl);
        // what alpha_ii removes, from 1 - z > alpha_ii
        auto removed_ii = omz > alpha_ii;
        FVal<T> l_alpha = log(omz / alpha_ii);
        auto p_l = load_mom<T>(momenta[l]);
        std::size_t o = 1 - l;
        for (std::size_t b = 0; b < weights.size(); ++b) {
            CSConvolution<T> conv_same{zl, 1. - eta, l_eta, 0., 0.};
            CSConvolution<T> conv_other{zl, 1. - eta, l_eta, 0., 0.};
            FVal<T> born_b = born[b];
            // P: (1 / T_a'^2) sum_j <T_j . T_a'> log(mu_F^2 / s_ja') P^{aa'}
            FVal<T> coef_p = 0.;
            for (std::size_t j = 0; j < n; ++j) {
                if (j == l || single_value(casimir[j]) == 0.) {
                    continue;
                }
                FVal<T> s = max(2. * ldot<T>(p_l, load_mom<T>(momenta[j])), 1e-300);
                coef_p = coef_p + normalized(b, l, j) * log(mu_f * mu_f / s);
            }
            conv_same.regular(coef_p, cs_p_regular<T>(same, c_same, zl));
            conv_same.plus(2. * coef_p * t2, 1. / omz, 1. / omz, -l_eta);
            conv_same.delta(coef_p, boson ? gamma_l : 1.5 * t2);
            conv_other.regular(coef_p, cs_p_regular<T>(other, c_other, zl));
            // Kbar of CS eqs. 10.27-10.28 times the Born
            FVal<T> int_ratio = -0.5 * l_eta * l_eta - dilog<T>(1. - eta) + pi2 / 6.;
            if (boson) {
                conv_same.regular(born_b, cs_p_regular<T>(CS_GG, ca, zl) * l_ratio);
                conv_same.plus(
                    2. * ca * born_b, l_ratio / omz, l_ratio / omz, int_ratio
                );
                conv_same.delta(-born_b, (50. / 9. - pi2) * ca - 16. / 9. * trnf);
                conv_other.regular(
                    born_b, cs_p_regular<T>(CS_QG, c_other, zl) * l_ratio + c_other * zl
                );
            } else {
                conv_same.regular(born_b, t2 * (-(1. + zl) * l_ratio + omz));
                conv_same.plus(
                    2. * t2 * born_b, l_ratio / omz, l_ratio / omz, int_ratio
                );
                conv_same.delta(-born_b, t2 * (5. - pi2));
                conv_other.regular(
                    born_b,
                    cs_p_regular<T>(CS_GQ, c_other, zl) * l_ratio +
                        2. * c_other * zl * omz
                );
            }
            // Ktilde of CS eq. 10.29 with the other incoming parton, and what
            // alpha_ii removes
            if (single_value(casimir[o]) != 0.) {
                FVal<T> coef = -normalized(b, l, o);
                conv_same.regular(coef, cs_p_regular<T>(same, c_same, zl) * l_omz);
                conv_same.plus(
                    2. * t2 * coef, l_omz / omz, l_omz / omz, -0.5 * l_eta * l_eta
                );
                conv_same.delta(coef, -t2 * pi2 / 3.);
                conv_other.regular(coef, cs_p_regular<T>(other, c_other, zl) * l_omz);
                conv_same.regular(
                    -coef,
                    where(
                        removed_ii,
                        cs_p_full<T>(same, c_same, zl) * l_alpha,
                        FVal<T>(0.)
                    )
                );
                conv_other.regular(
                    -coef,
                    where(
                        removed_ii,
                        cs_p_full<T>(other, c_other, zl) * l_alpha,
                        FVal<T>(0.)
                    )
                );
            }
            // the final-state partons j as emitters with the spectator a' and as
            // spectators of a'
            for (std::size_t j = 2; j < n; ++j) {
                double cj = single_value(casimir[j]);
                double ca_j = single_value(self_coupling[j]);
                double trnf_j = single_value(pair_splitting[j]);
                double gamma_j = single_value(gamma[j]);
                bool boson_j = ca_j != 0. || trnf_j != 0.;
                if (cj == 0. && !boson_j) {
                    continue;
                }
                double mj = single_value(masses[j]);
                FVal<T> s = max(2. * ldot<T>(p_l, load_mom<T>(momenta[j])), 1e-300);
                // a photon j emits with the spectator a', but is none of a'
                if (cj != 0.) {
                    FVal<T> cc = correlated(b, j, l);
                    FVal<T> norm = normalized(b, l, j);
                    if (mj == 0.) {
                        FVal<T> coef = normalized(b, j, l) * gamma_j;
                        conv_same.plus(coef, 1. / omz, 1. / omz, -l_eta);
                        conv_same.delta(coef, 1.);
                    } else {
                        cs_massive_spectator<T>(
                            conv_same, same, c_same, cc, norm, ca, gamma_l, s, mj
                        );
                        cs_massive_spectator<T>(
                            conv_other, other, c_other, cc, norm, ca, gamma_l, s, mj
                        );
                    }
                    conv_same.regular(
                        norm,
                        cs_removed_initial_final<T>(
                            same, c_same, zl, s / zl, mj, alpha_if
                        )
                    );
                    conv_other.regular(
                        norm,
                        cs_removed_initial_final<T>(
                            other, c_other, zl, s / zl, mj, alpha_if
                        )
                    );
                } else {
                    FVal<T> coef = normalized(b, j, l) * gamma_j;
                    conv_same.plus(coef, 1. / omz, 1. / omz, -l_eta);
                    conv_same.delta(coef, 1.);
                }
                conv_same.regular(
                    normalized(b, j, l),
                    cs_removed_final_initial<T>(
                        boson_j, zl, s / zl, mj, alpha_fi, cj, ca_j, gamma_j
                    )
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

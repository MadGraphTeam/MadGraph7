#pragma once

#include "definitions.h"

namespace madevent {
namespace kernels {

constexpr double INV_GEV2_TO_PB = 0.38937937217186e9;

// Helper functions

template <typename T>
struct FourMom {
    KERNELSPEC FVal<T>& operator[](int i) { return p[i]; }
    KERNELSPEC FVal<T>& operator[](int i) const { return p[i]; }
    FVal<T> p[4];
};

template <typename F, typename S>
struct Pair {
    F first;
    S second;
};

template <typename A, typename B, typename C>
struct Triplet {
    A first;
    B second;
    C third;
};

template <typename T>
KERNELSPEC FourMom<T> load_mom(FIn<T, 1> p) {
    return {p[0], p[1], p[2], p[3]};
}

template <typename T>
KERNELSPEC void store_mom(FOut<T, 1> p_to, FourMom<T> p_from) {
    for (int i = 0; i < 4; ++i) {
        p_to[i] = p_from[i];
    }
}

template <typename T>
KERNELSPEC FVal<T> kaellen(FVal<T> x, FVal<T> y, FVal<T> z) {
    auto xyz = x - y - z;
    return xyz * xyz - 4 * y * z;
}

template <typename T>
KERNELSPEC FVal<T> bk_V(
    FVal<T> m0_2,
    FVal<T> ma_2,
    FVal<T> mb_2,
    FVal<T> m1_2,
    FVal<T> m2_2,
    FVal<T> m3_2,
    FVal<T> t1_abs,
    FVal<T> t2,
    FVal<T> s12
) {
    // Determinant of the 3x3 V-matrix,
    // see Eq.(11) in 10.1103/PhysRev.187.2008.
    // Note: expects the absolute value of t1
    auto a11 = 2.0 * s12;
    auto a12 = ma_2 + s12 - t2;
    auto a13 = s12 + m1_2 - m2_2;
    auto a22 = 2.0 * ma_2;
    auto a23 = ma_2 + m1_2 + t1_abs;
    auto a31 = m0_2 + s12 - m3_2;
    auto a32 = m0_2 + ma_2 - mb_2;

    // Computes the determinant of the 3x3 V-matrix (hard-coded because easier)
    auto det = a12 * a23 * a31 + a12 * a13 * a32 - a11 * a23 * a32 - a13 * a22 * a31;
    return -det / 8.0;
}

template <typename T>
KERNELSPEC FVal<T> bk_gram4(
    FVal<T> m0_2,
    FVal<T> ma_2,
    FVal<T> mb_2,
    FVal<T> m1_2,
    FVal<T> m2_2,
    FVal<T> m3_2,
    FVal<T> t1_abs,
    FVal<T> t2,
    FVal<T> s12,
    FVal<T> s23
) {
    // omputes the 4x4 Gram determinant,
    // see Eq.(B6) in 10.1103/PhysRev.187.2008.
    // Note: expects the absolute value of t1

    // Get upper triangular matrix components which are non-zero
    // as the Gram matrix is symmetric, i.e. (a_{ij} = a_{ji})
    auto a11 = 2.0 * ma_2;
    auto a12 = ma_2 - t1_abs - m1_2;
    auto a13 = ma_2 + t2 - s12;
    auto a14 = ma_2 + mb_2 - m0_2;
    auto a22 = -2.0 * t1_abs;
    auto a23 = t2 - t1_abs - m2_2;
    auto a24 = mb_2 - t1_abs - s23;
    auto a33 = 2.0 * t2;
    auto a34 = t2 + mb_2 - m3_2;
    auto a44 = 2.0 * mb_2;

    // Computes the determinant of the 4x4 Gram matrix (hard-coded because easier)
    auto det = a14 * a23 * a14 * a23 + a13 * a24 * a13 * a24 + a12 * a34 * a12 * a34 -
        a14 * a14 * a22 * a33 - a13 * a13 * a22 * a44 - a12 * a12 * a33 * a44 -
        a23 * a23 * a11 * a44 - a24 * a24 * a11 * a33 - a34 * a34 * a11 * a22 +
        2 * a11 * a23 * a24 * a34 + 2 * a12 * a13 * a23 * a44 +
        2 * a12 * a14 * a24 * a33 + 2 * a13 * a14 * a22 * a34 -
        2 * a12 * a13 * a24 * a34 - 2 * a12 * a14 * a23 * a34 -
        2 * a13 * a14 * a23 * a24 + a11 * a22 * a33 * a44;
    return det / 16.0;
}

template <typename T>
KERNELSPEC FVal<T> bk_sqrt_g3i_g3im1(
    FVal<T> m0_2,
    FVal<T> ma_2,
    FVal<T> mb_2,
    FVal<T> m1_2,
    FVal<T> m2_2,
    FVal<T> m3_2,
    FVal<T> t1_abs,
    FVal<T> t2,
    FVal<T> s12
) {
    // This is the squaet root of the product of the two 3x3 Gram determinants g3i and
    // g3im1, as in Eq.(11) in 10.1103/PhysRev.187.2008. Note: expects the absolute
    // value of t1
    auto a11 = 2 * s12;
    auto a12 = s12 + ma_2 - t2;
    auto a13 = s12 + m0_2 - m3_2;
    auto b13 = s12 + m1_2 - m2_2;
    auto a22 = 2 * ma_2;
    auto a23 = ma_2 + m0_2 - mb_2;
    auto b23 = ma_2 + m1_2 + t1_abs;
    auto a33 = 2 * m0_2;
    auto b33 = 2 * m1_2;

    // Calculate the two gramm determinants g3i and g3im1
    // (ard-coded because easier)
    auto g3i = a11 * a22 * a33 + 2 * a12 * a23 * a13 - a11 * a23 * a23 -
        a22 * a13 * a13 - a33 * a12 * a12;
    auto g3im1 = a11 * a22 * b33 + 2 * a12 * b23 * b13 - a11 * b23 * b23 -
        a22 * b13 * b13 - b33 * a12 * a12;
    return sqrt(g3i * g3im1) / 8.0;
}

template <typename T>
KERNELSPEC FVal<T> lsquare(FourMom<T> p) {
    return p[0] * p[0] - p[1] * p[1] - p[2] * p[2] - p[3] * p[3];
}

template <typename T>
KERNELSPEC FourMom<T> rotate(FourMom<T> p, FourMom<T> q) {
    // this function is based on the rotxxx subroutine from HELAS
    // used in MG5 (aloha_functions.f)
    auto qt2 = q[1] * q[1] + q[2] * q[2];
    auto qq = sqrt(qt2 + q[3] * q[3]);
    auto qt = sqrt(qt2);
    return {
        p[0],
        q[1] * q[3] / qq / qt * p[1] - q[2] / qt * p[2] + q[1] / qq * p[3],
        q[2] * q[3] / qq / qt * p[1] + q[1] / qt * p[2] + q[2] / qq * p[3],
        -qt / qq * p[1] + q[3] / qq * p[3]
    };
}

template <typename T>
KERNELSPEC FourMom<T> boost(FourMom<T> k, FourMom<T> p_boost, FVal<T> sign) {
    // Perform the boost
    // This is in fact a numerically more stable implementation than often used
    auto p2_boost = lsquare<T>(p_boost);
    auto rsq = sqrt(max(EPS2, p2_boost));
    auto k_dot_p = k[1] * p_boost[1] + k[2] * p_boost[2] + k[3] * p_boost[3];
    auto e = (k[0] * p_boost[0] + sign * k_dot_p) / rsq;
    auto c1 = sign * (k[0] + e) / (rsq + p_boost[0]);
    return FourMom<T>{
        e, k[1] + c1 * p_boost[1], k[2] + c1 * p_boost[2], k[3] + c1 * p_boost[3]
    };
}

template <typename T>
KERNELSPEC void
boost_beam(FIn<T, 2> q, FVal<T> x1, FVal<T> x2, FVal<T> sign, FOut<T, 2> p_out) {
    auto exp_rap = sqrt(x1 / x2);
    auto exp_rap_inv = 1. / exp_rap;
    auto cosh_rap = 0.5 * (exp_rap + exp_rap_inv);
    auto sinh_rap = 0.5 * (exp_rap - exp_rap_inv);
    for (std::size_t i = 0; i < q.size(); ++i) {
        auto q_i = q[i];
        auto p_out_i = p_out[i];
        p_out_i[0] = q_i[0] * cosh_rap + sign * q_i[3] * sinh_rap;
        p_out_i[1] = q_i[1];
        p_out_i[2] = q_i[2];
        p_out_i[3] = q_i[3] * cosh_rap + sign * q_i[0] * sinh_rap;
    }
}

template <typename T>
KERNELSPEC Pair<FourMom<T>, FVal<T>>
two_body_decay(FVal<T> r_phi, FVal<T> r_cos_theta, FVal<T> m0, FVal<T> m1, FVal<T> m2) {
    auto phi = PI * (2. * r_phi - 1.);
    auto cos_theta = 2. * r_cos_theta - 1.;
    auto m0_clip = max(m0, EPS);

    // this part is based on the mom2cx subroutine from HELAS
    // used in MG5 (aloha_functions.f)
    auto ed = (m1 - m2) * (m1 + m2) / m0_clip;
    auto pp2 = ed * ed - 2. * (m1 * m1 + m2 * m2) + m0 * m0;
    auto pp = 0.5 * where(m1 * m2 == 0., m0 - fabs(ed), sqrt(max(pp2, EPS)));
    auto sin_theta = sqrt((1. - cos_theta) * (1 + cos_theta));
    auto e1 = 0.5 * (m0 + ed);
    FourMom<T> p1{
        max(e1, 0.),
        pp * sin_theta * cos(phi),
        pp * sin_theta * sin(phi),
        pp * cos_theta
    };

    auto det = PI * pp / m0_clip;
    return {p1, det};
}

template <typename T>
KERNELSPEC Pair<FourMom<T>, FVal<T>> p1com_from_tabs_phi(
    FourMom<T> pa_com,
    FVal<T> s_tot,
    FVal<T> phi,
    FVal<T> t_abs,
    FVal<T> m1,
    FVal<T> m2,
    FVal<T> ma_2,
    FVal<T> mb_2
) {
    // this function is based on the gentcms subroutine in MG5 (genps.f)
    // Note: expects t_abs (positive t-ivariant)
    auto m_tot = sqrt(s_tot);

    auto ed = (m1 - m2) * (m1 + m2) / m_tot;
    auto pp2 = ed * ed - 2. * (m1 * m1 + m2 * m2) + s_tot;
    auto pp = 0.5 * where(m1 * m2 == 0., m_tot - fabs(ed), sqrt(max(pp2, EPS)));

    auto pa_com_mag =
        sqrt(pa_com[1] * pa_com[1] + pa_com[2] * pa_com[2] + pa_com[3] * pa_com[3]);
    FourMom<T> p1_com;
    auto e1_com = 0.5 * (m_tot + ed);
    p1_com[0] = max(e1_com, 0.);
    p1_com[3] =
        -(m1 * m1 + ma_2 + t_abs - 2. * p1_com[0] * pa_com[0]) / (2.0 * pa_com_mag);
    auto pt2 = pp * pp - p1_com[3] * p1_com[3];
    auto pt = sqrt(max(pt2, 0.));
    p1_com[1] = pt * cos(phi);
    p1_com[2] = pt * sin(phi);

    auto det = PI / (2. * sqrt(kaellen<T>(s_tot, ma_2, mb_2)));
    return {p1_com, det};
}

template <typename T>
KERNELSPEC Triplet<FourMom<T>, FourMom<T>, FVal<T>> three_body_decay(
    FVal<T> r_e1,
    FVal<T> r_e2,
    FVal<T> r_phi,
    FVal<T> r_cos_theta,
    FVal<T> r_beta,
    FVal<T> m0,
    FVal<T> m1,
    FVal<T> m2,
    FVal<T> m3
) {
    // this is based on section G.3 in
    // https://inspirehep.net/literature/1784296

    // define angles and determinants
    auto phi = PI * (2. * r_phi - 1.);
    auto cos_theta = 2. * r_cos_theta - 1.;
    auto beta = PI * (2. * r_beta - 1.);
    auto det_omega = 8 * PI * PI;

    // Define mass squares
    auto m1sq = m1 * m1;
    auto m2sq = m2 * m2;
    auto m3sq = m3 * m3;

    // define energy E1
    auto E1_max = m0 / 2 + (m1sq - (m2 + m3) * (m2 + m3)) / (2 * m0);
    auto E1 = m1 + (E1_max - m1) * r_e1;
    auto det_E1 = E1_max - m1;

    // get boundaries
    auto Delta = 2 * m0 * (m0 / 2 - E1) + m1sq;
    auto Delta23 = m2sq - m3sq;
    auto dE2 =
        (E1 * E1 - m1sq) * ((Delta + Delta23) * (Delta + Delta23) - 4 * m2sq * Delta);
    auto E2a = 1 / (2 * Delta) * ((m0 - E1) * (Delta + Delta23) - sqrt(dE2));
    auto E2b = 1 / (2 * Delta) * ((m0 - E1) * (Delta + Delta23) + sqrt(dE2));
    auto E2_min = min(E2a, E2b);
    auto E2_max = max(E2a, E2b);
    auto E2 = E2_min + (E2_max - E2_min) * r_e2;
    auto det_E2 = E2_max - E2_min;

    // calculate abs momentas
    auto pp1s = E1 * E1 - m1sq;
    auto pp1 = where(m1sq == 0, E1, sqrt(max(pp1s, EPS)));
    auto pp2s = E2 * E2 - m2sq;
    auto pp2 = where(m2sq == 0, E2, sqrt(max(pp2s, EPS)));

    // calculate cosalpha
    auto num_alpha_1 = 2 * m0 * (m0 / 2 - E1 - E2);
    auto num_alpha_2 = m1sq + m2sq + 2 * E1 * E2 - m3sq;
    auto denom_alpha = 2 * pp1 * pp2;
    auto cos_alpha = (num_alpha_1 + num_alpha_2) / denom_alpha;

    // build momenta p1
    auto sin_theta = sqrt((1. - cos_theta) * (1 + cos_theta));
    FourMom<T> p1{
        max(E1, 0.),
        pp1 * sin_theta * cos(phi),
        pp1 * sin_theta * sin(phi),
        pp1 * cos_theta
    };

    // build momenta p2
    auto sin_alpha = sqrt((1. - cos_alpha) * (1 + cos_alpha));
    FourMom<T> p2{
        max(E2, 0.),
        pp2 *
            (sin_alpha * cos(beta) * cos_theta * cos(phi) +
             cos_alpha * sin_theta * cos(phi) - sin_alpha * sin(beta) * sin(phi)),
        pp2 *
            (sin_alpha * sin(beta) * cos(phi) +
             sin_alpha * cos(beta) * cos_theta * sin(phi) +
             cos_alpha * sin_theta * sin(phi)),
        pp2 * (cos_alpha * cos_theta - sin_alpha * cos(beta) * sin_theta)
    };

    auto det = det_omega * det_E1 * det_E2 / 8;
    return {p1, p2, det};
}

template <typename T>
KERNELSPEC FVal<T> get_phi_from_s23(
    IVal<T> phi_choice,
    FVal<T> m0_2,
    FVal<T> s23,
    FVal<T> s12,
    FVal<T> t1_abs,
    FVal<T> t2,
    FVal<T> ma_2,
    FVal<T> mb_2,
    FVal<T> m1_2,
    FVal<T> m2_2,
    FVal<T> m3_2
) {
    // Computes the azimuthal angle phi from s23, using the formulae
    // in appendix B of 10.1103/PhysRev.187.2008
    auto sqrtGG =
        bk_sqrt_g3i_g3im1<T>(m0_2, ma_2, mb_2, m1_2, m2_2, m3_2, t1_abs, t2, s12);
    auto V = bk_V<T>(m0_2, ma_2, mb_2, m1_2, m2_2, m3_2, t1_abs, t2, s12);
    auto lambda = kaellen<T>(s12, ma_2, t2);
    auto cos_phi = (lambda * (s23 - m0_2 - m1_2) - 8 * V) / (8 * sqrtGG);
    auto phi = where(phi_choice == 1, (-acos(cos_phi) + 2. * PI), acos(cos_phi));
    return phi;
}

// Kernels

template <typename T>
KERNELSPEC void
kernel_boost_beam(FIn<T, 2> p1, FIn<T, 0> x1, FIn<T, 0> x2, FOut<T, 2> p_out) {
    boost_beam<T>(p1, x1, x2, 1.0, p_out);
}

template <typename T>
KERNELSPEC void
kernel_boost_beam_inverse(FIn<T, 2> p1, FIn<T, 0> x1, FIn<T, 0> x2, FOut<T, 2> p_out) {
    boost_beam<T>(p1, x1, x2, -1.0, p_out);
}

template <typename T>
KERNELSPEC void kernel_com_p_in(FIn<T, 0> e_cm, FOut<T, 1> p1, FOut<T, 1> p2) {
    auto p_com = e_cm / 2;
    p1[0] = p_com;
    p1[1] = 0;
    p1[2] = 0;
    p1[3] = p_com;
    p2[0] = p_com;
    p2[1] = 0;
    p2[2] = 0;
    p2[3] = -p_com;
}

template <typename T>
KERNELSPEC void kernel_r_to_x1x2(
    FIn<T, 0> r,
    FIn<T, 0> s_hat,
    FIn<T, 0> s_lab,
    FOut<T, 0> x1,
    FOut<T, 0> x2,
    FOut<T, 0> det
) {
    auto tau = s_hat / s_lab;
    x1 = pow(tau, r);
    x2 = pow(tau, (1 - r));
    det = fabs(log(tau)) / s_lab;
}

template <typename T>
KERNELSPEC void kernel_x1x2_to_r(
    FIn<T, 0> x1, FIn<T, 0> x2, FIn<T, 0> s_lab, FOut<T, 0> r, FOut<T, 0> det
) {
    auto tau = x1 * x2;
    auto log_tau = log(tau);
    r = log(x1) / log_tau;
    det = fabs(1 / log_tau) * s_lab;
}

template <typename T>
KERNELSPEC void kernel_diff_cross_section(
    FIn<T, 0> x1,
    FIn<T, 0> x2,
    FIn<T, 0> pdf1,
    FIn<T, 0> pdf2,
    FIn<T, 0> matrix_element,
    FIn<T, 0> e_cm2,
    FOut<T, 0> result
) {
    result = INV_GEV2_TO_PB * matrix_element * pdf1 * pdf2 /
        (2. * e_cm2 * x1 * x1 * x2 * x2);
}

template <typename T>
KERNELSPEC void kernel_two_body_decay_com(
    FIn<T, 0> r_phi,
    FIn<T, 0> r_cos_theta,
    FIn<T, 0> m0,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FOut<T, 1> p1,
    FOut<T, 1> p2,
    FOut<T, 0> det
) {
    auto decay_out = two_body_decay<T>(r_phi, r_cos_theta, m0, m1, m2);
    auto p1_tmp = decay_out.first;
    auto det_tmp = decay_out.second;
    store_mom<T>(p1, p1_tmp);
    det = det_tmp;
    auto e2 = m0 - p1_tmp[0];
    p2[0] = max(e2, 0.);
    p2[1] = -p1_tmp[1];
    p2[2] = -p1_tmp[2];
    p2[3] = -p1_tmp[3];
}

template <typename T>
KERNELSPEC void kernel_two_body_decay(
    FIn<T, 0> r_phi,
    FIn<T, 0> r_cos_theta,
    FIn<T, 0> m0,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FIn<T, 1> p0,
    FOut<T, 1> p1,
    FOut<T, 1> p2,
    FOut<T, 0> det
) {
    auto decay_out = two_body_decay<T>(r_phi, r_cos_theta, m0, m1, m2);
    auto p1_tmp = decay_out.first;
    auto det_tmp = decay_out.second;
    store_mom<T>(p1, boost<T>(p1_tmp, load_mom<T>(p0), 1.));
    det = det_tmp;
    auto e2 = p0[0] - p1[0];
    p2[0] = max(e2, 0.);
    p2[1] = p0[1] - p1[1];
    p2[2] = p0[2] - p1[2];
    p2[3] = p0[3] - p1[3];
}

template <typename T>
KERNELSPEC void kernel_two_to_two_particle_scattering_com(
    FIn<T, 0> r_phi,
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 0> t_abs,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FOut<T, 1> p1,
    FOut<T, 1> p2,
    FOut<T, 0> det
) {
    FourMom<T> p_tot;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
    }
    auto s_tot = lsquare<T>(p_tot);
    auto ma_2 = lsquare<T>(load_mom<T>(pa)), mb_2 = lsquare<T>(load_mom<T>(pb));
    auto phi = PI * (2. * r_phi - 1.);
    auto scatter_out =
        p1com_from_tabs_phi<T>(load_mom<T>(pa), s_tot, phi, t_abs, m1, m2, ma_2, mb_2);
    auto p1_com = scatter_out.first;
    auto det_tmp = scatter_out.second;
    store_mom<T>(p1, p1_com);
    for (int i = 0; i < 4; ++i) {
        p2[i] = p_tot[i] - p1_com[i];
    }
    det = det_tmp;
}

template <typename T>
KERNELSPEC void kernel_two_to_two_particle_scattering(
    FIn<T, 0> r_phi,
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 0> t_abs,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FOut<T, 1> p1,
    FOut<T, 1> p2,
    FOut<T, 0> det
) {
    FourMom<T> p_tot;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
    }
    auto pa_com = boost<T>(load_mom<T>(pa), p_tot, -1.);
    auto s_tot = lsquare<T>(p_tot);
    auto ma_2 = lsquare<T>(load_mom<T>(pa)), mb_2 = lsquare<T>(load_mom<T>(pb));
    auto phi = PI * (2. * r_phi - 1.);
    auto scatter_out =
        p1com_from_tabs_phi<T>(pa_com, s_tot, phi, t_abs, m1, m2, ma_2, mb_2);
    auto p1_com = scatter_out.first;
    auto det_tmp = scatter_out.second;
    auto p1_rot = rotate<T>(p1_com, pa_com);
    auto p1_lab = boost<T>(p1_rot, p_tot, 1.);
    store_mom<T>(p1, p1_lab);
    for (int i = 0; i < 4; ++i) {
        p2[i] = p_tot[i] - p1_lab[i];
    }
    det = det_tmp;
}

template <typename T>
KERNELSPEC void kernel_two_to_three_particle_scattering(
    IIn<T, 0> phi_index,
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 1> p3,
    FIn<T, 0> s23,
    FIn<T, 0> t1_abs,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FOut<T, 1> p1,
    FOut<T, 1> p2,
    FOut<T, 0> det
) {
    FourMom<T> p_12, p_c, p_tot;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
    }
    for (int i = 0; i < 4; ++i) {
        p_12[i] = pa[i] + pb[i] - p3[i];
    }
    for (int i = 0; i < 4; ++i) {
        p_c[i] = pb[i] - p3[i];
    }
    auto pa_com = boost<T>(load_mom<T>(pa), p_12, -1.);
    auto ma_2 = lsquare<T>(load_mom<T>(pa));
    auto mb_2 = lsquare<T>(load_mom<T>(pb));
    auto m3_2 = lsquare<T>(load_mom<T>(p3));
    auto m0_2 = lsquare<T>(p_tot);
    auto s12 = lsquare<T>(p_12);
    auto t2 = lsquare<T>(p_c);
    auto m1_2 = m1 * m1;
    auto m2_2 = m2 * m2;

    auto phi = get_phi_from_s23<T>(
        phi_index, m0_2, s23, s12, t1_abs, t2, ma_2, mb_2, m1_2, m2_2, m3_2
    );

    auto scatter_out =
        p1com_from_tabs_phi<T>(pa_com, s12, phi, t1_abs, m1, m2, ma_2, t2);
    auto p1_com = scatter_out.first;
    auto gram4 = bk_gram4<T>(m0_2, ma_2, mb_2, m1_2, m2_2, m3_2, t1_abs, t2, s12, s23);
    auto det_2to3 = 1 / (8 * sqrt(max(-gram4, EPS)));
    auto p1_rot = rotate<T>(p1_com, pa_com);
    auto p1_lab = boost<T>(p1_rot, p_12, 1.);
    store_mom<T>(p1, p1_lab);
    for (int i = 0; i < 4; ++i) {
        p2[i] = p_12[i] - p1_lab[i];
    }
    det = det_2to3 / 2; // factor 1/2 as acos allows for two choices of phi
}

template <typename T>
KERNELSPEC void kernel_three_body_decay_com(
    FIn<T, 0> r_e1,
    FIn<T, 0> r_e2,
    FIn<T, 0> r_phi,
    FIn<T, 0> r_cos_theta,
    FIn<T, 0> r_beta,
    FIn<T, 0> m0,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FIn<T, 0> m3,
    FOut<T, 1> p1,
    FOut<T, 1> p2,
    FOut<T, 1> p3,
    FOut<T, 0> det
) {
    auto decay_out =
        three_body_decay<T>(r_e1, r_e2, r_phi, r_cos_theta, r_beta, m0, m1, m2, m3);
    auto p1_tmp = decay_out.first;
    auto p2_tmp = decay_out.second;
    auto det_tmp = decay_out.third;
    store_mom<T>(p1, p1_tmp);
    store_mom<T>(p2, p2_tmp);
    det = det_tmp;
    auto e3 = m0 - p1_tmp[0] - p2_tmp[0];
    p3[0] = max(e3, 0.);
    p3[1] = -p1_tmp[1] - p2_tmp[1];
    p3[2] = -p1_tmp[2] - p2_tmp[2];
    p3[3] = -p1_tmp[3] - p2_tmp[3];
}

template <typename T>
KERNELSPEC void kernel_three_body_decay(
    FIn<T, 0> r_e1,
    FIn<T, 0> r_e2,
    FIn<T, 0> r_phi,
    FIn<T, 0> r_cos_theta,
    FIn<T, 0> r_beta,
    FIn<T, 0> m0,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FIn<T, 0> m3,
    FIn<T, 1> p0,
    FOut<T, 1> p1,
    FOut<T, 1> p2,
    FOut<T, 1> p3,
    FOut<T, 0> det
) {
    auto decay_out =
        three_body_decay<T>(r_e1, r_e2, r_phi, r_cos_theta, r_beta, m0, m1, m2, m3);
    auto p1_tmp = decay_out.first;
    auto p2_tmp = decay_out.second;
    auto det_tmp = decay_out.third;
    store_mom<T>(p1, boost<T>(p1_tmp, load_mom<T>(p0), 1.));
    store_mom<T>(p2, boost<T>(p2_tmp, load_mom<T>(p0), 1.));
    det = det_tmp;
    auto e3 = p0[0] - p1[0] - p2[0];
    p3[0] = max(e3, 0.);
    p3[1] = p0[1] - p1[1] - p2[1];
    p3[2] = p0[2] - p1[2] - p2[2];
    p3[3] = p0[3] - p1[3] - p2[3];
}

template <typename T>
KERNELSPEC void kernel_t_inv_min_max(
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FOut<T, 0> t_min,
    FOut<T, 0> t_max
) {
    // this function is based on the yminmax subroutine in MG5 (genps.f)
    // returns the absolute value of the t invariant
    FourMom<T> p_tot;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
    }

    auto s = lsquare<T>(p_tot);
    auto ma_2 = lsquare<T>(load_mom<T>(pa));
    auto mb_2 = lsquare<T>(load_mom<T>(pb));
    auto m1_2 = m1 * m1;
    auto m2_2 = m2 * m2;

    auto ysqr = kaellen<T>(s, ma_2, mb_2) * kaellen<T>(s, m1_2, m2_2);
    auto yr = where(ysqr > EPS, sqrt(ysqr), EPS);
    auto m_sum = ma_2 + m1_2;
    auto prod = (s + ma_2 - mb_2) * (s + m1_2 - m2_2);
    auto s_eps = s + EPS;
    auto y1 = m_sum - 0.5 * (prod - yr) / s_eps;
    auto y2 = m_sum - 0.5 * (prod + yr) / s_eps;
    auto t_min_tmp = -max(y2, y1);
    auto t_max_tmp = -min(y1, y2);
    t_min = max(t_min_tmp, 0.);
    t_max = where(t_max_tmp > t_min, t_max_tmp, t_min + EPS);
}

template <typename T>
KERNELSPEC void kernel_s_inv_min_max(
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 1> p3,
    FIn<T, 0> t1_abs,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FOut<T, 0> s_min,
    FOut<T, 0> s_max
) {
    // this function is based on the sminmax subroutine from Rikkert
    // expects t1_abs (positive t invariant) as input
    FourMom<T> p_tot, p_12, pt2;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
    }
    for (int i = 0; i < 4; ++i) {
        p_12[i] = pa[i] + pb[i] - p3[i];
    }
    for (int i = 0; i < 4; ++i) {
        pt2[i] = pb[i] - p3[i];
    }
    auto m0_2 = lsquare<T>(p_tot);
    auto ma_2 = lsquare<T>(load_mom<T>(pa));
    auto mb_2 = lsquare<T>(load_mom<T>(pb));
    auto m3_2 = lsquare<T>(load_mom<T>(p3));
    auto s12 = lsquare<T>(p_12);
    auto m1_2 = m1 * m1;
    auto m2_2 = m2 * m2;
    auto t2 = lsquare<T>(pt2);
    auto sqrtGG =
        bk_sqrt_g3i_g3im1<T>(m0_2, ma_2, mb_2, m1_2, m2_2, m3_2, t1_abs, t2, s12);
    auto V = bk_V<T>(m0_2, ma_2, mb_2, m1_2, m2_2, m3_2, t1_abs, t2, s12);
    auto lambda = kaellen<T>(s12, ma_2, t2);

    auto sa = m0_2 + m1_2 + 8 * (V + sqrtGG) / (lambda);
    auto sb = m0_2 + m1_2 + 8 * (V - sqrtGG) / (lambda);
    s_min = min(sa, sb);
    s_max = max(sa, sb);
}

template <typename T>
KERNELSPEC void kernel_invariants_from_momenta(
    FIn<T, 2> p_ext, FIn<T, 2> factors, FOut<T, 1> invariants
) {
    for (std::size_t i = 0; i < invariants.size(); ++i) {
        auto factors_i = factors[i];
        FourMom<T> p_sum{0., 0., 0., 0.};
        for (std::size_t j = 0; j < p_ext.size(); ++j) {
            auto p_j = p_ext[j];
            auto factor_ij = factors_i[j];
            p_sum[0] = p_sum[0] + factor_ij * p_j[0];
            p_sum[1] = p_sum[1] + factor_ij * p_j[1];
            p_sum[2] = p_sum[2] + factor_ij * p_j[2];
            p_sum[3] = p_sum[3] + factor_ij * p_j[3];
        }
        invariants[i] = lsquare<T>(p_sum);
    }
}

template <typename T>
KERNELSPEC void
kernel_pt_eta_phi_x(FIn<T, 2> p_ext, FIn<T, 0> x1, FIn<T, 0> x2, FOut<T, 1> output) {
    output[0] = x1;
    output[1] = x2;
    for (std::size_t i = 2; i < p_ext.size(); ++i) {
        auto p_i = p_ext[i];
        auto px = p_i[1], py = p_i[2], pz = p_i[3];
        auto pt2 = px * px + py * py + 1e-6;
        output[3 * i - 4] = 0.5 * log(pt2);
        output[3 * i - 3] = atan2(py, px);
        output[3 * i - 2] = atanh(pz / sqrt(pt2 + pz * pz));
    }
}

template <typename T>
KERNELSPEC void
kernel_mirror_momenta(FIn<T, 2> p_ext, IIn<T, 0> mirror, FOut<T, 2> p_out) {
    auto sign = 1. - 2. * FVal<T>(IVal<T>(mirror));
    for (std::size_t i = 0; i < p_ext.size(); ++i) {
        auto p_i = p_ext[i];
        auto q_i = p_out[i];
        q_i[0] = p_i[0];
        q_i[1] = p_i[1];
        q_i[2] = sign * p_i[2];
        q_i[3] = sign * p_i[3];
    }
}

template <typename T>
KERNELSPEC void
kernel_momenta_to_x1x2(FIn<T, 2> p_ext, FIn<T, 0> e_cm, FOut<T, 0> x1, FOut<T, 0> x2) {
    x1 = 2. * p_ext[0][0] / e_cm;
    x2 = 2. * p_ext[1][0] / e_cm;
}

} // namespace kernels
} // namespace madevent

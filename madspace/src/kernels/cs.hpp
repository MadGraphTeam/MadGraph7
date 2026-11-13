#pragma once

#include "kinematics.hpp"

namespace madspace {
namespace kernels {

// Catani-Seymour radiation (arXiv:hep-ph/9605323, Sec. 5; masses:
// arXiv:hep-ph/0201036, Sec. 5). The emitted parton i is massless and the
// emitter keeps its mass (m_ij = m_j), so a splitting of a gluon into massive
// quarks has no map here. The maps are covariant. The azimuth of i comes from
// a frame built from the momenta and the x axis, which boosts along z leave
// unchanged, so that a map and its inverse agree in the lab frame and in the
// center-of-mass frames of the Born and of the real emission.

// p.q for a massless q and p of mass m_p, without the cancellation of p.q for p
// along q: |q| (E_p - n.p) with E_p - n.p = (m_p^2 + |p x n|^2) / (E_p + n.p)
// where n.p > 0, n the direction of q.
template <typename T>
KERNELSPEC FVal<T> cs_lc_dot(FourMom<T> p, FVal<T> m_p, FourMom<T> q) {
    auto lq = sqrt(max(q[1] * q[1] + q[2] * q[2] + q[3] * q[3], EPS2));
    auto n1 = q[1] / lq, n2 = q[2] / lq, n3 = q[3] / lq;
    auto pn = p[1] * n1 + p[2] * n2 + p[3] * n3;
    auto c1 = p[2] * n3 - p[3] * n2;
    auto c2 = p[3] * n1 - p[1] * n3;
    auto c3 = p[1] * n2 - p[2] * n1;
    auto forward = pn > 0.;
    auto den = where(forward, p[0] + pn, FVal<T>(1.));
    auto stable = (m_p * m_p + c1 * c1 + c2 * c2 + c3 * c3) / den;
    return lq * where(forward, stable, p[0] - pn);
}

// determinant of three three-vectors
template <typename T>
KERNELSPEC FVal<T> cs_det3(
    FVal<T> a0,
    FVal<T> a1,
    FVal<T> a2,
    FVal<T> b0,
    FVal<T> b1,
    FVal<T> b2,
    FVal<T> c0,
    FVal<T> c1,
    FVal<T> c2
) {
    return a0 * (b1 * c2 - b2 * c1) - a1 * (b0 * c2 - b2 * c0) +
        a2 * (b0 * c1 - b1 * c0);
}

// The vector eps^{mu nu rho sigma} a_nu b_rho c_sigma with the sign for which
// cs_eps(e0, e3, e1) is the y axis for e0 = (1, 0), e3 the z axis and e1 the x
// axis.
template <typename T>
KERNELSPEC FourMom<T> cs_eps(FourMom<T> a, FourMom<T> b, FourMom<T> c) {
    // the components with lower indices
    FVal<T> A0 = a[0], A1 = -a[1], A2 = -a[2], A3 = -a[3];
    FVal<T> B0 = b[0], B1 = -b[1], B2 = -b[2], B3 = -b[3];
    FVal<T> C0 = c[0], C1 = -c[1], C2 = -c[2], C3 = -c[3];
    return {
        -cs_det3<T>(A1, A2, A3, B1, B2, B3, C1, C2, C3),
        cs_det3<T>(A0, A2, A3, B0, B2, B3, C0, C2, C3),
        -cs_det3<T>(A0, A1, A3, B0, B1, B3, C0, C1, C3),
        cs_det3<T>(A0, A1, A2, B0, B1, B2, C0, C1, C2)
    };
}

// The light-cone frame of P = u + w, from u (mass m_u), w (mass m_w) and their
// product uw = u.w: n_plus along w and n_minus opposite to it in the rest frame
// of P (n_plus.n_minus = 2, (n_plus + n_minus) / 2 = P / sqrt(P^2)), e1 from the
// x axis and e2 = e3 x e1 there (e3 = (n_plus - n_minus) / 2). Written in u and
// w, it avoids the cancellations of a boost into the rest frame of P when P is
// close to the light cone. e1 is normalized with -e1'^2 = 1 + n_plus^x n_minus^x
// of the unnormalized e1', and e2 = -eps(n_plus, n_minus, e1) / 2 is taken as
// -eps(w, u, x) / (2 N) times the normalization of e1, N = (u.w)^2 - m_u^2 m_w^2
// to the power 1/2 over 2 (the same vector, without the large components of
// n_plus and n_minus).
template <typename T>
KERNELSPEC Quartuplet<FourMom<T>, FourMom<T>, FourMom<T>, FourMom<T>>
cs_frame(FourMom<T> u, FVal<T> m_u, FourMom<T> w, FVal<T> m_w, FVal<T> uw) {
    auto mu2 = m_u * m_u, mw2 = m_w * m_w;
    auto rp = sqrt(max(mu2 + mw2 + 2. * uw, EPS2));
    // energies and momentum of u and w in the rest frame of P
    auto e_w = (uw + mw2) / rp;
    auto e_u = (uw + mu2) / rp;
    auto l_w = sqrt(max(uw * uw - mu2 * mw2, 0.)) / rp;
    auto a = 0.5 * (e_w + l_w);
    auto b = 0.5 * mw2 / max(e_w + l_w, EPS);
    auto d = 0.5 * (e_u + l_w);
    auto c = 0.5 * mu2 / max(e_u + l_w, EPS);
    auto norm = 1. / max(0.5 * l_w * rp, EPS2);
    FourMom<T> n_plus, n_minus;
    for (int k = 0; k < 4; ++k) {
        n_plus[k] = (d * w[k] - b * u[k]) * norm;
        n_minus[k] = (a * u[k] - c * w[k]) * norm;
    }
    // the x axis without its components along n_plus and n_minus
    FourMom<T> e1;
    for (int k = 0; k < 4; ++k) {
        e1[k] = 0.5 * (n_plus[1] * n_minus[k] + n_minus[1] * n_plus[k]);
    }
    e1[1] = e1[1] + 1.;
    auto len = 1. / sqrt(max(1. + n_plus[1] * n_minus[1], EPS2));
    for (int k = 0; k < 4; ++k) {
        e1[k] = e1[k] * len;
    }
    FourMom<T> x_axis{0., 1., 0., 0.};
    auto e2 = cs_eps<T>(w, u, x_axis);
    auto scale = -0.5 * len * norm;
    for (int k = 0; k < 4; ++k) {
        e2[k] = scale * e2[k];
    }
    return {n_plus, n_minus, e1, e2};
}

// The components p^0 + s p^3 and p^0 - s p^3 of p (mass m_p) along and against
// the beam direction of sign s, the smaller one from the mass shell
// (m_p^2 + p_T^2 over the larger), so that both keep their relative precision.
template <typename T>
KERNELSPEC Pair<FVal<T>, FVal<T>>
cs_beam_components(FourMom<T> p, FVal<T> m_p, FVal<T> s) {
    auto ahead = s * p[3] >= 0.;
    auto trans2 = m_p * m_p + p[1] * p[1] + p[2] * p[2];
    auto big = where(ahead, p[0] + s * p[3], p[0] - s * p[3]);
    auto small = trans2 / max(big, EPS2);
    return {where(ahead, big, small), where(ahead, small, big)};
}

// the momentum with the components along and against the beam direction of sign
// s and the transverse ones; the smaller of the first two from the mass shell
// if it would come out smaller than that from them directly
template <typename T>
KERNELSPEC FourMom<T> cs_from_beam_components(
    FVal<T> along, FVal<T> against, FVal<T> p1, FVal<T> p2, FVal<T> m, FVal<T> s
) {
    auto trans2 = m * m + p1 * p1 + p2 * p2;
    auto ahead = along >= against;
    along = where(ahead, along, trans2 / max(against, EPS2));
    against = where(ahead, trans2 / max(along, EPS2), against);
    return {0.5 * (along + against), p1, p2, 0.5 * s * (along - against)};
}

// A massless momentum of energy e in the rest frame of the frame f, at the
// angle theta (omc = 1 - cos theta) to n_plus and the azimuth 2 pi r_phi about
// it.
template <typename T>
KERNELSPEC FourMom<T> cs_massless_in_frame(
    FVal<T> e,
    FVal<T> omc,
    FVal<T> r_phi,
    Quartuplet<FourMom<T>, FourMom<T>, FourMom<T>, FourMom<T>> f
) {
    auto phi = 2. * PI * r_phi;
    omc = min(max(omc, FVal<T>(0.)), FVal<T>(2.));
    auto sth = sqrt(max(omc * (2. - omc), 0.));
    auto cp = (1. - 0.5 * omc) * e, cm = 0.5 * omc * e;
    auto c1 = sth * cos(phi) * e, c2 = sth * sin(phi) * e;
    FourMom<T> p;
    for (int k = 0; k < 4; ++k) {
        p[k] = cp * f.first[k] + cm * f.second[k] + c1 * f.third[k] + c2 * f.fourth[k];
    }
    return p;
}

// r_phi of a momentum in the frame f, in [0, 1)
template <typename T>
KERNELSPEC FVal<T>
cs_azimuth(FourMom<T> p, Quartuplet<FourMom<T>, FourMom<T>, FourMom<T>, FourMom<T>> f) {
    auto r_phi = atan2(-ldot<T>(p, f.fourth), -ldot<T>(p, f.third)) / (2. * PI);
    return where(r_phi < 0., r_phi + 1., r_phi);
}

// The ranges of the final-final variables: y_max, z_min, z_max at y, and the
// factor Q^2 a^2 / sqrt(lambda(1, mu_j^2, mu_k^2)) / (16 pi^2) of the phase
// space (CDST eqs. 5.11-5.13 for m_i = 0, m_ij = m_j), from Q^2 and
// two_born = Q^2 - m_j^2 - m_k^2 = 2 p_ij.p_k.
template <typename T>
KERNELSPEC Quartuplet<FVal<T>, FVal<T>, FVal<T>, FVal<T>> cs_final_final_ranges(
    FVal<T> q2, FVal<T> two_born, FVal<T> m_j, FVal<T> m_k, FVal<T> y
) {
    auto mj2 = m_j * m_j / q2, mk2 = m_k * m_k / q2;
    auto a = two_born / q2;
    auto muk = sqrt(mk2);
    auto y_max = 1. - 2. * muk * (1. - muk) / a;
    auto omy = a * (1. - y);
    auto v =
        sqrt(max((2. * mk2 + omy) * (2. * mk2 + omy) - 4. * mk2, 0.)) / max(omy, EPS);
    auto z0 = a * y / max(2. * (mj2 + a * y), EPS2);
    auto norm = q2 * a * a / (16. * PI * PI * sqrt(max(a * a - 4. * mj2 * mk2, EPS2)));
    // y_max, z_min, z_max and the phase-space factor
    return {y_max, z0 * (1. - v), z0 * (1. + v), norm};
}

template <typename T>
KERNELSPEC void kernel_cs_final_final_radiate(
    FIn<T, 0> r_y,
    FIn<T, 0> r_z,
    FIn<T, 0> r_phi,
    FIn<T, 1> p_ij,
    FIn<T, 1> p_k,
    FIn<T, 0> m_j,
    FIn<T, 0> m_k,
    FOut<T, 1> p_i,
    FOut<T, 1> p_j,
    FOut<T, 1> p_k_out,
    FOut<T, 0> y,
    FOut<T, 0> z,
    FOut<T, 0> det
) {
    FVal<T> ry = r_y, rz = r_z, rphi = r_phi, mj = m_j, mk = m_k;
    auto pij = load_mom<T>(p_ij);
    auto pk = load_mom<T>(p_k);
    FourMom<T> q;
    for (int k = 0; k < 4; ++k) {
        q[k] = pij[k] + pk[k];
    }
    auto two_born = 2. * ldot<T>(pij, pk);
    auto q2 = mj * mj + mk * mk + two_born;
    auto mk2 = mk * mk;

    // y from its range at fixed Q, then z from its range at that y
    auto y_max = cs_final_final_ranges<T>(q2, two_born, mj, mk, FVal<T>(0.)).first;
    auto yv = y_max * ry * ry;
    auto ranges = cs_final_final_ranges<T>(q2, two_born, mj, mk, yv);
    auto z_min = ranges.second, z_max = ranges.third;
    auto zv = z_min + (z_max - z_min) * rz * rz;

    // the spectator, rescaled in the frame of Q (CDST eq. 5.8 inverted), with
    // 2 (p_i + p_j).p_k = two_k
    auto s_ij = mj * mj + yv * two_born;
    auto two_k = two_born * (1. - yv);
    auto lam = max(two_k * two_k - 4. * s_ij * mk2, 0.);
    auto lam_born = max(two_born * two_born - 4. * mj * mj * mk2, EPS2);
    auto scale = sqrt(lam / lam_born);
    auto qk = ldot<T>(q, pk);
    auto shift = (0.5 * two_k + mk2) / q2;
    FourMom<T> pk_out;
    for (int k = 0; k < 4; ++k) {
        pk_out[k] = scale * (pk[k] - qk / q2 * q[k]) + shift * q[k];
    }

    // i in the frame of Q: its energy from p_i.Q = p_i.p_j + p_i.p_k, its angle
    // to the direction of k from p_i.p_k
    auto rq = sqrt(q2);
    auto e_k = (0.5 * two_k + mk2) / rq;
    auto l_k = sqrt(lam) / (2. * rq);
    auto pipj = 0.5 * yv * two_born;
    auto pipk = 0.5 * zv * two_k;
    auto e_i = (pipj + pipk) / rq;
    auto e_i_safe = max(e_i, EPS);
    auto omc = (pipk - e_i * mk2 / max(e_k + l_k, EPS)) / max(e_i_safe * l_k, EPS2);
    auto frame = cs_frame<T>(pij, mj, pk, mk, 0.5 * two_born);
    auto pi = cs_massless_in_frame<T>(e_i, omc, rphi, frame);
    FourMom<T> pj;
    for (int k = 0; k < 4; ++k) {
        pj[k] = q[k] - pi[k] - pk_out[k];
    }
    store_mom<T>(p_i, pi);
    store_mom<T>(p_j, pj);
    store_mom<T>(p_k_out, pk_out);
    y = yv;
    z = zv;
    det = ranges.fourth * (1. - yv) * 2. * y_max * ry * 2. * (z_max - z_min) * rz;
}

template <typename T>
KERNELSPEC void kernel_cs_final_final_project(
    FIn<T, 1> p_i,
    FIn<T, 1> p_j,
    FIn<T, 1> p_k,
    FIn<T, 0> m_j,
    FIn<T, 0> m_k,
    FOut<T, 1> p_ij,
    FOut<T, 1> p_k_out,
    FOut<T, 0> r_y,
    FOut<T, 0> r_z,
    FOut<T, 0> r_phi,
    FOut<T, 0> y,
    FOut<T, 0> z,
    FOut<T, 0> det
) {
    FVal<T> mj = m_j, mk = m_k;
    auto pi = load_mom<T>(p_i);
    auto pj = load_mom<T>(p_j);
    auto pk = load_mom<T>(p_k);
    auto mk2 = mk * mk;
    auto pipj = cs_lc_dot<T>(pj, mj, pi);
    auto pipk = cs_lc_dot<T>(pk, mk, pi);
    auto pjpk = where(
        mj == 0.,
        cs_lc_dot<T>(pk, mk, pj),
        where(mk == 0., cs_lc_dot<T>(pj, mj, pk), ldot<T>(pj, pk))
    );
    auto yv = pipj / max(pipj + pipk + pjpk, EPS2);
    auto zv = pipk / max(pipk + pjpk, EPS2);
    FourMom<T> q;
    for (int k = 0; k < 4; ++k) {
        q[k] = pi[k] + pj[k] + pk[k];
    }
    auto two_k = 2. * (pipk + pjpk);
    auto two_born = two_k + 2. * pipj;
    auto q2 = mj * mj + mk2 + two_born;
    auto s_ij = mj * mj + 2. * pipj;

    // CDST eq. 5.8
    auto lam = max(two_k * two_k - 4. * s_ij * mk2, EPS2);
    auto lam_born = max(two_born * two_born - 4. * mj * mj * mk2, 0.);
    auto scale = sqrt(lam_born / lam);
    auto qk = ldot<T>(q, pk);
    auto shift = (0.5 * two_born + mk2) / q2;
    FourMom<T> pk_out, pij;
    for (int k = 0; k < 4; ++k) {
        pk_out[k] = scale * (pk[k] - qk / q2 * q[k]) + shift * q[k];
        pij[k] = q[k] - pk_out[k];
    }

    auto ranges = cs_final_final_ranges<T>(q2, two_born, mj, mk, yv);
    auto y_max = ranges.first, z_min = ranges.second, z_max = ranges.third;
    auto ry = sqrt(max(yv / max(y_max, EPS2), 0.));
    auto rz = sqrt(max((zv - z_min) / max(z_max - z_min, EPS2), 0.));
    auto frame = cs_frame<T>(pij, mj, pk_out, mk, 0.5 * two_born);
    store_mom<T>(p_ij, pij);
    store_mom<T>(p_k_out, pk_out);
    r_y = ry;
    r_z = rz;
    r_phi = cs_azimuth<T>(pi, frame);
    y = yv;
    z = zv;
    det = 1. /
        max(ranges.fourth * (1. - yv) * 2. * y_max * ry * 2. * (z_max - z_min) * rz,
            EPS2);
}

// the factor of the initial-final phase space (CS eqs. 5.48, 5.72 and CDST eqs.
// 5.48, 5.79): 2 p_f.p_a / (16 pi^2) dx dz dphi / (2 pi) for the incoming
// momentum p_a of the real emission, times 1 / x from its momentum fraction,
// times the Jacobian of r_x and r_z
template <typename T>
KERNELSPEC FVal<T> cs_initial_final_det(
    FVal<T> fa, FVal<T> x, FVal<T> x_min, FVal<T> r_x, FVal<T> z_max, FVal<T> r_z
) {
    return 2. * fa / (16. * PI * PI) * 2. * (1. - x_min) * r_x * 2. * z_max * r_z / x;
}

// The emission with an incoming parton a in the Sudakov decomposition
// p_i = z p_f + beta p_a + k_perp along the beam (p_a and p_f those of the
// Born and the real emission, k_perp orthogonal to both), in the components
// along and against the direction of a. That is the map of the light-cone
// frame of p_f + (1 - x) p_a with e1 from the x axis, but without the
// cancellations of that frame when p_f is close to the beam.
template <typename T>
KERNELSPEC void kernel_cs_initial_final_radiate(
    FIn<T, 0> r_x,
    FIn<T, 0> r_z,
    FIn<T, 0> r_phi,
    FIn<T, 1> p_a,
    FIn<T, 1> p_f,
    FIn<T, 0> x_min,
    FIn<T, 0> m_f,
    FOut<T, 1> p_a_out,
    FOut<T, 1> p_i,
    FOut<T, 1> p_f_out,
    FOut<T, 0> x,
    FOut<T, 0> z,
    FOut<T, 0> det
) {
    FVal<T> rx = r_x, rz = r_z, rphi = r_phi, xmin = x_min, mf = m_f;
    auto pa = load_mom<T>(p_a);
    auto pf = load_mom<T>(p_f);
    auto s = where(pa[3] >= 0., FVal<T>(1.), FVal<T>(-1.));
    // x from x_min (the momentum fraction of a in the Born) to 1
    auto omx = (1. - xmin) * rx * rx;
    auto xv = 1. - omx;
    FourMom<T> pa_out;
    for (int k = 0; k < 4; ++k) {
        pa_out[k] = pa[k] / xv;
    }
    auto a_along = pa_out[0] + s * pa_out[3];
    auto u = cs_beam_components<T>(pf, mf, s);
    auto fa = 0.5 * a_along * u.second;
    // z = p_i.p_a / (p_i + p_f).p_a from 0 to its largest value at this x,
    // where k_perp vanishes
    auto mu2 = mf * mf / max(2. * fa, EPS2);
    auto z_max = omx / max(omx + mu2, EPS2);
    auto zv = z_max * rz * rz;
    auto beta = omx * (1. - zv) - zv * mf * mf / max(fa, EPS2);
    auto kt = sqrt(max(2. * zv * (1. - zv) * omx * fa - zv * zv * mf * mf, 0.));
    // the azimuth from the x axis, about the direction of a
    auto phi = 2. * PI * rphi;
    auto k1 = kt * cos(phi), k2 = s * kt * sin(phi);
    auto k_along = 2. * (k1 * pf[1] + k2 * pf[2]) / max(u.second, EPS2);
    auto pi = cs_from_beam_components<T>(
        zv * u.first + beta * a_along + k_along,
        zv * u.second,
        zv * pf[1] + k1,
        zv * pf[2] + k2,
        0.,
        s
    );
    auto pf_out = cs_from_beam_components<T>(
        (1. - zv) * u.first + (omx - beta) * a_along - k_along,
        (1. - zv) * u.second,
        (1. - zv) * pf[1] - k1,
        (1. - zv) * pf[2] - k2,
        mf,
        s
    );
    store_mom<T>(p_a_out, pa_out);
    store_mom<T>(p_i, pi);
    store_mom<T>(p_f_out, pf_out);
    x = xv;
    z = zv;
    det = cs_initial_final_det<T>(fa, xv, xmin, rx, z_max, rz);
}

template <typename T>
KERNELSPEC void kernel_cs_initial_final_project(
    FIn<T, 1> p_a,
    FIn<T, 1> p_i,
    FIn<T, 1> p_f,
    FIn<T, 0> x_real,
    FIn<T, 0> m_f,
    FOut<T, 1> p_a_out,
    FOut<T, 1> p_f_out,
    FOut<T, 0> r_x,
    FOut<T, 0> r_z,
    FOut<T, 0> r_phi,
    FOut<T, 0> x,
    FOut<T, 0> z,
    FOut<T, 0> det
) {
    FVal<T> xr = x_real, mf = m_f;
    auto pa = load_mom<T>(p_a);
    auto pi = load_mom<T>(p_i);
    auto pf = load_mom<T>(p_f);
    auto s = where(pa[3] >= 0., FVal<T>(1.), FVal<T>(-1.));
    auto a_along = pa[0] + s * pa[3];
    auto i = cs_beam_components<T>(pi, 0., s);
    auto f = cs_beam_components<T>(pf, mf, s);
    auto pipa = 0.5 * a_along * i.second;
    auto pfpa = 0.5 * a_along * f.second;
    auto pipf = cs_lc_dot<T>(pf, mf, pi);
    auto fa = pipa + pfpa;
    auto omx = pipf / max(fa, EPS2);
    auto xv = 1. - omx;
    auto zv = pipa / max(fa, EPS2);
    FourMom<T> pa_out;
    for (int k = 0; k < 4; ++k) {
        pa_out[k] = xv * pa[k];
    }
    // p_i + p_f - (1 - x) p_a
    auto pf_out = cs_from_beam_components<T>(
        i.first + f.first - omx * a_along,
        i.second + f.second,
        pi[1] + pf[1],
        pi[2] + pf[2],
        mf,
        s
    );
    auto xmin = xv * xr;
    auto mu2 = mf * mf / max(2. * fa, EPS2);
    auto z_max = omx / max(omx + mu2, EPS2);
    auto rx = sqrt(max(omx / max(1. - xmin, EPS2), 0.));
    auto rz = sqrt(max(zv / max(z_max, EPS2), 0.));
    // k_perp = p_i - z (p_i + p_f) transverse to the beam
    auto k1 = (1. - zv) * pi[1] - zv * pf[1];
    auto k2 = (1. - zv) * pi[2] - zv * pf[2];
    auto rphi = atan2(s * k2, k1) / (2. * PI);
    store_mom<T>(p_a_out, pa_out);
    store_mom<T>(p_f_out, pf_out);
    r_x = rx;
    r_z = rz;
    r_phi = where(rphi < 0., rphi + 1., rphi);
    x = xv;
    z = zv;
    det = 1. / max(cs_initial_final_det<T>(fa, xv, xmin, rx, z_max, rz), EPS2);
}

// Lambda(K, K~) k of CS eq. 5.143, the Lorentz transformation that takes K to
// K~ (K^2 = K~^2)
template <typename T>
KERNELSPEC FourMom<T> cs_lorentz(FourMom<T> k, FourMom<T> kk, FourMom<T> kt) {
    FourMom<T> sum;
    for (int n = 0; n < 4; ++n) {
        sum[n] = kk[n] + kt[n];
    }
    auto c_sum = 2. * ldot<T>(k, sum) / max(ldot<T>(sum, sum), EPS2);
    auto c_kt = 2. * ldot<T>(k, kk) / max(ldot<T>(kk, kk), EPS2);
    FourMom<T> out;
    for (int n = 0; n < 4; ++n) {
        out[n] = k[n] - c_sum * sum[n] + c_kt * kt[n];
    }
    return out;
}

// the factor of the initial-initial phase space (CS eq. 5.151): 2 p_a.p_b /
// (16 pi^2) dx dv dphi / (2 pi) for the incoming momenta of the real emission,
// times 1 / x from the momentum fraction of a, times the Jacobian of r_x, r_v
template <typename T>
KERNELSPEC FVal<T>
cs_initial_initial_det(FVal<T> s, FVal<T> x, FVal<T> x_min, FVal<T> r_x, FVal<T> r_v) {
    auto omx = 1. - x;
    return s / (16. * PI * PI) * 2. * (1. - x_min) * r_x * 2. * omx * r_v / x;
}

template <typename T>
KERNELSPEC void kernel_cs_initial_initial_radiate(
    FIn<T, 0> r_x,
    FIn<T, 0> r_v,
    FIn<T, 0> r_phi,
    FIn<T, 1> p_a,
    FIn<T, 1> p_b,
    FIn<T, 2> p,
    FIn<T, 0> x_min,
    FOut<T, 1> p_a_out,
    FOut<T, 1> p_i,
    FOut<T, 2> p_out,
    FOut<T, 0> x,
    FOut<T, 0> v,
    FOut<T, 0> det
) {
    FVal<T> rx = r_x, rv = r_v, rphi = r_phi, xmin = x_min;
    auto pa = load_mom<T>(p_a);
    auto pb = load_mom<T>(p_b);
    auto omx = (1. - xmin) * rx * rx;
    auto xv = 1. - omx;
    FourMom<T> pa_out;
    for (int k = 0; k < 4; ++k) {
        pa_out[k] = pa[k] / xv;
    }
    // v = p_a.p_i / p_a.p_b from 0 to 1 - x
    auto vv = omx * rv * rv;
    auto pab = cs_lc_dot<T>(pa_out, 0., pb);
    auto s = 2. * pab;
    // i in the frame of p_a + p_b: energy (1 - x) sqrt(s) / 2, 1 - cos of its
    // angle to p_a 2 v / (1 - x)
    auto e_i = 0.5 * omx * sqrt(max(s, 0.));
    auto omc = 2. * vv / max(omx, EPS2);
    auto frame = cs_frame<T>(pb, 0., pa_out, 0., pab);
    auto pi = cs_massless_in_frame<T>(e_i, omc, rphi, frame);
    // the final state from K~ = p~_a + p_b to K = p_a + p_b - p_i
    FourMom<T> k_born, k_real;
    for (int k = 0; k < 4; ++k) {
        k_born[k] = pa[k] + pb[k];
        k_real[k] = pa_out[k] + pb[k] - pi[k];
    }
    for (std::size_t n = 0; n < p.size(); ++n) {
        store_mom<T>(p_out[n], cs_lorentz<T>(load_mom<T>(p[n]), k_born, k_real));
    }
    store_mom<T>(p_a_out, pa_out);
    store_mom<T>(p_i, pi);
    x = xv;
    v = vv;
    det = cs_initial_initial_det<T>(s, xv, xmin, rx, rv);
}

template <typename T>
KERNELSPEC void kernel_cs_initial_initial_project(
    FIn<T, 1> p_a,
    FIn<T, 1> p_b,
    FIn<T, 1> p_i,
    FIn<T, 2> p,
    FIn<T, 0> x_real,
    FOut<T, 1> p_a_out,
    FOut<T, 2> p_out,
    FOut<T, 0> r_x,
    FOut<T, 0> r_v,
    FOut<T, 0> r_phi,
    FOut<T, 0> x,
    FOut<T, 0> v,
    FOut<T, 0> det
) {
    FVal<T> xr = x_real;
    auto pa = load_mom<T>(p_a);
    auto pb = load_mom<T>(p_b);
    auto pi = load_mom<T>(p_i);
    auto pab = cs_lc_dot<T>(pa, 0., pb);
    auto pia = cs_lc_dot<T>(pi, 0., pa);
    auto pib = cs_lc_dot<T>(pi, 0., pb);
    auto omx = (pia + pib) / max(pab, EPS2);
    auto xv = 1. - omx;
    auto vv = pia / max(pab, EPS2);
    FourMom<T> pa_out, k_real, k_born;
    for (int k = 0; k < 4; ++k) {
        pa_out[k] = xv * pa[k];
        k_real[k] = pa[k] + pb[k] - pi[k];
        k_born[k] = pa_out[k] + pb[k];
    }
    for (std::size_t n = 0; n < p.size(); ++n) {
        store_mom<T>(p_out[n], cs_lorentz<T>(load_mom<T>(p[n]), k_real, k_born));
    }
    auto xmin = xv * xr;
    auto rx = sqrt(max(omx / max(1. - xmin, EPS2), 0.));
    auto rv = sqrt(max(vv / max(omx, EPS2), 0.));
    auto frame = cs_frame<T>(pb, 0., pa, 0., pab);
    store_mom<T>(p_a_out, pa_out);
    r_x = rx;
    r_v = rv;
    r_phi = cs_azimuth<T>(pi, frame);
    x = xv;
    v = vv;
    det = 1. / max(cs_initial_initial_det<T>(2. * pab, xv, xmin, rx, rv), EPS2);
}

// Catani-Seymour dipoles (arXiv:hep-ph/9605323, Sec. 5 and 10; masses:
// arXiv:hep-ph/0201036, Sec. 5). A dipole is
//
//   D = -g_s^2 [color_weight <T_j . T_k> + spin_weight <T_j . T_k>(k_t)]
//
// with the color-correlated Born <T_j . T_k> of the Born emitter j and
// spectator k, and <T_j . T_k>(k_t) the same with k_t in place of the
// polarization vector of the Born emitter (a gluon), so that the tensor part
// k~^mu k~^nu of the kernel contracts with the Born through k_t: k_t is the
// vector k~ of the kernel (z_i p_i - z_j p_j, p_i / u - p_k / (1 - u),
// p_i - p_i.p_a / p_a.p_b p_b) without its part along the Born emitter, by
// the Born spectator, which the gauge invariance of the Born leaves out. The
// weights are the kernels of the papers over the invariant of the dipole
// (p_i.p_j, x p_i.p_j or x p_a.p_i) and over the Casimir of the Born
// emitter, zero outside the alpha of the family (y < alpha y_+ for ff,
// 1 - x < alpha for fi, u < alpha for if, v < alpha for ii). The options are
// the kernel (see the yaml), alpha and kappa, and the masses of the emitter
// and the spectator where they can be massive.

constexpr double CS_CA = 3.;
constexpr double CS_CF = 4. / 3.;
constexpr double CS_TR = 0.5;

// v - (v.ref / dir.ref) dir
template <typename T>
KERNELSPEC FourMom<T> cs_without_along(FourMom<T> v, FourMom<T> dir, FourMom<T> ref) {
    auto c = ldot<T>(v, ref) /
        where(ldot<T>(dir, ref) == 0., FVal<T>(1.), ldot<T>(dir, ref));
    FourMom<T> out;
    for (int n = 0; n < 4; ++n) {
        out[n] = v[n] - c * dir[n];
    }
    return out;
}

template <typename T>
KERNELSPEC void cs_store_dipole(
    FVal<T> gate,
    FVal<T> s,
    FVal<T> a,
    FVal<T> b,
    FourMom<T> k_t,
    FOut<T, 0> color_weight,
    FOut<T, 0> spin_weight,
    FOut<T, 1> spin_vector
) {
    auto inverse = where(gate > 0., 1. / where(s > 0., s, FVal<T>(1.)), FVal<T>(0.));
    color_weight = a * inverse;
    spin_weight = b * inverse;
    store_mom<T>(spin_vector, k_t);
}

template <typename T>
KERNELSPEC void kernel_cs_final_final_dipole(
    FIn<T, 1> p_i,
    FIn<T, 1> p_j,
    FIn<T, 1> p_k,
    FIn<T, 1> born_ij,
    FIn<T, 1> born_k,
    FSIn<T, 1> options,
    FOut<T, 0> color_weight,
    FOut<T, 0> spin_weight,
    FOut<T, 1> spin_vector
) {
    FVal<T> kernel = options[0], alpha = options[1], kappa = options[2];
    FVal<T> mj = options[3], mk = options[4];
    auto pi = load_mom<T>(p_i);
    auto pj = load_mom<T>(p_j);
    auto pk = load_mom<T>(p_k);
    auto pipj = cs_lc_dot<T>(pj, mj, pi);
    auto pipk = cs_lc_dot<T>(pk, mk, pi);
    auto pjpk = where(
        mj == 0.,
        cs_lc_dot<T>(pk, mk, pj),
        where(mk == 0., cs_lc_dot<T>(pj, mj, pk), ldot<T>(pj, pk))
    );
    auto yv = pipj / max(pipj + pipk + pjpk, EPS2);
    auto zi = pipk / max(pipk + pjpk, EPS2);
    auto zj = 1. - zi;
    auto q2 = mj * mj + mk * mk + 2. * (pipj + pipk + pjpk);
    auto mj2 = mj * mj / q2, mk2 = mk * mk / q2;
    auto av = 1. - mj2 - mk2;
    auto muk = sqrt(mk2);
    auto y_plus = 1. - 2. * muk * (1. - muk) / av;
    auto omy = av * (1. - yv);
    // the velocity of the dipole and that of the Born (CDST eqs. 5.12, 5.14)
    auto vel =
        sqrt(max((2. * mk2 + omy) * (2. * mk2 + omy) - 4. * mk2, 0.)) / max(omy, EPS2);
    auto vel_born = sqrt(max(av * av - 4. * mj2 * mk2, 0.)) / av;
    auto zz = (zi - 0.5 * (1. - vel)) * (zj - 0.5 * (1. - vel));
    auto zpm = 0.25 * (1. - vel * vel);
    // CDST eqs. 5.16-5.18: q -> q g, g -> g g, g -> q q~, with 1 - z_j (1 -
    // y) = z_i + y z_j and 1 - z_i (1 - y) = z_j + y z_i, which keep their
    // digits in the soft limit
    auto soft_i = zi + yv * zj, soft_j = zj + yv * zi;
    auto a_qg = 2. / soft_i - vel_born / vel * (1. + zj + mj * mj / max(pipj, EPS2));
    auto a_gg = 2. * (1. / soft_j + 1. / soft_i + (kappa * zpm - 2.) / vel);
    auto a_gq = CS_TR / CS_CA * (1. - 2. * kappa * zpm) / vel;
    auto b_gg = 2. / (vel * max(pipj, EPS2));
    auto b_gq = -CS_TR / CS_CA * b_gg;
    auto a = where(kernel == 0., a_qg, where(kernel == 1., a_gg, a_gq));
    auto b = where(kernel == 0., FVal<T>(0.), where(kernel == 1., b_gg, b_gq));
    FourMom<T> kk;
    for (int n = 0; n < 4; ++n) {
        kk[n] = (zi - 0.5 * (1. - vel)) * pi[n] - (zj - 0.5 * (1. - vel)) * pj[n];
    }
    auto k_t = cs_without_along<T>(kk, load_mom<T>(born_ij), load_mom<T>(born_k));
    cs_store_dipole<T>(
        where(yv < alpha * y_plus, FVal<T>(1.), FVal<T>(0.)),
        pipj,
        a,
        b,
        k_t,
        color_weight,
        spin_weight,
        spin_vector
    );
}

template <typename T>
KERNELSPEC void kernel_cs_final_initial_dipole(
    FIn<T, 1> p_i,
    FIn<T, 1> p_j,
    FIn<T, 1> p_a,
    FIn<T, 1> born_ij,
    FIn<T, 1> born_a,
    FSIn<T, 1> options,
    FOut<T, 0> color_weight,
    FOut<T, 0> spin_weight,
    FOut<T, 1> spin_vector
) {
    FVal<T> kernel = options[0], alpha = options[1], mj = options[3];
    auto pi = load_mom<T>(p_i);
    auto pj = load_mom<T>(p_j);
    auto pa = load_mom<T>(p_a);
    auto pipj = cs_lc_dot<T>(pj, mj, pi);
    auto pipa = cs_lc_dot<T>(pi, 0., pa);
    auto pjpa = cs_lc_dot<T>(pj, mj, pa);
    auto omx = pipj / max(pipa + pjpa, EPS2);
    auto xv = 1. - omx;
    auto zi = pipa / max(pipa + pjpa, EPS2);
    auto zj = 1. - zi;
    // CS eqs. 5.65-5.67, CDST eq. 5.50: q -> q g, g -> g g, g -> q q~
    auto a_qg = 2. / (omx + zi) - 1. - zj - mj * mj / max(pipj, EPS2);
    auto a_gg = 2. * (1. / (zj + omx) + 1. / (zi + omx) - 2.);
    auto a_gq = FVal<T>(CS_TR / CS_CA);
    auto b_gg = 2. / max(pipj, EPS2);
    auto b_gq = -CS_TR / CS_CA * b_gg;
    auto a = where(kernel == 0., a_qg, where(kernel == 1., a_gg, a_gq));
    auto b = where(kernel == 0., FVal<T>(0.), where(kernel == 1., b_gg, b_gq));
    FourMom<T> kk;
    for (int n = 0; n < 4; ++n) {
        kk[n] = zi * pi[n] - zj * pj[n];
    }
    auto k_t = cs_without_along<T>(kk, load_mom<T>(born_ij), load_mom<T>(born_a));
    cs_store_dipole<T>(
        where(omx < alpha, FVal<T>(1.), FVal<T>(0.)),
        xv * pipj,
        a,
        b,
        k_t,
        color_weight,
        spin_weight,
        spin_vector
    );
}

// the kernels of an incoming emitter (CS eqs. 5.77-5.80, 5.145-5.148, CDST
// eqs. 5.79-5.82): q -> q g, g -> g g, g -> q~ q, q -> g q, with the
// tensor parts as weights of k~^mu k~^nu, t_k = 2 u (1 - u) / p_i.p_k for an
// incoming emitter with a final-state spectator and 2 p_a.p_b / (p_i.p_a
// p_i.p_b) with an incoming one
template <typename T>
KERNELSPEC Pair<FVal<T>, FVal<T>>
cs_initial_kernels(FVal<T> kernel, FVal<T> x, FVal<T> u, FVal<T> t_k) {
    auto omx = 1. - x;
    auto a_qg = 2. / (omx + u) - 1. - x;
    auto a_gg = 2. * (1. / (omx + u) - 1. + x * omx);
    auto a_qq = CS_TR / CS_CF * (1. - 2. * x * omx);
    auto a_gq = CS_CF / CS_CA * x;
    auto b_gg = omx / x * t_k;
    auto b_gq = CS_CF / CS_CA * omx / x * t_k;
    auto a = where(
        kernel == 0., a_qg, where(kernel == 1., a_gg, where(kernel == 2., a_qq, a_gq))
    );
    auto b = where(kernel == 1., b_gg, where(kernel == 3., b_gq, FVal<T>(0.)));
    return {a, b};
}

template <typename T>
KERNELSPEC void kernel_cs_initial_final_dipole(
    FIn<T, 1> p_a,
    FIn<T, 1> p_i,
    FIn<T, 1> p_k,
    FIn<T, 1> born_a,
    FIn<T, 1> born_k,
    FSIn<T, 1> options,
    FOut<T, 0> color_weight,
    FOut<T, 0> spin_weight,
    FOut<T, 1> spin_vector
) {
    FVal<T> kernel = options[0], alpha = options[1], mk = options[4];
    auto pa = load_mom<T>(p_a);
    auto pi = load_mom<T>(p_i);
    auto pk = load_mom<T>(p_k);
    auto pipa = cs_lc_dot<T>(pi, 0., pa);
    auto pkpa = cs_lc_dot<T>(pk, mk, pa);
    auto pipk = cs_lc_dot<T>(pk, mk, pi);
    auto omx = pipk / max(pipa + pkpa, EPS2);
    auto xv = 1. - omx;
    auto u = pipa / max(pipa + pkpa, EPS2);
    auto ab = cs_initial_kernels<T>(kernel, xv, u, 2. * u * (1. - u) / max(pipk, EPS2));
    FourMom<T> kk;
    auto cu = 1. / max(u, EPS2), cv = 1. / max(1. - u, EPS2);
    for (int n = 0; n < 4; ++n) {
        kk[n] = cu * pi[n] - cv * pk[n];
    }
    auto k_t = cs_without_along<T>(kk, load_mom<T>(born_a), load_mom<T>(born_k));
    cs_store_dipole<T>(
        where(u < alpha, FVal<T>(1.), FVal<T>(0.)),
        xv * pipa,
        ab.first,
        ab.second,
        k_t,
        color_weight,
        spin_weight,
        spin_vector
    );
}

template <typename T>
KERNELSPEC void kernel_cs_initial_initial_dipole(
    FIn<T, 1> p_a,
    FIn<T, 1> p_b,
    FIn<T, 1> p_i,
    FSIn<T, 1> options,
    FOut<T, 0> color_weight,
    FOut<T, 0> spin_weight,
    FOut<T, 1> spin_vector
) {
    FVal<T> kernel = options[0], alpha = options[1];
    auto pa = load_mom<T>(p_a);
    auto pb = load_mom<T>(p_b);
    auto pi = load_mom<T>(p_i);
    auto pab = cs_lc_dot<T>(pa, 0., pb);
    auto pia = cs_lc_dot<T>(pi, 0., pa);
    auto pib = cs_lc_dot<T>(pi, 0., pb);
    auto omx = (pia + pib) / max(pab, EPS2);
    auto xv = 1. - omx;
    auto vv = pia / max(pab, EPS2);
    auto ab =
        cs_initial_kernels<T>(kernel, xv, FVal<T>(0.), 2. * pab / max(pia * pib, EPS2));
    // p_i without its parts along p_a and p_b
    auto ca = pib / max(pab, EPS2), cb = pia / max(pab, EPS2);
    FourMom<T> k_t;
    for (int n = 0; n < 4; ++n) {
        k_t[n] = pi[n] - ca * pa[n] - cb * pb[n];
    }
    cs_store_dipole<T>(
        where(vv < alpha, FVal<T>(1.), FVal<T>(0.)),
        xv * pia,
        ab.first,
        ab.second,
        k_t,
        color_weight,
        spin_weight,
        spin_vector
    );
}

} // namespace kernels
} // namespace madspace

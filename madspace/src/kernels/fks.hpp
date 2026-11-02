#pragma once

#include "kinematics.hpp"

namespace madspace {
namespace kernels {

// FKS radiation as in genps_fks.f of MadGraph (arXiv:0908.4272, Sec. 5). All
// momenta are in the partonic centre-of-mass frame of the Born, where the
// incoming momenta add up to (sqrt_s, 0, 0, 0).

inline constexpr double FKS_STINY = 1e-6;
inline constexpr double FKS_CTINY = 5e-7;

// cos, sin of the polar angle and azimuth of a three-momentum, as getangles in
// genps_fks.f: a vector on the z axis (or of zero length) has azimuth zero.
// Unlike getangles, the sines come from the transverse components, since
// sqrt(1 - cos^2) loses half the digits of the polar angle near the axis.
template <typename T>
KERNELSPEC Quartuplet<FVal<T>, FVal<T>, FVal<T>, FVal<T>> fks_angles(FourMom<T> p) {
    auto pt2 = p[1] * p[1] + p[2] * p[2];
    auto len = sqrt(max(pt2 + p[3] * p[3], EPS2));
    auto pt = sqrt(pt2);
    auto axis = pt2 == 0.;
    auto pt_safe = where(axis, FVal<T>(1.), pt);
    auto cth = where(axis, where(p[3] < 0., FVal<T>(-1.), FVal<T>(1.)), p[3] / len);
    auto sth = pt / len;
    auto cphi = where(axis, FVal<T>(1.), p[1] / pt_safe);
    auto sphi = where(axis, FVal<T>(0.), p[2] / pt_safe);
    return {cth, sth, cphi, sphi};
}

// rotate_invar of fks_singular.f: rotation by theta about y, then by phi about z
template <typename T>
KERNELSPEC FourMom<T> fks_rotate(
    FVal<T> e,
    FVal<T> q1,
    FVal<T> q2,
    FVal<T> q3,
    Quartuplet<FVal<T>, FVal<T>, FVal<T>, FVal<T>> a
) {
    auto cth = a.first, sth = a.second, cphi = a.third, sphi = a.fourth;
    return {
        e,
        q1 * cphi * cth - q2 * sphi + q3 * cphi * sth,
        q1 * sphi * cth + q2 * cphi + q3 * sphi * sth,
        -q1 * sth + q3 * cth
    };
}

// trp_rotate_invar of fks_singular.f, the inverse of fks_rotate
template <typename T>
KERNELSPEC FourMom<T>
fks_rotate_inverse(FourMom<T> q, Quartuplet<FVal<T>, FVal<T>, FVal<T>, FVal<T>> a) {
    auto cth = a.first, sth = a.second, cphi = a.third, sphi = a.fourth;
    return {
        q[0],
        q[1] * cphi * cth + q[2] * sphi * cth - q[3] * sth,
        -q[1] * sphi + q[2] * cphi,
        q[1] * cphi * sth + q[2] * sphi * sth + q[3] * cth
    };
}

// det_xi of the massless final-state emission: 2 pi from the azimuth, xi_max
// from xi = xi_hat xi_max and the phase-space factor of genps_fks.f
template <typename T>
KERNELSPEC FVal<T> fks_final_massless_det_xi(
    FVal<T> s, FVal<T> xi, FVal<T> xi_max, FVal<T> y, FVal<T> kj, FVal<T> kbar
) {
    constexpr double norm = 2. * PI * 2. / ((4. * PI) * (4. * PI) * (4. * PI));
    return norm * xi_max * s * kj / max(kbar, EPS) / (2. - xi * (1. - y));
}

template <typename T>
KERNELSPEC void kernel_fks_xi_y_sample(
    FIn<T, 0> r_xi, FIn<T, 0> r_y, FOut<T, 0> xi_hat, FOut<T, 0> y, FOut<T, 0> det
) {
    FVal<T> rx = r_xi, ry = r_y;
    xi_hat = FKS_STINY + (1. - FKS_STINY) * rx * rx;
    y = 1. - 2. * (FKS_CTINY + (1. - FKS_CTINY) * ry * ry);
    det = 8. * (1. - FKS_STINY) * (1. - FKS_CTINY) * rx * ry;
}

template <typename T>
KERNELSPEC void kernel_fks_xi_y_sample_inverse(
    FIn<T, 0> xi_hat, FIn<T, 0> y, FOut<T, 0> r_xi, FOut<T, 0> r_y, FOut<T, 0> det
) {
    FVal<T> xh = xi_hat, yv = y;
    auto rx = sqrt(max((xh - FKS_STINY) / (1. - FKS_STINY), 0.));
    auto ry = sqrt(max((0.5 * (1. - yv) - FKS_CTINY) / (1. - FKS_CTINY), 0.));
    r_xi = rx;
    r_y = ry;
    det = 1. / max(8. * (1. - FKS_STINY) * (1. - FKS_CTINY) * rx * ry, EPS2);
}

template <typename T>
KERNELSPEC void kernel_fks_final_massless_radiate(
    FIn<T, 0> xi_hat,
    FIn<T, 0> y,
    FIn<T, 0> r_phi,
    FIn<T, 0> sqrt_s,
    FIn<T, 1> p_jbar,
    FOut<T, 1> p_i,
    FOut<T, 1> p_j,
    FOut<T, 0> xi,
    FOut<T, 0> xi_max,
    FOut<T, 0> det_xi,
    FOut<T, 1> p_i_hat
) {
    FVal<T> xh = xi_hat, yv = y, rs = sqrt_s, rp = r_phi;
    auto pjb = load_mom<T>(p_jbar);
    auto s = rs * rs;
    auto kbar = sqrt(max(pjb[1] * pjb[1] + pjb[2] * pjb[2] + pjb[3] * pjb[3], EPS2));
    auto erec = rs - pjb[0];
    auto mrec2 = erec * erec - kbar * kbar;
    auto xm = 1. - mrec2 / s;
    auto xiv = xh * xm;
    auto phi = 2. * PI * rp;

    // energy of i, three-momentum of j and of the pair i + j
    auto e_i = 0.5 * xiv * rs;
    auto kj = (s - mrec2 - 2. * rs * e_i) / (2. * (rs - e_i * (1. - yv)));
    auto km = sqrt(max(e_i * e_i + kj * kj + 2. * e_i * kj * yv, 0.));

    // Angle of i to the pair from the components of i along and transverse to
    // i + j. genps_fks.f takes its cosine from the law of cosines, which loses
    // up to half the digits of a small angle (series in the soft and collinear
    // limits only partly restore them), these expressions are exact in every
    // limit.
    auto km_inv = 1. / max(km, EPS);
    auto cth = (e_i + kj * yv) * km_inv;
    auto sth = kj * sqrt(max((1. - yv) * (1. + yv), 0.)) * km_inv;

    // i and j about the +z axis, then rotated onto the Born emitter
    auto ix = e_i * sth * cos(phi);
    auto iy = e_i * sth * sin(phi);
    auto iz = e_i * cth;
    auto jz = kj * (kj + e_i * yv) * km_inv;
    auto rot = fks_angles<T>(pjb);
    store_mom<T>(p_i, fks_rotate<T>(e_i, ix, iy, iz, rot));
    store_mom<T>(p_j, fks_rotate<T>(kj, -ix, -iy, jz, rot));
    // i per unit xi, with the direction it has at this xi (that of the soft
    // limit at xi = 0)
    auto eh = 0.5 * rs;
    store_mom<T>(
        p_i_hat,
        fks_rotate<T>(eh, eh * sth * cos(phi), eh * sth * sin(phi), eh * cth, rot)
    );
    xi = xiv;
    xi_max = xm;
    det_xi = fks_final_massless_det_xi<T>(s, xiv, xm, yv, kj, kbar);
}

template <typename T>
KERNELSPEC void kernel_fks_final_massless_cluster(
    FIn<T, 1> p_i,
    FIn<T, 1> p_j,
    FIn<T, 0> sqrt_s,
    FOut<T, 1> p_jbar,
    FOut<T, 0> xi_hat,
    FOut<T, 0> y,
    FOut<T, 0> r_phi,
    FOut<T, 0> xi,
    FOut<T, 0> xi_max,
    FOut<T, 0> det_xi
) {
    FVal<T> rs = sqrt_s;
    auto pi = load_mom<T>(p_i);
    auto pj = load_mom<T>(p_j);
    auto s = rs * rs;
    FourMom<T> k{pi[0] + pj[0], pi[1] + pj[1], pi[2] + pj[2], pi[3] + pj[3]};
    auto klen = sqrt(max(k[1] * k[1] + k[2] * k[2] + k[3] * k[3], EPS2));
    auto erec = rs - k[0];
    auto mrec2 = erec * erec - klen * klen;
    auto xm = 1. - mrec2 / s;
    auto xiv = 2. * pi[0] / rs;
    auto li = sqrt(max(pi[1] * pi[1] + pi[2] * pi[2] + pi[3] * pi[3], EPS2));
    auto lj = sqrt(max(pj[1] * pj[1] + pj[2] * pj[2] + pj[3] * pj[3], EPS2));
    auto yv = (pi[1] * pj[1] + pi[2] * pj[2] + pi[3] * pj[3]) / (li * lj);
    yv = max(min(yv, FVal<T>(1.)), FVal<T>(-1.));

    auto kbar = 0.5 * (s - mrec2) / rs;
    FourMom<T> pjb{kbar, kbar * k[1] / klen, kbar * k[2] / klen, kbar * k[3] / klen};
    auto local = fks_rotate_inverse<T>(pi, fks_angles<T>(pjb));
    auto rphi = atan2(local[2], local[1]) / (2. * PI);
    rphi = where(rphi < 0., rphi + 1., rphi);

    store_mom<T>(p_jbar, pjb);
    xi_hat = xiv / max(xm, EPS);
    y = yv;
    r_phi = rphi;
    xi = xiv;
    xi_max = xm;
    det_xi = 1. / max(fks_final_massless_det_xi<T>(s, xiv, xm, yv, lj, kbar), EPS2);
}

// Boost along the direction of the FKS pair k that takes the Born recoil
// (sqrt_s, 0) - p_jbar to the recoil (sqrt_s, 0) - k of the real emission at
// fixed invariant mass, as genps_fks.f with boostwdir2: the light-cone
// components E + |p| of the two recoils along -k differ by the factor
// exp(eta) of the boost (the one of the Born is sqrt_s for a massless
// emitter). sign = -1 gives the inverse.
template <typename T>
KERNELSPEC void fks_recoil_boost(
    FIn<T, 2> p,
    FIn<T, 1> k,
    FIn<T, 1> p_jbar,
    FVal<T> rs,
    double sign,
    FOut<T, 2> p_out
) {
    auto pjb = load_mom<T>(p_jbar);
    auto klen = sqrt(max(k[1] * k[1] + k[2] * k[2] + k[3] * k[3], EPS2));
    auto kbar = sqrt(max(pjb[1] * pjb[1] + pjb[2] * pjb[2] + pjb[3] * pjb[3], EPS2));
    auto born = rs - (pjb[0] - kbar);
    auto real = rs - (k[0] - klen);
    auto den = 2. * real * born;
    auto sh = -sign * (born - real) * (born + real) / den;
    auto ch = (born * born + real * real) / den;
    auto chmo = (born - real) * (born - real) / den;
    auto d1 = k[1] / klen, d2 = k[2] / klen, d3 = k[3] / klen;
    for (std::size_t i = 0; i < p.size(); ++i) {
        auto p_i = p[i];
        auto p_out_i = p_out[i];
        FVal<T> en = p_i[0], q1 = p_i[1], q2 = p_i[2], q3 = p_i[3];
        auto pz = q1 * d1 + q2 * d2 + q3 * d3;
        auto shift = pz * chmo - en * sh;
        p_out_i[0] = en * ch - pz * sh;
        p_out_i[1] = q1 + d1 * shift;
        p_out_i[2] = q2 + d2 * shift;
        p_out_i[3] = q3 + d3 * shift;
    }
}

template <typename T>
KERNELSPEC void kernel_fks_recoil_boost(
    FIn<T, 2> p, FIn<T, 1> k, FIn<T, 1> p_jbar, FIn<T, 0> sqrt_s, FOut<T, 2> p_out
) {
    fks_recoil_boost<T>(p, k, p_jbar, sqrt_s, 1., p_out);
}

template <typename T>
KERNELSPEC void kernel_fks_recoil_boost_inverse(
    FIn<T, 2> p, FIn<T, 1> k, FIn<T, 1> p_jbar, FIn<T, 0> sqrt_s, FOut<T, 2> p_out
) {
    fks_recoil_boost<T>(p, k, p_jbar, sqrt_s, -1., p_out);
}

// Emission off a massive final-state parton, as
// generate_momenta_massive_final of genps_fks.f. Energies, momenta and masses
// are in units of sqrt_s: m_j of the emitter, m_rec of the system recoiling
// against it, lambda = lambda(1, m_rec^2, m_j^2), xi = 2 E_i / sqrt_s.

// m_j, m_rec and lambda from the mass m_j of the emitter and the invariant
// mass squared of the recoil
template <typename T>
KERNELSPEC Triplet<FVal<T>, FVal<T>, FVal<T>>
fks_massive_masses(FVal<T> rs, FVal<T> mrec2, FVal<T> m_j) {
    auto mh = sqrt(max(mrec2, 0.)) / rs;
    auto mj = m_j / rs;
    auto dm = mh - mj, sm = mh + mj;
    auto lam = max((1. - dm) * (1. + dm) * (1. - sm) * (1. + sm), 0.);
    return {mj, mh, lam};
}

// Range of xi at fixed y. j comes to rest at xi_m = ((1 - m_j)^2 - m_rec^2) /
// (1 - m_j). The momentum of j solves a quadratic equation with discriminant
// D(xi) = A (xi_1 - xi)(xi_2 - xi), A = 1 - m_j^2 (1 - y^2), with roots
// xi_2 - xi_1 = 2 sqrt_delta / A, sqrt_delta = m_j sqrt(4 m_rec^2 + (1 - y^2)
// lambda), and D(xi_m) = (m_j xi_m y)^2. So u = xi_1 - xi_m >= 0 solves
// A u^2 + 2 sqrt_delta u = (m_j xi_m y)^2, which gives it without
// cancellations (genps_fks.f takes xi_1 from the quadratic formula, which
// cancels near threshold, and its distance to xi_m, which matters near both,
// as a difference). As genps_fks.f, xi_hat runs along xi from 0 to xi_max
// (xi_m for y >= 0, else xi_1) with the first solution and, for y < 0, back
// to xi_m with the second one, a path of length xi_norm, and
// rat = xi_max / xi_norm is where it turns. Returns xi_m, u, sqrt_delta,
// xi_max, xi_norm and rat.
template <typename T>
KERNELSPEC Sextuplet<FVal<T>, FVal<T>, FVal<T>, FVal<T>, FVal<T>, FVal<T>>
fks_massive_range(FVal<T> y, FVal<T> mj, FVal<T> mh, FVal<T> lam) {
    auto omy2 = max((1. - y) * (1. + y), 0.);
    auto sq = mj * sqrt(4. * mh * mh + omy2 * lam);
    auto xim = max((1. - mj - mh) * (1. - mj + mh) / (1. - mj), 0.);
    auto a = 1. - mj * mj * omy2;
    auto w2 = (mj * xim * y) * (mj * xim * y);
    auto u = w2 / max(sq + sqrt(sq * sq + a * w2), EPS2);
    auto pos = y >= 0.;
    auto xm = where(pos, xim, xim + u);
    auto xn = where(pos, xim, xim + 2. * u);
    auto rat = where(pos, FVal<T>(1.), xm / max(xn, EPS2));
    return {xim, u, sq, xm, xn, rat};
}

// det_xi of the massive final-state emission: 2 pi from the azimuth, xi_norm
// from xi = xi_norm xi_hat and the phase-space factor of genps_fks.f,
// 2 s / (4 pi)^3 |k_j| / |kbar_j| / (2 - xi (1 - E_j y / |k_j|)), with
// 2 - xi (1 - E_j y / |k_j|) = +-sqrt(D) / |k_j|
template <typename T>
KERNELSPEC FVal<T> fks_massive_det_xi(
    FVal<T> s, FVal<T> xi_norm, FVal<T> kj, FVal<T> kbar, FVal<T> sqrt_d
) {
    constexpr double norm = 2. * PI * 2. / ((4. * PI) * (4. * PI) * (4. * PI));
    return norm * s * xi_norm * kj / max(kbar, EPS2) * kj / max(sqrt_d, EPS2);
}

// Born emitter -> m_j, m_rec, lambda
template <typename T>
KERNELSPEC Triplet<FVal<T>, FVal<T>, FVal<T>>
fks_massive_born(FVal<T> rs, FourMom<T> pjb, FVal<T> m_j) {
    auto kbar = sqrt(max(pjb[1] * pjb[1] + pjb[2] * pjb[2] + pjb[3] * pjb[3], EPS2));
    auto erec = rs - pjb[0];
    return fks_massive_masses<T>(rs, (erec - kbar) * (erec + kbar), m_j);
}

template <typename T>
KERNELSPEC void kernel_fks_final_massive_sample(
    FIn<T, 0> r_xi,
    FIn<T, 0> r_y,
    FIn<T, 0> sqrt_s,
    FIn<T, 1> p_jbar,
    FIn<T, 0> m_j,
    FOut<T, 0> xi_hat,
    FOut<T, 0> y,
    FOut<T, 0> det
) {
    FVal<T> rx = r_xi, ry = r_y, rs = sqrt_s;
    auto yv = 1. - 2. * (FKS_CTINY + (1. - FKS_CTINY) * ry * ry);
    auto masses = fks_massive_born<T>(rs, load_mom<T>(p_jbar), FVal<T>(m_j));
    auto rat =
        fks_massive_range<T>(yv, masses.first, masses.second, masses.third).sixth;
    auto plus = rx <= rat;
    auto t = rx / max(rat, EPS2);
    xi_hat = where(plus, rat * (FKS_STINY + (1. - FKS_STINY) * t * t), rx);
    y = yv;
    det = where(plus, 2. * (1. - FKS_STINY) * t, FVal<T>(1.)) * 4. * (1. - FKS_CTINY) *
        ry;
}

template <typename T>
KERNELSPEC void kernel_fks_final_massive_sample_inverse(
    FIn<T, 0> xi_hat,
    FIn<T, 0> y,
    FIn<T, 0> sqrt_s,
    FIn<T, 1> p_jbar,
    FIn<T, 0> m_j,
    FOut<T, 0> r_xi,
    FOut<T, 0> r_y,
    FOut<T, 0> det
) {
    FVal<T> xh = xi_hat, yv = y, rs = sqrt_s;
    auto masses = fks_massive_born<T>(rs, load_mom<T>(p_jbar), FVal<T>(m_j));
    auto rat =
        fks_massive_range<T>(yv, masses.first, masses.second, masses.third).sixth;
    auto plus = xh <= rat;
    auto t = sqrt(max((xh / max(rat, EPS2) - FKS_STINY) / (1. - FKS_STINY), 0.));
    auto ry = sqrt(max((0.5 * (1. - yv) - FKS_CTINY) / (1. - FKS_CTINY), 0.));
    r_xi = where(plus, rat * t, xh);
    r_y = ry;
    det = 1. /
        max(where(plus, 2. * (1. - FKS_STINY) * t, FVal<T>(1.)) * 4. *
                (1. - FKS_CTINY) * ry,
            EPS2);
}

template <typename T>
KERNELSPEC void kernel_fks_final_massive_radiate(
    FIn<T, 0> xi_hat,
    FIn<T, 0> y,
    FIn<T, 0> r_phi,
    FIn<T, 0> sqrt_s,
    FIn<T, 1> p_jbar,
    FIn<T, 0> m_j,
    FOut<T, 1> p_i,
    FOut<T, 1> p_j,
    FOut<T, 0> xi,
    FOut<T, 0> xi_max,
    FOut<T, 0> xi_norm,
    FOut<T, 0> det_xi,
    FOut<T, 1> p_i_hat
) {
    FVal<T> xh = xi_hat, yv = y, rs = sqrt_s, rp = r_phi;
    auto pjb = load_mom<T>(p_jbar);
    auto masses = fks_massive_born<T>(rs, pjb, FVal<T>(m_j));
    auto mj = masses.first, mh = masses.second, lam = masses.third;
    auto range = fks_massive_range<T>(yv, mj, mh, lam);
    auto xim = range.first, u = range.second, sq = range.third;
    auto xm = range.fourth, xn = range.fifth, rat = range.sixth;

    // xi along the path, and xi_1 - xi and xi_m - xi without cancellations
    auto omy2 = max((1. - yv) * (1. + yv), 0.);
    auto a = 1. - mj * mj * omy2;
    auto pos = yv >= 0.;
    auto plus = xh <= rat;
    auto xiv = where(plus, xn * xh, 2. * (xim + u) - xn * xh);
    auto d1 = where(
        plus, where(pos, u + xim * (1. - xh), xn * (1. - xh) - u), u - xn * (1. - xh)
    );
    auto dm = where(plus, where(pos, xim * (1. - xh), xim - xiv), -xn * (1. - xh));

    // |k_j| from energy conservation, the two solutions
    // (-c xi y +- (2 - xi) sqrt(D)) / ((2 - xi (1 - y)) (2 - xi (1 + y))) with
    // c = 1 - xi + m_j^2 - m_rec^2 and D = c^2 - (2 - xi (1 - y)) (2 - xi (1 + y))
    // m_j^2 = (xi_1 - xi) (A (xi_1 - xi) + 2 sqrt_delta). Where the two terms
    // cancel, |k_j| = (1 - m_j) (xi_m - xi) (c + (2 - xi) m_j) /
    // (c xi y +- (2 - xi) sqrt(D)) instead, which vanishes at xi_m.
    auto dd = max(d1 * (a * d1 + 2. * sq), 0.);
    auto sqd = sqrt(dd);
    auto c = (1. - xiv) + (mj - mh) * (mj + mh);
    auto den = (2. - xiv * (1. - yv)) * (2. - xiv * (1. + yv));
    auto q1 = -c * xiv * yv;
    auto q2 = where(plus, FVal<T>(1.), FVal<T>(-1.)) * (2. - xiv) * sqd;
    auto direct = q1 * q2 >= 0.;
    auto num = where(direct, q1 + q2, (1. - mj) * dm * (c + (2. - xiv) * mj));
    auto kj = num / where(direct, den, q2 - q1); // q2 - q1 != 0 where not direct

    // energy of i, momentum of the pair i + j and angle of i to it, as for a
    // massless emitter
    auto ei = 0.5 * xiv;
    auto km2 = where(
        yv < 0.,
        (ei - kj) * (ei - kj) + 2. * ei * kj * (1. + yv),
        ei * ei + kj * kj + 2. * ei * kj * yv
    );
    auto km_inv = 1. / max(sqrt(max(km2, 0.)), EPS2);
    auto cth = (ei + kj * yv) * km_inv;
    auto sth = kj * sqrt(omy2) * km_inv;
    auto phi = 2. * PI * rp;
    auto ix = rs * ei * sth * cos(phi);
    auto iy = rs * ei * sth * sin(phi);
    auto iz = rs * ei * cth;
    auto jz = rs * kj * (kj + ei * yv) * km_inv;
    auto ej = rs * sqrt(kj * kj + mj * mj);
    auto rot = fks_angles<T>(pjb);
    store_mom<T>(p_i, fks_rotate<T>(rs * ei, ix, iy, iz, rot));
    store_mom<T>(p_j, fks_rotate<T>(ej, -ix, -iy, jz, rot));
    // i per unit xi, with the direction it has at this xi
    auto eh = 0.5 * rs;
    store_mom<T>(
        p_i_hat,
        fks_rotate<T>(eh, eh * sth * cos(phi), eh * sth * sin(phi), eh * cth, rot)
    );
    auto kbar =
        sqrt(max(pjb[1] * pjb[1] + pjb[2] * pjb[2] + pjb[3] * pjb[3], EPS2)) / rs;
    xi = xiv;
    xi_max = xm;
    xi_norm = xn;
    det_xi = fks_massive_det_xi<T>(rs * rs, xn, kj, kbar, sqd);
}

template <typename T>
KERNELSPEC void kernel_fks_final_massive_cluster(
    FIn<T, 1> p_i,
    FIn<T, 1> p_j,
    FIn<T, 0> sqrt_s,
    FIn<T, 0> m_j,
    FOut<T, 1> p_jbar,
    FOut<T, 0> xi_hat,
    FOut<T, 0> y,
    FOut<T, 0> r_phi,
    FOut<T, 0> xi,
    FOut<T, 0> xi_max,
    FOut<T, 0> xi_norm,
    FOut<T, 0> det_xi
) {
    FVal<T> rs = sqrt_s;
    auto pi = load_mom<T>(p_i);
    auto pj = load_mom<T>(p_j);
    FourMom<T> k{pi[0] + pj[0], pi[1] + pj[1], pi[2] + pj[2], pi[3] + pj[3]};
    auto klen = sqrt(max(k[1] * k[1] + k[2] * k[2] + k[3] * k[3], EPS2));
    auto erec = rs - k[0];
    auto masses =
        fks_massive_masses<T>(rs, (erec - klen) * (erec + klen), FVal<T>(m_j));
    auto mj = masses.first, mh = masses.second, lam = masses.third;
    auto xiv = 2. * pi[0] / rs;
    auto li = sqrt(max(pi[1] * pi[1] + pi[2] * pi[2] + pi[3] * pi[3], EPS2));
    auto lj = sqrt(max(pj[1] * pj[1] + pj[2] * pj[2] + pj[3] * pj[3], EPS2));
    auto yv = (pi[1] * pj[1] + pi[2] * pj[2] + pi[3] * pj[3]) / (li * lj);
    yv = max(min(yv, FVal<T>(1.)), FVal<T>(-1.));
    auto range = fks_massive_range<T>(yv, mj, mh, lam);
    auto xi1 = range.first + range.second, xm = range.fourth, xn = range.fifth;

    // the solution from the sign of (2 - xi) |k_j| + xi y E_j = +-sqrt(D)
    auto kj = lj / rs;
    auto bal = (2. - xiv) * kj + xiv * yv * pj[0] / rs;
    auto plus = bal >= 0.;
    auto xn_safe = max(xn, EPS2);

    // Born emitter along i + j
    auto kbar = 0.5 * sqrt(lam);
    auto ejb = 0.5 * (1. + (mj - mh) * (mj + mh));
    auto scale = rs * kbar / klen;
    FourMom<T> pjb{rs * ejb, scale * k[1], scale * k[2], scale * k[3]};
    auto local = fks_rotate_inverse<T>(pi, fks_angles<T>(pjb));
    auto rphi = atan2(local[2], local[1]) / (2. * PI);

    store_mom<T>(p_jbar, pjb);
    xi_hat = where(plus, xiv, 2. * xi1 - xiv) / xn_safe;
    y = yv;
    r_phi = where(rphi < 0., rphi + 1., rphi);
    xi = xiv;
    xi_max = xm;
    xi_norm = xn;
    det_xi = 1. / max(fks_massive_det_xi<T>(rs * rs, xn, kj, kbar, fabs(bal)), EPS2);
}

// Emission off an incoming parton. y_beam = beam_sign y is the cosine of the
// angle between i and the +z axis in the centre-of-mass frame of the real
// emission.

// Upper bound of xi from x <= 1 for the beam with Born fraction x, with
// omy = 1 - y and opy = 1 + y for the cosine y of i to the direction of that
// beam: smaller root of (1 + y) xi^2 - b xi + 2 (1 - x^2) = 0. Written without
// cancellations, b = 2 (1 - x^2) + (1 + x^2)(1 + y) and the discriminant is
// ((1 + x^2)(1 - y))^2 + 16 y x^2 (as genps_fks.f) for y >= 0 and
// ((1 + x^2)(1 + y))^2 - 4 y (1 - x^2)^2 for y < 0. Both forms of genps_fks.f
// cancel for y -> -1 and x -> 1, which loses up to 1e-4 relative there.
template <typename T>
KERNELSPEC FVal<T> fks_initial_xi_max_beam(FVal<T> x, FVal<T> omy, FVal<T> opy) {
    auto x2 = x * x;
    auto omx2 = (1. - x) * (1. + x);
    auto opx2 = 1. + x2;
    auto y = 0.5 * (opy - omy);
    auto b = 2. * omx2 + opx2 * opy;
    auto u = opx2 * omy;
    auto v = opx2 * opy;
    auto disc = where(y >= 0., u * u + 16. * y * x2, v * v - 4. * y * omx2 * omx2);
    return 4. * omx2 / max(b + sqrt(max(disc, 0.)), EPS);
}

template <typename T>
KERNELSPEC FVal<T>
fks_initial_xi_max(FVal<T> x1, FVal<T> x2, FVal<T> omy_beam, FVal<T> opy_beam) {
    return min(
        fks_initial_xi_max_beam<T>(x1, omy_beam, opy_beam),
        fks_initial_xi_max_beam<T>(x2, opy_beam, omy_beam)
    );
}

// det_xi of the initial-state emission: 2 pi from the azimuth, xi_max from
// xi = xi_hat xi_max, s / (4 pi)^3 from d^3k / (2 E_k (2 pi)^3) and
// 1 / (1 - xi) from dx1 dx2 = dx1_born dx2_born / (1 - xi)
template <typename T>
KERNELSPEC FVal<T> fks_initial_det_xi(FVal<T> s_born, FVal<T> xi_width, FVal<T> omx) {
    constexpr double norm = 2. * PI / ((4. * PI) * (4. * PI) * (4. * PI));
    return norm * xi_width * s_born / max(omx * omx, EPS2);
}

// Range of the emission off an incoming parton with a lower bound s_hat_min on
// the partonic energy of the real emission, as tau_lower_bound in
// genps_fks.f: xi >= xi_min = 1 - s_born / s_hat_min, and y restricted to where
// xi_max(y) >= xi_min. With rho = 1 - xi_min, x1 <= 1 at xi_min bounds the
// cosine y_beam of i to +z from above and x2 <= 1 from below,
//   1 + y_upp = 2 (1 - x1^2) / (xi_min (1 + x1^2 / rho)),
//   1 - y_upp = 2 (x1^2 - rho^2) / (rho xi_min (1 + x1^2 / rho)),
//   1 + y_low = 2 (x2^2 - rho^2) / (xi_min (x2^2 + rho)),
//   1 - y_low = 2 rho (1 - x2^2) / (xi_min (x2^2 + rho)),
// each clamped to [-1, 1]. Returns xi_min, 1 -+ y at the upper and lower end
// of the range of y, the cosine of i to the emitter j, and the width
// y_upp - y_low of the range.
template <typename T>
KERNELSPEC Sextuplet<FVal<T>, FVal<T>, FVal<T>, FVal<T>, FVal<T>, FVal<T>>
fks_initial_range(
    FVal<T> s_born, FVal<T> x1b, FVal<T> x2b, FVal<T> s_hat_min, FVal<T> sign
) {
    auto active = s_hat_min > s_born;
    auto s_min = where(active, s_hat_min, FVal<T>(1.));
    auto rho = where(active, s_born / s_min, FVal<T>(1.));
    auto xi_min = where(active, (s_min - s_born) / s_min, FVal<T>(0.));
    auto xi_min_safe = where(active, xi_min, FVal<T>(1.));
    auto d1 = xi_min_safe * (1. + x1b * x1b / rho);
    auto d2 = xi_min_safe * (x2b * x2b + rho);
    // the smaller of 1 -+ y at each end from its formula and the larger one as
    // 2 minus it, so that 1 - y and 1 + y add up to 2 inside the range
    auto omy_uf =
        min(max(2. * (x1b - rho) * (x1b + rho) / (rho * d1), 0.), FVal<T>(2.));
    auto opy_uf = min(2. * (1. - x1b) * (1. + x1b) / d1, FVal<T>(2.));
    auto low_u = omy_uf <= opy_uf;
    auto omy_u = where(active, where(low_u, omy_uf, 2. - opy_uf), FVal<T>(0.));
    auto opy_u = where(active, where(low_u, 2. - omy_uf, opy_uf), FVal<T>(2.));
    auto opy_lf = min(max(2. * (x2b - rho) * (x2b + rho) / d2, 0.), FVal<T>(2.));
    auto omy_lf = min(2. * rho * (1. - x2b) * (1. + x2b) / d2, FVal<T>(2.));
    auto low_l = opy_lf <= omy_lf;
    auto opy_l = where(active, where(low_l, opy_lf, 2. - omy_lf), FVal<T>(0.));
    auto omy_l = where(active, where(low_l, 2. - opy_lf, omy_lf), FVal<T>(2.));
    auto plus = sign > 0.;
    auto omy_hi = where(plus, omy_u, opy_l);
    auto opy_hi = where(plus, opy_u, omy_l);
    auto omy_lo = where(plus, omy_l, opy_u);
    auto opy_lo = where(plus, opy_l, omy_u);
    // width of the range of y from the smaller pair
    auto width = where(omy_lo <= opy_hi, omy_lo - omy_hi, opy_hi - opy_lo);
    return {xi_min, omy_hi, opy_hi, omy_lo, opy_lo, max(width, 0.)};
}

template <typename T>
KERNELSPEC void kernel_fks_initial_radiate(
    FIn<T, 0> xi_hat,
    FIn<T, 0> y_hat,
    FIn<T, 0> r_phi,
    FIn<T, 0> sqrt_s,
    FIn<T, 0> x1_born,
    FIn<T, 0> x2_born,
    FIn<T, 0> s_hat_min,
    FIn<T, 0> beam_sign,
    FOut<T, 1> p_in1,
    FOut<T, 1> p_in2,
    FOut<T, 1> p_i,
    FOut<T, 0> x1,
    FOut<T, 0> x2,
    FOut<T, 0> xi,
    FOut<T, 0> y,
    FOut<T, 0> xi_min,
    FOut<T, 0> xi_max,
    FOut<T, 0> det_xi,
    FOut<T, 1> p_i_hat
) {
    FVal<T> xh = xi_hat, yh = y_hat, rp = r_phi, rs_born = sqrt_s;
    FVal<T> x1b = x1_born, x2b = x2_born, sign(beam_sign);
    auto s_born = rs_born * rs_born;

    // y_hat in [-1, 1] mapped linearly onto the range of y, keeping 1 - y and
    // 1 + y as sums of non-negative terms (y = y_hat without a bound)
    auto range = fks_initial_range<T>(s_born, x1b, x2b, FVal<T>(s_hat_min), sign);
    auto xmin = range.first;
    auto width_y = range.sixth;
    auto omy = range.second + width_y * 0.5 * (1. - yh);
    auto opy = range.fifth + width_y * 0.5 * (1. + yh);
    auto plus = sign > 0.;
    auto omy_beam = where(plus, omy, opy);
    auto opy_beam = where(plus, opy, omy);
    auto yb = 0.5 * (opy_beam - omy_beam);

    auto xm = fks_initial_xi_max<T>(x1b, x2b, omy_beam, opy_beam);
    auto width_xi = max(xm - xmin, 0.);
    auto xiv = xmin + width_xi * xh;
    auto omx = 1. - xiv;
    auto sq_omx = sqrt(max(omx, EPS2));
    auto rs = rs_born / sq_omx;

    // 2 - xi (1 -+ y_beam) and the longitudinal boost from the centre-of-mass
    // frame of the real emission into the Born one
    auto om_m = 2. - xiv * omy_beam;
    auto om_p = 2. - xiv * opy_beam;
    auto bst = sqrt(om_m * om_p);
    auto omega = sqrt(om_p / om_m);
    auto sin2 = max(omy_beam * opy_beam, 0.);
    auto sth = sqrt(sin2);
    auto phi = 2. * PI * rp;

    auto e1 = 0.5 * rs * om_m / bst;
    auto e2 = 0.5 * rs * om_p / bst;
    store_mom<T>(p_in1, FourMom<T>{e1, FVal<T>(0.), FVal<T>(0.), e1});
    store_mom<T>(p_in2, FourMom<T>{e2, FVal<T>(0.), FVal<T>(0.), -e2});
    auto e_i = 0.5 * xiv * rs;
    store_mom<T>(
        p_i,
        FourMom<T>{
            e_i * (2. - xiv * sin2) / bst,
            e_i * sth * cos(phi),
            e_i * sth * sin(phi),
            2. * e_i * yb / bst
        }
    );
    // i per unit xi, in the same frame
    auto eh = 0.5 * rs;
    store_mom<T>(
        p_i_hat,
        FourMom<T>{
            eh * (2. - xiv * sin2) / bst,
            eh * sth * cos(phi),
            eh * sth * sin(phi),
            2. * eh * yb / bst
        }
    );
    x1 = x1b / (sq_omx * omega);
    x2 = x2b * omega / sq_omx;
    xi = xiv;
    y = where(omy <= opy, 1. - omy, opy - 1.);
    xi_min = xmin;
    xi_max = xm;
    det_xi = fks_initial_det_xi<T>(s_born, width_xi, omx) * 0.5 * width_y;
}

template <typename T>
KERNELSPEC void kernel_fks_initial_cluster(
    FIn<T, 1> p_in1,
    FIn<T, 1> p_in2,
    FIn<T, 1> p_i,
    FIn<T, 0> x1,
    FIn<T, 0> x2,
    FIn<T, 0> s_hat_min,
    FIn<T, 0> beam_sign,
    FOut<T, 0> sqrt_s,
    FOut<T, 0> x1_born,
    FOut<T, 0> x2_born,
    FOut<T, 0> xi_hat,
    FOut<T, 0> y_hat,
    FOut<T, 0> r_phi,
    FOut<T, 0> xi,
    FOut<T, 0> y,
    FOut<T, 0> xi_min,
    FOut<T, 0> xi_max,
    FOut<T, 0> det_xi
) {
    FVal<T> sign(beam_sign);
    auto p1 = load_mom<T>(p_in1);
    auto p2 = load_mom<T>(p_in2);
    auto pi = load_mom<T>(p_i);
    auto e1 = p1[0], e2 = p2[0];
    auto s = 4. * e1 * e2;

    // light-cone components of i, the smaller one from its mass shell
    auto pt2 = pi[1] * pi[1] + pi[2] * pi[2];
    auto forward = pi[3] >= 0.;
    auto big = where(forward, pi[0] + pi[3], pi[0] - pi[3]);
    auto small = pt2 / max(big, EPS);
    auto plus = where(forward, big, small);
    auto minus = where(forward, small, big);
    auto pi_dot_1 = e1 * minus;
    auto pi_dot_2 = e2 * plus;
    auto pi_dot_p = pi_dot_1 + pi_dot_2;
    auto xiv = 2. * pi_dot_p / s;
    auto omy_beam = 2. * pi_dot_1 / pi_dot_p; // 1 - y_beam
    auto opy_beam = 2. * pi_dot_2 / pi_dot_p; // 1 + y_beam

    auto omx = 1. - xiv;
    auto sq_omx = sqrt(max(omx, EPS2));
    auto om_m = 2. - xiv * omy_beam;
    auto om_p = 2. - xiv * opy_beam;
    auto omega = sqrt(om_p / om_m);
    FVal<T> xr1 = x1, xr2 = x2;
    auto x1b = xr1 * sq_omx * omega;
    auto x2b = xr2 * sq_omx / omega;
    auto xm = fks_initial_xi_max<T>(x1b, x2b, omy_beam, opy_beam);
    auto s_born = s * omx;

    auto range = fks_initial_range<T>(s_born, x1b, x2b, FVal<T>(s_hat_min), sign);
    auto xmin = range.first;
    auto width_y = range.sixth;
    auto positive = sign > 0.;
    auto omy = where(positive, omy_beam, opy_beam);
    auto opy = where(positive, opy_beam, omy_beam);
    // distance from the upper end of the range of y, from the smaller pair
    auto from_upp = where(omy <= opy, omy - range.second, range.third - opy);
    auto width_xi = max(xm - xmin, 0.);

    auto rphi = atan2(pi[2], pi[1]) / (2. * PI);
    sqrt_s = sqrt(max(s_born, 0.));
    x1_born = x1b;
    x2_born = x2b;
    xi_hat = (xiv - xmin) / max(width_xi, EPS2);
    y_hat = 1. - 2. * from_upp / max(width_y, EPS2);
    r_phi = where(rphi < 0., rphi + 1., rphi);
    xi = xiv;
    y = where(omy <= opy, 1. - omy, opy - 1.);
    xi_min = xmin;
    xi_max = xm;
    det_xi =
        1. / max(fks_initial_det_xi<T>(s_born, width_xi, omx) * 0.5 * width_y, EPS2);
}

// Transverse boost of the final state of the Born along -(cos phi, sin phi, 0)
// with sinh = -xi sin(theta) / (2 sqrt(1 - xi)) and
// cosh = sqrt((2 - xi (1 - y)) (2 - xi (1 + y))) / (2 sqrt(1 - xi)), as
// genps_fks.f. sign = -1 gives the inverse.
template <typename T>
KERNELSPEC void fks_initial_recoil_boost(
    FIn<T, 2> p,
    FVal<T> xi,
    FVal<T> y,
    FVal<T> r_phi,
    FVal<T> beam_sign,
    double sign,
    FOut<T, 2> p_out
) {
    auto yb = beam_sign * y;
    auto sq_omx = sqrt(max(1. - xi, EPS2));
    auto sth = sqrt(max((1. - y) * (1. + y), 0.));
    auto sh = -sign * xi * sth / (2. * sq_omx);
    auto ch = sqrt((2. - xi * (1. - yb)) * (2. - xi * (1. + yb))) / (2. * sq_omx);
    auto chmo = sh * sh / (ch + 1.);
    auto phi = 2. * PI * r_phi;
    auto d1 = -cos(phi), d2 = -sin(phi);
    for (std::size_t i = 0; i < p.size(); ++i) {
        auto p_i = p[i];
        auto p_out_i = p_out[i];
        FVal<T> en = p_i[0], q1 = p_i[1], q2 = p_i[2], q3 = p_i[3];
        auto pz = q1 * d1 + q2 * d2;
        auto shift = pz * chmo - en * sh;
        p_out_i[0] = en * ch - pz * sh;
        p_out_i[1] = q1 + d1 * shift;
        p_out_i[2] = q2 + d2 * shift;
        p_out_i[3] = q3;
    }
}

template <typename T>
KERNELSPEC void kernel_fks_initial_recoil_boost(
    FIn<T, 2> p,
    FIn<T, 0> xi,
    FIn<T, 0> y,
    FIn<T, 0> r_phi,
    FIn<T, 0> beam_sign,
    FOut<T, 2> p_out
) {
    fks_initial_recoil_boost<T>(p, xi, y, r_phi, FVal<T>(beam_sign), 1., p_out);
}

template <typename T>
KERNELSPEC void kernel_fks_initial_recoil_boost_inverse(
    FIn<T, 2> p,
    FIn<T, 0> xi,
    FIn<T, 0> y,
    FIn<T, 0> r_phi,
    FIn<T, 0> beam_sign,
    FOut<T, 2> p_out
) {
    fks_initial_recoil_boost<T>(p, xi, y, r_phi, FVal<T>(beam_sign), -1., p_out);
}

// Light-cone components E + p_z and E - p_z of p_in1 + p_in2 - p_i, the system
// recoiling against i in an emission off an incoming parton. It has zero
// longitudinal momentum in the centre-of-mass frame of the Born, so their
// ratio is exp(2 y_born) for the rapidity y_born of that frame. The incoming
// momenta are along +-z. The smaller light-cone component of i comes from its
// mass shell, as in @ref fks_initial_cluster.
template <typename T>
KERNELSPEC void kernel_fks_recoil_light_cone(
    FIn<T, 1> p_in1, FIn<T, 1> p_in2, FIn<T, 1> p_i, FOut<T, 0> plus, FOut<T, 0> minus
) {
    auto p1 = load_mom<T>(p_in1);
    auto p2 = load_mom<T>(p_in2);
    auto pi = load_mom<T>(p_i);
    auto pt2 = pi[1] * pi[1] + pi[2] * pi[2];
    auto forward = pi[3] >= 0.;
    auto big = where(forward, pi[0] + pi[3], pi[0] - pi[3]);
    auto small = pt2 / max(big, EPS);
    plus = (p1[0] + p1[3]) + (p2[0] + p2[3]) - where(forward, big, small);
    minus = (p1[0] - p1[3]) + (p2[0] - p2[3]) - where(forward, small, big);
}

// Boost along z by the rapidity y = log(plus / minus) / 2, in light-cone
// components: q_+ = E + p_z -> e^{+y} q_+, q_- = E - p_z -> e^{-y} q_-, the
// larger of the two taken directly and the smaller from the mass shell,
// q_+ q_- = m^2 + p_T^2, with the given mass (a negative one means unknown, and
// that momentum is boosted linearly). Each momentum keeps its relative
// precision in both components, in either direction of the boost, so the
// centre-of-mass frame of the Born comes out of the lab frame as precisely as
// it went in, whatever the rapidity. (boost_beam projects onto the mass shell
// only the momenta whose energy decreases and hands the shifts to the hardest
// one, which loses up to e^{2|y|} of relative precision in that momentum when
// it is boosted out of the lab.)
template <typename T>
KERNELSPEC void kernel_fks_beam_boost(
    FIn<T, 2> p, FSIn<T, 1> masses, FIn<T, 0> plus, FIn<T, 0> minus, FOut<T, 2> p_out
) {
    FVal<T> pl = plus, mi = minus;
    auto e_plus = sqrt(pl / mi);
    auto e_minus = sqrt(mi / pl);
    auto cosh_rap = 0.5 * (e_plus + e_minus);
    auto sinh_rap = 0.5 * (e_plus - e_minus);
    for (std::size_t i = 0; i < p.size(); ++i) {
        auto q = p[i];
        auto q_out = p_out[i];
        FVal<T> mass(masses[i]);
        FVal<T> q0(q[0]), q1(q[1]), q2(q[2]), q3(q[3]);
        auto trans2 = mass * mass + q1 * q1 + q2 * q2;
        auto forward = q3 >= 0.;
        auto big = where(forward, q0 + q3, q0 - q3);
        auto small = trans2 / max(big, FVal<T>(EPS2));
        auto p_plus = e_plus * where(forward, big, small);
        auto p_minus = e_minus * where(forward, small, big);
        auto known = mass >= 0.;
        q_out[0] =
            where(known, 0.5 * (p_plus + p_minus), q0 * cosh_rap + q3 * sinh_rap);
        q_out[1] = q1;
        q_out[2] = q2;
        q_out[3] =
            where(known, 0.5 * (p_plus - p_minus), q3 * cosh_rap + q0 * sinh_rap);
    }
}

// Jacobian of a counter-event, zero where genps_fks.f does not generate it: a
// soft one needs xi = 0 (an incoming emitter with a lower bound on the partonic
// energy can have xi_min > 0) and the first solution for the momentum of a
// massive emitter (xi_hat <= rat), and a collinear one needs y = 1 (which the
// bound can exclude). A massive emitter has no collinear ones (exists = 0).
template <typename T>
KERNELSPEC void kernel_fks_counter_event_det(
    FIn<T, 0> det,
    FIn<T, 0> xi,
    FIn<T, 0> y,
    FIn<T, 0> xi_hat,
    FIn<T, 0> rat,
    FIn<T, 0> soft,
    FIn<T, 0> collinear,
    FIn<T, 0> exists,
    FOut<T, 0> det_out
) {
    FVal<T> d = det, xv = xi, yv = y, xh = xi_hat, rt = rat, s(soft), c(collinear);
    FVal<T> e(exists);
    auto soft_ok = (s == 0.) | ((xv == 0.) & (xh <= rt));
    auto collinear_ok = (c == 0.) | (yv == 1.);
    det_out = where((e != 0.) & soft_ok & collinear_ok, d, FVal<T>(0.));
}

// The FKS sector functions S_ij of fks_Sij.f. d_kl = (E_k E_l / (sqrt_s / 2)^2)^a
// (1 - beta_k beta_l cos theta_kl)^b, energies and angles in the partonic
// centre-of-mass frame of the momenta (written with invariants, so that the
// momenta can be in any frame), and S_ij = h / sum_kl d_ij / d_kl over the
// FKS pairs (k, l) of the real process, with h = h_damp(E_i / (E_i + E_j))
// for a final-state pair of two gluons or photons.

// energy of q in the rest frame of p_tot, of mass rs
template <typename T>
KERNELSPEC FVal<T> fks_frame_energy(FourMom<T> q, FourMom<T> p_tot, FVal<T> rs) {
    return (q[0] * p_tot[0] - q[1] * p_tot[1] - q[2] * p_tot[2] - q[3] * p_tot[3]) / rs;
}

// d_kl of get_dkl_Sij for the momenta q1, q2 with energies e1, e2 (for the
// energy factor; q1 can be the momentum of i per unit xi with the energy of
// i), masses m1, m2 and colour octets o1, o2. With both, the energy of 1 is
// in d_kl (d_ij / d_kl of a pair without i), else only that of 2. zero is set
// where an energy or the angle falls below vtiny, as get_dkl_Sij does.
template <typename T>
KERNELSPEC FVal<T> fks_dkl(
    FourMom<T> q1,
    FVal<T> e1,
    FVal<T> m1,
    bool o1,
    FourMom<T> q2,
    FVal<T> e2,
    FVal<T> m2,
    bool o2,
    bool both,
    FourMom<T> p_tot,
    FVal<T> half,
    FVal<T> fks_a,
    FVal<T> fks_b,
    bool use_energy,
    bool use_beta,
    bool& zero
) {
    constexpr double vtiny = 1e-8;
    zero = false;
    FVal<T> energy = 1.;
    if (both && (o1 || use_energy)) {
        energy = energy * e1 / half;
        if (e1 < vtiny * half) {
            zero = true;
            return 0.;
        }
    }
    if (o2 || use_energy) {
        energy = energy * e2 / half;
        if (e2 < vtiny * half) {
            zero = true;
            return 0.;
        }
    }
    if (energy > 0.) {
        energy = pow(energy, fks_a);
    }
    // the angle in the frame of p_tot: with q's own energy E and length k,
    // 1 - cos = (q1 q2) / (E1 E2) for massless momenta and 1 - beta_1 beta_2
    // cos = (q1 q2) / (E1 E2) for massive ones whose velocity counts
    auto rs = 2. * half;
    auto f1 = fks_frame_energy<T>(q1, p_tot, rs);
    auto f2 = fks_frame_energy<T>(q2, p_tot, rs);
    auto dot12 = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3];
    auto k1 = sqrt(max(f1 * f1 - m1 * m1, 0.));
    auto k2 = sqrt(max(f2 * f2 - m2 * m2, 0.));
    // g = 1 where beta enters (or the particle is massless), E / k otherwise
    FVal<T> g1 = (!o1 && m1 != 0. && use_beta) || m1 == 0. ? FVal<T>(1.) : f1 / k1;
    FVal<T> g2 = (!o2 && m2 != 0. && use_beta) || m2 == 0. ? FVal<T>(1.) : f2 / k2;
    auto c0 = dot12 / (f1 * f2);
    auto g12 = g1 * g2;
    FVal<T> angle = g12 == 1. ? c0 : 1. - g12 * (1. - c0);
    angle = max(angle, 0.);
    zero = angle < vtiny;
    if (angle > 0.) {
        angle = pow(angle, fks_b);
    }
    return energy * angle;
}

template <typename T>
KERNELSPEC FVal<T> fks_h_damp(FVal<T> x, FVal<T> one, FVal<T> a) {
    if (x <= one) {
        return 1.;
    }
    if (x >= 1. - one) {
        return 0.;
    }
    auto y = (x - one) / (1. - 2. * one);
    auto u = pow(1. - y, 2. * a), v = pow(y, 2. * a);
    return u / (u + v);
}

template <typename T>
KERNELSPEC void kernel_fks_sector_function(
    FIn<T, 2> p,
    FIn<T, 1> p_i_hat,
    IIn<T, 0> index,
    IIn<T, 1> fks_i,
    IIn<T, 1> fks_j,
    IIn<T, 3> pairs,
    IIn<T, 2> octet,
    IIn<T, 2> aorg,
    FIn<T, 2> masses,
    FIn<T, 1> options,
    FOut<T, 0> s
) {
    std::size_t c = single_index(index);
    std::size_t ii = single_index(fks_i[c]), jj = single_index(fks_j[c]);
    auto pair = pairs[c];
    auto oct = octet[c];
    auto ag = aorg[c];
    auto mass = masses[c];
    FVal<T> fks_a = options[0], fks_b = options[1];
    bool use_energy = FVal<T>(options[2]) != 0., use_beta = FVal<T>(options[3]) != 0.;
    FVal<T> h_one = options[4], h_a = options[5];

    auto p1 = load_mom<T>(p[0]), p2 = load_mom<T>(p[1]);
    FourMom<T> p_tot{p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2], p1[3] + p2[3]};
    auto rs = sqrt(max(lsquare<T>(p_tot), 0.));
    if (p1[0] <= 0. || rs <= 0.) {
        // unphysical kinematics, as fks_Sij.f
        s = 0.;
        return;
    }
    auto half = 0.5 * rs;
    // i: the direction from its momentum per unit xi, its energy from its
    // momentum (0 at a soft counter-event)
    auto i_hat = load_mom<T>(p_i_hat);
    auto e_i = fks_frame_energy<T>(load_mom<T>(p[ii]), p_tot, rs);
    auto e_hat = fks_frame_energy<T>(i_hat, p_tot, rs);

    FVal<T> inverse = 0., hfact = 1.;
    std::size_t n = p.size();
    for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t l = 0; l < n; ++l) {
            if (single_index(pair[k][l]) == 0) {
                continue;
            }
            bool zero_num = false, zero_den = false;
            auto qk = load_mom<T>(p[k]), ql = load_mom<T>(p[l]);
            auto qj = load_mom<T>(p[jj]);
            bool ok = single_index(oct[k]) != 0, ol = single_index(oct[l]) != 0;
            bool oi = single_index(oct[ii]) != 0, oj = single_index(oct[jj]) != 0;
            FVal<T> mk = mass[k], ml = mass[l], mi = mass[ii], mj = mass[jj];
            if ((k == ii && l == jj) || (l == ii && k == jj)) {
                inverse = inverse + 1.;
                if (single_index(ag[k]) != 0 && single_index(ag[l]) != 0 && jj >= 2) {
                    auto e_j = fks_frame_energy<T>(qj, p_tot, rs);
                    hfact = hfact * fks_h_damp<T>(e_i / (e_i + e_j), h_one, h_a);
                }
            } else if (k == ii || l == ii) {
                // d_ij / d_il: the energy of i cancels
                std::size_t other = k == ii ? l : k;
                auto qo = load_mom<T>(p[other]);
                bool oo = single_index(oct[other]) != 0;
                FVal<T> mo = mass[other];
                auto num = fks_dkl<T>(
                    i_hat,
                    e_hat,
                    mi,
                    oi,
                    qj,
                    fks_frame_energy<T>(qj, p_tot, rs),
                    mj,
                    oj,
                    false,
                    p_tot,
                    half,
                    fks_a,
                    fks_b,
                    use_energy,
                    use_beta,
                    zero_num
                );
                auto den = fks_dkl<T>(
                    i_hat,
                    e_hat,
                    mi,
                    oi,
                    qo,
                    fks_frame_energy<T>(qo, p_tot, rs),
                    mo,
                    oo,
                    false,
                    p_tot,
                    half,
                    fks_a,
                    fks_b,
                    use_energy,
                    use_beta,
                    zero_den
                );
                if (zero_den) {
                    s = 0.;
                    return;
                }
                inverse = inverse + num / den;
            } else {
                auto num = fks_dkl<T>(
                    i_hat,
                    e_i,
                    mi,
                    oi,
                    qj,
                    fks_frame_energy<T>(qj, p_tot, rs),
                    mj,
                    oj,
                    true,
                    p_tot,
                    half,
                    fks_a,
                    fks_b,
                    use_energy,
                    use_beta,
                    zero_num
                );
                auto den = fks_dkl<T>(
                    qk,
                    fks_frame_energy<T>(qk, p_tot, rs),
                    mk,
                    ok,
                    ql,
                    fks_frame_energy<T>(ql, p_tot, rs),
                    ml,
                    ol,
                    true,
                    p_tot,
                    half,
                    fks_a,
                    fks_b,
                    use_energy,
                    use_beta,
                    zero_den
                );
                if (zero_den) {
                    s = 0.;
                    return;
                }
                inverse = inverse + num / den;
            }
        }
    }
    s = hfact / inverse;
}

// Reduced eikonal factors of eikonal_reduced (fks_singular.f) for the pairs
// (m, n) of the configuration: (p_m p_n) (1 - y) / ((p_m p_i_hat) (p_n
// p_i_hat)), with (p_j p_i_hat) / (1 - y) = sqrt_s E_j / 2 for a massless j,
// which is finite at y = 1. Entries past the pairs of the configuration are
// zero.
template <typename T>
KERNELSPEC void kernel_fks_eikonals(
    FIn<T, 2> p,
    FIn<T, 1> p_i_hat,
    FIn<T, 0> y,
    IIn<T, 0> index,
    IIn<T, 1> fks_j,
    IIn<T, 1> pair_count,
    IIn<T, 2> pair_m,
    IIn<T, 2> pair_n,
    FIn<T, 2> masses,
    FOut<T, 1> eik
) {
    std::size_t c = single_index(index);
    std::size_t jj = single_index(fks_j[c]);
    std::size_t count = single_index(pair_count[c]);
    FVal<T> mj = masses[c][jj];
    bool massless_j = mj == 0.;
    auto p1 = load_mom<T>(p[0]), p2 = load_mom<T>(p[1]);
    FourMom<T> p_tot{p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2], p1[3] + p2[3]};
    auto i_hat = load_mom<T>(p_i_hat);
    FVal<T> one_minus_y = 1. - FVal<T>(y);
    for (std::size_t k = 0; k < eik.size(); ++k) {
        if (k >= count) {
            eik[k] = 0.;
            continue;
        }
        std::size_t m = single_index(pair_m[c][k]), n = single_index(pair_n[c][k]);
        auto pm = load_mom<T>(p[m]), pn = load_mom<T>(p[n]);
        auto dot_nm = ldot<T>(pn, pm);
        FVal<T> dot_mi, dot_ni, fact;
        if ((m != jj && n != jj) || !massless_j) {
            dot_mi = ldot<T>(pm, i_hat);
            dot_ni = ldot<T>(pn, i_hat);
            fact = one_minus_y;
        } else if (m == jj) {
            dot_ni = ldot<T>(pn, i_hat);
            dot_mi = 0.5 * ldot<T>(pm, p_tot);
            fact = 1.;
        } else {
            dot_ni = 0.5 * ldot<T>(pn, p_tot);
            dot_mi = ldot<T>(pm, i_hat);
            fact = 1.;
        }
        eik[k] = dot_nm / (dot_ni * dot_mi) * fact;
    }
}

// The collinear factors of sborncol_fsr and sborncol_isr (fks_singular.f):
// z, the Altarelli-Parisi kernel (1 - z) P(z) / t of AP_reduced and the
// azimuthal terms of Qterms_reduced_timelike (final-state j) or
// Qterms_reduced_spacelike (initial-state j), both without the couplings,
// with t = z s / 4. z = 1 - E_i / (E_i + E_j) in the partonic centre-of-mass
// frame of the momenta for a final-state j, 1 - xi = 1 - 2 E_i / sqrt(s) for
// an initial-state one.
// kind selects the splitting function of AP_reduced (0: g -> g g, 1: g/a ->
// q q~, 2: q -> q g/a, 3: q -> g/a q) and factors holds, per configuration,
// the QCD and QED factors of the kernel and of the azimuthal terms.
template <typename T>
KERNELSPEC void kernel_fks_splitting_kernels(
    FIn<T, 2> p,
    IIn<T, 0> index,
    IIn<T, 1> fks_i,
    IIn<T, 1> fks_j,
    IIn<T, 1> kind,
    FIn<T, 2> factors,
    FOut<T, 0> z,
    FOut<T, 1> ap,
    FOut<T, 1> q
) {
    std::size_t c = single_index(index);
    std::size_t ii = single_index(fks_i[c]), jj = single_index(fks_j[c]);
    auto p1 = load_mom<T>(p[0]), p2 = load_mom<T>(p[1]);
    FourMom<T> p_tot{p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2], p1[3] + p2[3]};
    auto s = lsquare<T>(p_tot);
    // energies in the centre-of-mass frame, up to the common 1 / sqrt(s)
    auto e_i = ldot<T>(load_mom<T>(p[ii]), p_tot);
    bool final_state = jj >= 2;
    FVal<T> zz;
    if (final_state) {
        auto e_j = ldot<T>(load_mom<T>(p[jj]), p_tot);
        zz = 1. - e_i / (e_i + e_j);
    } else {
        zz = 1. - 2. * e_i / s;
    }
    auto t = 0.25 * zz * s;
    auto w = 1. - zz;
    FVal<T> f_ap;
    switch (single_index(kind[c])) {
    case 0:
        f_ap = w * w / zz + zz + zz * w * w;
        break;
    case 1:
        f_ap = (zz * zz + w * w) * w;
        break;
    case 2:
        f_ap = 1. + zz * zz;
        break;
    default:
        f_ap = (1. + w * w) * w / zz;
        break;
    }
    FVal<T> f_q = final_state ? zz * w * w : w * w / zz;
    auto factor = factors[c];
    ap[0] = FVal<T>(factor[0]) * f_ap / t;
    ap[1] = FVal<T>(factor[1]) * f_ap / t;
    q[0] = FVal<T>(factor[2]) * f_q / t;
    q[1] = FVal<T>(factor[3]) * f_q / t;
    z = zz;
}

// The transverse direction of the emission k_perp (xij_kperp of genps_fks.f):
// the momentum of i per unit xi made orthogonal to the Born emitter and to the
// sum of the incoming Born momenta, normalized to k_perp^2 = -1. In the
// partonic centre-of-mass frame it is (0, cos phi, sin phi, 0) in the frame of
// the emitter (the beam axis for an initial-state one), up to its sign.
template <typename T>
KERNELSPEC void kernel_fks_transverse_vector(
    FIn<T, 1> p_i_hat,
    FIn<T, 2> p_born,
    IIn<T, 0> index,
    IIn<T, 1> born_j,
    FOut<T, 1> k_perp
) {
    std::size_t c = single_index(index);
    auto q = load_mom<T>(p_born[single_index(born_j[c])]);
    auto p1 = load_mom<T>(p_born[0]), p2 = load_mom<T>(p_born[1]);
    FourMom<T> P{p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2], p1[3] + p2[3]};
    auto v = load_mom<T>(p_i_hat);
    // v - a q - b P orthogonal to q and P
    auto qq = ldot<T>(q, q), qP = ldot<T>(q, P), PP = ldot<T>(P, P);
    auto vq = ldot<T>(v, q), vP = ldot<T>(v, P);
    auto det = qq * PP - qP * qP;
    auto a = (vq * PP - vP * qP) / det;
    auto b = (vP * qq - vq * qP) / det;
    FourMom<T> k;
    for (std::size_t mu = 0; mu < 4; ++mu) {
        k[mu] = v[mu] - a * q[mu] - b * P[mu];
    }
    auto norm = sqrt(max(-lsquare<T>(k), 0.));
    for (std::size_t mu = 0; mu < 4; ++mu) {
        k_perp[mu] = norm > 0. ? FVal<T>(k[mu] / norm) : FVal<T>(0.);
    }
}

} // namespace kernels
} // namespace madspace

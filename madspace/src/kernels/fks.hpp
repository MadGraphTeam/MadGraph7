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
// genps_fks.f: a vector along +z (or of zero length) has azimuth zero
template <typename T>
KERNELSPEC Quartuplet<FVal<T>, FVal<T>, FVal<T>, FVal<T>> fks_angles(FourMom<T> p) {
    auto len2 = p[1] * p[1] + p[2] * p[2] + p[3] * p[3];
    auto cth = where(len2 == 0., FVal<T>(1.), p[3] / sqrt(max(len2, EPS2)));
    auto pole = cth >= 1.;
    auto sth = where(pole, FVal<T>(0.), sqrt(max(1. - cth * cth, 0.)));
    auto phi = atan2(p[2], p[1]);
    auto cphi = where(pole, FVal<T>(1.), cos(phi));
    auto sphi = where(pole, FVal<T>(0.), sin(phi));
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
    FOut<T, 0> det_xi
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

// Boost along the direction of the FKS pair k that takes the Born recoil to
// the recoil of the real emission, as genps_fks.f with boostwdir2. sign = -1
// gives the inverse.
template <typename T>
KERNELSPEC void
fks_recoil_boost(FIn<T, 2> p, FIn<T, 1> k, FVal<T> rs, double sign, FOut<T, 2> p_out) {
    auto klen = sqrt(max(k[1] * k[1] + k[2] * k[2] + k[3] * k[3], EPS2));
    auto sumrec = rs - k[0] + klen;
    auto den = 2. * sumrec * rs;
    auto sh = -sign * (rs - sumrec) * (rs + sumrec) / den;
    auto ch = (rs * rs + sumrec * sumrec) / den;
    auto chmo = (rs - sumrec) * (rs - sumrec) / den;
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
KERNELSPEC void
kernel_fks_recoil_boost(FIn<T, 2> p, FIn<T, 1> k, FIn<T, 0> sqrt_s, FOut<T, 2> p_out) {
    fks_recoil_boost<T>(p, k, sqrt_s, 1., p_out);
}

template <typename T>
KERNELSPEC void kernel_fks_recoil_boost_inverse(
    FIn<T, 2> p, FIn<T, 1> k, FIn<T, 0> sqrt_s, FOut<T, 2> p_out
) {
    fks_recoil_boost<T>(p, k, sqrt_s, -1., p_out);
}

} // namespace kernels
} // namespace madspace

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
KERNELSPEC FVal<T> fks_initial_det_xi(FVal<T> s_born, FVal<T> xi_max, FVal<T> omx) {
    constexpr double norm = 2. * PI / ((4. * PI) * (4. * PI) * (4. * PI));
    return norm * xi_max * s_born / max(omx * omx, EPS2);
}

template <typename T>
KERNELSPEC void kernel_fks_initial_radiate(
    FIn<T, 0> xi_hat,
    FIn<T, 0> y,
    FIn<T, 0> r_phi,
    FIn<T, 0> sqrt_s,
    FIn<T, 0> x1_born,
    FIn<T, 0> x2_born,
    FSIn<T, 0> beam_sign,
    FOut<T, 1> p_in1,
    FOut<T, 1> p_in2,
    FOut<T, 1> p_i,
    FOut<T, 0> x1,
    FOut<T, 0> x2,
    FOut<T, 0> xi,
    FOut<T, 0> xi_max,
    FOut<T, 0> det_xi
) {
    FVal<T> xh = xi_hat, yv = y, rp = r_phi, rs_born = sqrt_s;
    FVal<T> x1b = x1_born, x2b = x2_born, sign(beam_sign);
    auto yb = sign * yv;
    auto omy_beam = 1. - yb;
    auto opy_beam = 1. + yb;
    auto xm = fks_initial_xi_max<T>(x1b, x2b, omy_beam, opy_beam);
    auto xiv = xh * xm;
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
    x1 = x1b / (sq_omx * omega);
    x2 = x2b * omega / sq_omx;
    xi = xiv;
    xi_max = xm;
    det_xi = fks_initial_det_xi<T>(rs_born * rs_born, xm, omx);
}

template <typename T>
KERNELSPEC void kernel_fks_initial_cluster(
    FIn<T, 1> p_in1,
    FIn<T, 1> p_in2,
    FIn<T, 1> p_i,
    FIn<T, 0> x1,
    FIn<T, 0> x2,
    FSIn<T, 0> beam_sign,
    FOut<T, 0> sqrt_s,
    FOut<T, 0> x1_born,
    FOut<T, 0> x2_born,
    FOut<T, 0> xi_hat,
    FOut<T, 0> y,
    FOut<T, 0> r_phi,
    FOut<T, 0> xi,
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
    auto yb = (pi_dot_2 - pi_dot_1) / pi_dot_p;

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

    auto rphi = atan2(pi[2], pi[1]) / (2. * PI);
    sqrt_s = sqrt(max(s_born, 0.));
    x1_born = x1b;
    x2_born = x2b;
    xi_hat = xiv / max(xm, EPS);
    y = sign * yb;
    r_phi = where(rphi < 0., rphi + 1., rphi);
    xi = xiv;
    xi_max = xm;
    det_xi = 1. / max(fks_initial_det_xi<T>(s_born, xm, omx), EPS2);
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
    FSIn<T, 0> beam_sign,
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
    FSIn<T, 0> beam_sign,
    FOut<T, 2> p_out
) {
    fks_initial_recoil_boost<T>(p, xi, y, r_phi, FVal<T>(beam_sign), -1., p_out);
}

} // namespace kernels
} // namespace madspace

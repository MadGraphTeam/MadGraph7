#pragma once

#include "kinematics.hpp"

namespace madspace {
namespace kernels {

// Helper functions

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

    auto det = det_omega * det_E2 * det_E1 / 8;
    return {p1, p2, det};
}

template <typename T>
KERNELSPEC Decuplet<
    FVal<T>,
    FVal<T>,
    FVal<T>,
    FVal<T>,
    FVal<T>,
    FVal<T>,
    FVal<T>,
    FVal<T>,
    FVal<T>,
    FVal<T>>
three_body_decay_inverse(FourMom<T> p1, FourMom<T> p2, FourMom<T> p3) {
    // this is based on section G.3 in
    // https://inspirehep.net/literature/1784296

    // Define total momentum
    FourMom<T> p0;
    for (int i = 0; i < 4; ++i) {
        p0[i] = p1[i] + p2[i] + p3[i];
    }
    auto m0 = sqrt(max(EPS2, lsquare<T>(p0)));
    auto m1 = sqrt(max(EPS2, lsquare<T>(p1)));
    auto m2 = sqrt(max(EPS2, lsquare<T>(p2)));
    auto m3 = sqrt(max(EPS2, lsquare<T>(p3)));

    // Define mass squares
    auto m1sq = m1 * m1;
    auto m2sq = m2 * m2;
    auto m3sq = m3 * m3;

    // define energy E1
    auto E1_max = m0 / 2 + (m1sq - (m2 + m3) * (m2 + m3)) / (2 * m0);
    auto E1 = p1[0];
    auto r_e1 = (p1[0] - m1) / (E1_max - m1);
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
    auto E2 = p2[0];
    auto r_e2 = (p2[0] - E2_min) / (E2_max - E2_min);
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

    // calculate angles and determinants
    auto phi = atan2(p1[2], p1[1]);
    auto cos_theta = p1[3] / pp1;
    auto sin_theta = sqrt((1. - cos_theta) * (1 + cos_theta));

    auto r_phi = (phi / PI + 1.) / 2.;
    auto r_cos_theta = (cos_theta + 1.) / 2.;

    // calculate beta angle
    auto A = (cos_alpha * cos_theta - p2[3] / pp2) / sin_theta;
    auto B = -p2[1] / pp2 * sin(phi) + p2[2] / pp2 * cos(phi);
    auto beta = atan2(B, A);

    auto r_beta = (beta / PI + 1.) / 2.;
    auto det_omega = 8 * PI * PI;

    auto det = det_omega * det_E1 * det_E2 / 8.;
    return {r_e1, r_e2, r_phi, r_cos_theta, r_beta, m0, m1, m2, m3, 1. / det};
}

template <typename T>
KERNELSPEC Pair<FVal<T>, FVal<T>> s23_min_max(
    FourMom<T> pa,
    FourMom<T> pb,
    FourMom<T> p3,
    FourMom<T> p_12,
    FVal<T> t1_abs,
    FVal<T> m1,
    FVal<T> m2
) {
    // Range of s23 = (pa + pb - p1)^2 as p1 turns about pa at fixed s12 and t1.
    // In the p_12 rest frame, with z along pa and p3 in the phi = 0 half-plane
    // (the frame of rotate_two_ref),
    //   s23 = m0^2 + m1^2 - 2 [(sqrt(s12) + E3) E1 - p3_z p1_z]
    //         + 2 p3_x p1_t cos(phi),
    // a range of width 4 p3_x p1_t about the phi-independent part. Every factor
    // of the width is read off momenta, so it keeps its relative precision when
    // a particle is soft. The Byckling-Kajantie form of the same range, through
    // 3x3 Gram determinants, cancels down to the square of the width and loses
    // it all below a softness of about 1e-8.
    FourMom<T> p_tot;
    for (int i = 0; i < 4; ++i) {
        p_tot[i] = pa[i] + pb[i];
    }
    auto m0_2 = lsquare<T>(p_tot);
    auto s12 = lsquare<T>(p_12);
    auto ma_2 = lsquare<T>(pa);
    auto pa_com = boost<T>(pa, p_12, -1.);
    auto p3_com = boost<T>(p3, p_12, -1.);

    // p1 at phi = 0: energy, momentum along pa (p1_z) and across it (p1_t >= 0)
    auto p1_out = p1com_from_tabs_phi<T>(
        pa_com, s12, FVal<T>(0.), t1_abs, m1, m2, ma_2, FVal<T>(0.)
    );
    auto p1_com = p1_out.first;

    // p3 along pa (p3_z) and across it (p3_x >= 0)
    auto pa_mag = sqrt(max(esquare<T>(pa_com), EPS2));
    auto p3_z =
        (p3_com[1] * pa_com[1] + p3_com[2] * pa_com[2] + p3_com[3] * pa_com[3]) /
        pa_mag;
    auto cross_x = p3_com[2] * pa_com[3] - p3_com[3] * pa_com[2];
    auto cross_y = p3_com[3] * pa_com[1] - p3_com[1] * pa_com[3];
    auto cross_z = p3_com[1] * pa_com[2] - p3_com[2] * pa_com[1];
    auto p3_x =
        sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z) / pa_mag;

    auto center = m0_2 + m1 * m1 -
        2. * ((sqrt(max(s12, 0.)) + p3_com[0]) * p1_com[0] - p3_z * p1_com[3]);
    auto half_width = 2. * p3_x * p1_com[1];
    return {center - half_width, center + half_width};
}

// Kernels

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
KERNELSPEC void kernel_three_body_decay_com_inverse(
    FIn<T, 1> p1,
    FIn<T, 1> p2,
    FIn<T, 1> p3,
    FOut<T, 0> r_e1,
    FOut<T, 0> r_e2,
    FOut<T, 0> r_phi,
    FOut<T, 0> r_cos_theta,
    FOut<T, 0> r_beta,
    FOut<T, 0> m0,
    FOut<T, 0> m1,
    FOut<T, 0> m2,
    FOut<T, 0> m3,
    FOut<T, 0> det
) {
    auto decay_out =
        three_body_decay_inverse<T>(load_mom<T>(p1), load_mom<T>(p2), load_mom<T>(p3));
    r_e1 = decay_out.first;
    r_e2 = decay_out.second;
    r_phi = decay_out.third;
    r_cos_theta = decay_out.fourth;
    r_beta = decay_out.fifth;
    m0 = decay_out.sixth;
    m1 = decay_out.seventh;
    m2 = decay_out.eighth;
    m3 = decay_out.ninth;
    det = decay_out.tenth;
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
KERNELSPEC void kernel_three_body_decay_inverse(
    FIn<T, 1> p1,
    FIn<T, 1> p2,
    FIn<T, 1> p3,
    FOut<T, 0> r_e1,
    FOut<T, 0> r_e2,
    FOut<T, 0> r_phi,
    FOut<T, 0> r_cos_theta,
    FOut<T, 0> r_beta,
    FOut<T, 0> m0,
    FOut<T, 0> m1,
    FOut<T, 0> m2,
    FOut<T, 0> m3,
    FOut<T, 1> p0,
    FOut<T, 0> det
) {
    // Define total momentum
    FourMom<T> ptot;
    for (int i = 0; i < 4; ++i) {
        ptot[i] = p1[i] + p2[i] + p3[i];
    }
    store_mom<T>(p0, ptot);
    auto p1_com = boost<T>(load_mom<T>(p1), ptot, -1.);
    auto p2_com = boost<T>(load_mom<T>(p2), ptot, -1.);
    auto p3_com = boost<T>(load_mom<T>(p3), ptot, -1.);
    auto decay_out = three_body_decay_inverse<T>(p1_com, p2_com, p3_com);
    r_e1 = decay_out.first;
    r_e2 = decay_out.second;
    r_phi = decay_out.third;
    r_cos_theta = decay_out.fourth;
    r_beta = decay_out.fifth;
    m0 = decay_out.sixth;
    m1 = decay_out.seventh;
    m2 = decay_out.eighth;
    m3 = decay_out.ninth;
    det = decay_out.tenth;
}

// Kernels for 2->3 scattering

template <typename T>
KERNELSPEC void kernel_s23_min_max(
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 1> p3,
    FIn<T, 0> t1_abs,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FOut<T, 0> s23_min,
    FOut<T, 0> s23_max
) {
    // this function is based on the sminmax subroutine from Rikkert
    // expects t1_abs (positive t invariant) as input
    FourMom<T> p_12;
    for (int i = 0; i < 4; ++i) {
        p_12[i] = pa[i] + pb[i] - p3[i];
    }
    auto s23_out = s23_min_max<T>(
        load_mom<T>(pa), load_mom<T>(pb), load_mom<T>(p3), p_12, t1_abs, m1, m2
    );
    s23_min = s23_out.first;
    s23_max = s23_out.second;
}

template <typename T>
KERNELSPEC void kernel_s23_value_and_min_max(
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 1> p3,
    FIn<T, 0> t1_abs,
    FIn<T, 1> p1,
    FIn<T, 1> p2,
    FOut<T, 0> s_23,
    FOut<T, 0> s23_min,
    FOut<T, 0> s23_max
) {
    // this function is based on the sminmax subroutine from Rikkert
    // expects t1_abs (positive t invariant) as input
    FourMom<T> p_12, p_23;
    for (int i = 0; i < 4; ++i) {
        p_12[i] = p1[i] + p2[i];
        p_23[i] = p2[i] + p3[i];
    }
    auto m1 = sqrt(max(lsquare<T>(load_mom<T>(p1)), 0.));
    auto m2 = sqrt(max(lsquare<T>(load_mom<T>(p2)), 0.));

    auto s23_out = s23_min_max<T>(
        load_mom<T>(pa), load_mom<T>(pb), load_mom<T>(p3), p_12, t1_abs, m1, m2
    );
    s23_min = s23_out.first;
    s23_max = s23_out.second;
    s_23 = lsquare<T>(p_23);
}

// AmpliCol block-B s-channel ETmin refinement (gen23 lines 745-762). Tightens
// the sampled s-channel mass s23 = (peeled + im1)^2 using the recoil's ETmin
// floor (upper bound, smax) and the peeled particle's ETmin + dR cut (lower
// bound, smin -- only for a massless peeled particle).
template <typename T>
KERNELSPEC Pair<FVal<T>, FVal<T>> s23_etmin_clamp(
    FVal<T> smn,
    FVal<T> smx,
    FourMom<T> piir,
    FourMom<T> pim1,
    FourMom<T> pib,
    FVal<T> m1_2,
    FVal<T> m2_2,
    FVal<T> m3_2,
    FVal<T> t1_abs,
    FVal<T> etmin_1,
    FVal<T> etmin_2,
    FVal<T> drcut
) {
    auto active = (etmin_2 > EPS) | (etmin_1 > EPS);
    // Block B works in the frame where pz(im1)=0 (z-boost by the IM1 rapidity),
    // with im1 rotated onto +x. mT(im1)=sqrt(E^2-pz^2)=sqrt(m3^2+pt(p3)^2).
    auto Eim1 = pim1[0];
    auto pzim1 = pim1[3];
    auto mT_im1 = sqrt(max(Eim1 * Eim1 - pzim1 * pzim1, EPS));
    // Boost p_12 (=piir, the i+ir system) and pa (=pib) into that frame; im1's
    // own energy there is mT(im1) and its x-momentum is pt(p3).
    auto piir_0 = (piir[0] * Eim1 - piir[3] * pzim1) / mT_im1;
    auto pim1_0 = mT_im1;
    auto pim1_1 = sqrt(pim1[1] * pim1[1] + pim1[2] * pim1[2]);
    auto pib_0 = max((pib[0] * Eim1 - pib[3] * pzim1) / mT_im1, EPS);
    // Block-B recoil ET floor (t-dependent): invm(ir)-invm(ir+ib) = m1_2 + t1_abs.
    auto denom = max(m1_2 + t1_abs, EPS);
    auto etminir_b =
        max(etmin_1, pib_0 * etmin_1 * etmin_1 / denom + denom / (4. * pib_0));
    // smax: the recoil's ETmin caps the s-channel mass from above.
    auto e_eff = piir_0 - etminir_b;
    auto disc = e_eff * e_eff - m2_2;
    auto smax_b = m2_2 + m3_2 + 2. * e_eff * pim1_0 + 2. * sqrt(max(disc, 0.)) * pim1_1;
    auto smax_ok = active & (e_eff > 0.) & (disc > 0.) & (smax_b > smn);
    auto smx_new = where(smax_ok, min(smx, smax_b), smx);
    // smin: a massless peeled particle needs ET -> minimum s-channel mass.
    // Massless is judged relative to the energy scale: the inverse reads m2_2
    // off momenta, where a massless particle comes out at +-1e-16 E^2 or so,
    // and an absolute test (m2_2 < EPS) turned this bound off there and
    // changed the sampled range between forward and inverse.
    auto smin_b = 2. * etmin_2 * (pim1_0 - pim1_1 * cos(drcut));
    auto massless_2 = m2_2 < 1e-10 * piir_0 * piir_0;
    auto smin_ok = active & massless_2 & (smin_b > smn) & (smin_b < smx_new);
    auto smn_new = where(smin_ok, max(smn, smin_b), smn);
    smx_new = where(smx_new > smn_new, smx_new, smn_new + EPS);
    return {smn_new, smx_new};
}

// Clamp the s23 invariant-mass range against the cut-derived lower bound.
// s23_min_cut is the minimum invariant mass^2 of the (2,3) system implied by
// the pt / m_inv_min / dR cuts (gen23's invm_min).
template <typename T>
KERNELSPEC void kernel_s23_min_max_cut(
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 1> p3,
    FIn<T, 0> t1_abs,
    FIn<T, 0> m1,
    FIn<T, 0> m2,
    FIn<T, 0> etmin_1,
    FIn<T, 0> etmin_2,
    FIn<T, 0> drcut,
    FIn<T, 0> s23_min_cut,
    FOut<T, 0> s23_min,
    FOut<T, 0> s23_max
) {
    FourMom<T> p_12;
    for (int i = 0; i < 4; ++i) {
        p_12[i] = pa[i] + pb[i] - p3[i];
    }
    auto m3_2 = lsquare<T>(load_mom<T>(p3));
    auto m1_2 = m1 * m1;
    auto m2_2 = m2 * m2;

    auto s23_out = s23_min_max<T>(
        load_mom<T>(pa), load_mom<T>(pb), load_mom<T>(p3), p_12, t1_abs, m1, m2
    );
    auto smn = s23_out.first;
    auto smx = s23_out.second;

    FVal<T> smin_cut(s23_min_cut);
    smn = where(smin_cut > 0., max(smn, smin_cut), smn);
    smx = where(smx > smn, smx, smn + EPS);

    // Block-B ETmin refinement: piir = p_12 (recoil system), pim1 = p3, pib = pa.
    auto sb = s23_etmin_clamp<T>(
        smn,
        smx,
        p_12,
        load_mom<T>(p3),
        load_mom<T>(pa),
        m1_2,
        m2_2,
        m3_2,
        t1_abs,
        etmin_1,
        etmin_2,
        drcut
    );

    s23_min = sb.first;
    s23_max = sb.second;
}

template <typename T>
KERNELSPEC void kernel_s23_value_and_min_max_cut(
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 1> p3,
    FIn<T, 0> t1_abs,
    FIn<T, 1> p1,
    FIn<T, 1> p2,
    FIn<T, 0> etmin_1,
    FIn<T, 0> etmin_2,
    FIn<T, 0> drcut,
    FIn<T, 0> s23_min_cut,
    FOut<T, 0> s_23,
    FOut<T, 0> s23_min,
    FOut<T, 0> s23_max
) {
    FourMom<T> p_12, p_23;
    for (int i = 0; i < 4; ++i) {
        p_12[i] = p1[i] + p2[i];
        p_23[i] = p2[i] + p3[i];
    }
    auto m3_2 = lsquare<T>(load_mom<T>(p3));
    auto m1_2 = lsquare<T>(load_mom<T>(p1));
    auto m2_2 = lsquare<T>(load_mom<T>(p2));

    auto s23_out = s23_min_max<T>(
        load_mom<T>(pa),
        load_mom<T>(pb),
        load_mom<T>(p3),
        p_12,
        t1_abs,
        sqrt(max(m1_2, 0.)),
        sqrt(max(m2_2, 0.))
    );
    auto smn = s23_out.first;
    auto smx = s23_out.second;

    FVal<T> smin_cut(s23_min_cut);
    smn = where(smin_cut > 0., max(smn, smin_cut), smn);
    smx = where(smx > smn, smx, smn + EPS);

    // Block-B ETmin refinement (must mirror the forward kernel exactly so the
    // sampled s23 range is identical and the round-trip stays invertible).
    auto sb = s23_etmin_clamp<T>(
        smn,
        smx,
        p_12,
        load_mom<T>(p3),
        load_mom<T>(pa),
        m1_2,
        m2_2,
        m3_2,
        t1_abs,
        etmin_1,
        etmin_2,
        drcut
    );

    s23_min = sb.first;
    s23_max = sb.second;
    s_23 = lsquare<T>(p_23);
}

// Arcsine map of the s23 sampling variable.
//
// At fixed s12, t1 and t2 the invariant s23 is linear in cos(phi) on its
// physical range [s_phys_min, s_phys_max], and the 2->3 measure is flat in phi:
//   ds23 / (8 sqrt(-G4)) = dphi / (2 sqrt(lambda)).
// Sampling s23 (or any smooth function of it) with a density that stays finite
// at the edges of the physical range therefore leaves an integrable
// 1/|sin(phi)| in the weight, with a log-divergent variance. With
//   u = (s23 - s_phys_min) / (s_phys_max - s_phys_min) = sin^2(phi / 2),
// the map below is flat in theta = phi / 2 on the part [theta_a, theta_b] of
// [0, pi/2] that the sampling range [s_min, s_max] covers, and returns the
// position x in [0, 1] of the point in that range:
//   x = (sin^2 theta - sin^2 theta_a) / (sin^2 theta_b - sin^2 theta_a),
//   theta = theta_a + (theta_b - theta_a) r.
// x is then handed to the s23 importance sampling in place of r. Its Jacobian
// dx/dr vanishes like |sin(phi)| at a physical edge, so the product with the
// 2->3 Jacobian stays bounded; at an edge set by a cut instead (theta_a > 0 or
// theta_b < pi/2) it stays finite and nothing is cancelled. Differences of
// sin^2 are written as sin(B - A) sin(B + A), so that x keeps its relative
// precision where it is small, also when the sampled range is a narrow part of
// the kinematic one.
template <typename T>
KERNELSPEC Pair<FVal<T>, FVal<T>> s23_arcsine_angles(
    FVal<T> s_min, FVal<T> s_max, FVal<T> s_phys_min, FVal<T> s_phys_max
) {
    auto width = s_phys_max - s_phys_min;
    auto width_safe = where(width > 0., width, FVal<T>(1.));
    // u and 1 - u of both ends, each from its own difference
    auto ua = min(max((s_min - s_phys_min) / width_safe, 0.), 1.);
    auto ua_c = min(max((s_phys_max - s_min) / width_safe, 0.), 1.);
    auto ub = min(max((s_max - s_phys_min) / width_safe, 0.), 1.);
    auto ub_c = min(max((s_phys_max - s_max) / width_safe, 0.), 1.);
    auto theta_a = atan2(sqrt(ua), sqrt(ua_c));
    auto theta_b = atan2(sqrt(ub), sqrt(ub_c));
    // an empty physical range leaves nothing to map: theta_b = theta_a
    return {theta_a, where(width > 0., theta_b, theta_a)};
}

template <typename T>
KERNELSPEC void kernel_s23_arcsine(
    FIn<T, 0> r,
    FIn<T, 0> s_min,
    FIn<T, 0> s_max,
    FIn<T, 0> s_phys_min,
    FIn<T, 0> s_phys_max,
    FOut<T, 0> x,
    FOut<T, 0> det
) {
    auto angles = s23_arcsine_angles<T>(s_min, s_max, s_phys_min, s_phys_max);
    auto theta_a = angles.first;
    auto dtheta = angles.second - angles.first;
    // sin^2 theta_b - sin^2 theta_a
    auto den = sin(dtheta) * sin(angles.first + angles.second);
    auto ok = (dtheta > 0.) & (den > 0.);
    auto den_safe = where(ok, den, FVal<T>(1.));

    FVal<T> r_val(r);
    auto dtheta_r = dtheta * r_val;
    // x = (sin^2 theta - sin^2 theta_a) / den, dx/dr = dtheta sin(2 theta) / den
    auto x_val = sin(dtheta_r) * sin(2. * theta_a + dtheta_r) / den_safe;
    auto det_val = dtheta * sin(2. * (theta_a + dtheta_r)) / den_safe;
    x = where(ok, min(max(x_val, 0.), 1.), r_val);
    det = where(ok, det_val, FVal<T>(1.));
}

template <typename T>
KERNELSPEC void kernel_s23_arcsine_inverse(
    FIn<T, 0> x,
    FIn<T, 0> s_min,
    FIn<T, 0> s_max,
    FIn<T, 0> s_phys_min,
    FIn<T, 0> s_phys_max,
    FOut<T, 0> r,
    FOut<T, 0> det
) {
    auto angles = s23_arcsine_angles<T>(s_min, s_max, s_phys_min, s_phys_max);
    auto theta_a = angles.first;
    auto theta_b = angles.second;
    auto dtheta = theta_b - theta_a;
    auto den = sin(dtheta) * sin(theta_a + theta_b);
    auto ok = (dtheta > 0.) & (den > 0.);
    auto den_safe = where(ok, den, FVal<T>(1.));
    auto dtheta_safe = where(ok, dtheta, FVal<T>(1.));

    FVal<T> x_val(x);
    // sin^2 theta = sin^2 theta_a + x den, cos^2 theta = cos^2 theta_b + (1 - x) den
    auto sin_a = sin(theta_a);
    auto cos_b = cos(theta_b);
    auto sin2 = sin_a * sin_a + x_val * den_safe;
    auto cos2 = cos_b * cos_b + (1. - x_val) * den_safe;
    auto theta = atan2(sqrt(max(sin2, 0.)), sqrt(max(cos2, 0.)));
    auto r_val = (theta - theta_a) / dtheta_safe;
    // dr/dx; zero at the edges of the physical range, where dx/dr vanishes
    auto sin_2theta = sin(2. * theta);
    auto det_val =
        where(sin_2theta > 0., den_safe / (dtheta_safe * sin_2theta), FVal<T>(0.));
    r = where(ok, min(max(r_val, 0.), 1.), x_val);
    det = where(ok, det_val, FVal<T>(1.));
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
    FourMom<T> p_12, p_c;
    for (int i = 0; i < 4; ++i) {
        p_12[i] = pa[i] + pb[i] - p3[i];
        p_c[i] = pb[i] - p3[i];
    }
    auto pa_com = boost<T>(load_mom<T>(pa), p_12, -1.);
    auto ma_2 = lsquare<T>(load_mom<T>(pa));
    auto s12 = lsquare<T>(p_12);
    auto t2 = lsquare<T>(p_c);

    // s23 is linear in cos(phi) across its kinematic range, and the 4x4 Gram
    // determinant of Byckling-Kajantie factorises on that range:
    //   cos(phi) = 2 u - 1,  u = (s23 - s23_min) / (s23_max - s23_min),
    //   -G4 = lambda(s12, ma^2, t2) / 16 * (s23 - s23_min) * (s23_max - s23).
    // Both are evaluated through u here. Forming G4 (or the equivalent cos(phi)
    // expression) from the invariants instead cancels terms of order s^4 down
    // to a result that can be many orders of magnitude smaller whenever the
    // s23 range is narrow (a soft particle), and returned Jacobians of up to
    // 1e20 times the typical weight. The range itself comes from momenta for
    // the same reason, see s23_min_max.
    auto s23_range = s23_min_max<T>(
        load_mom<T>(pa), load_mom<T>(pb), load_mom<T>(p3), p_12, t1_abs, m1, m2
    );
    auto s23_width = s23_range.second - s23_range.first;
    auto u = min(max((s23 - s23_range.first) / s23_width, 0.), 1.);
    auto cos_phi = 2. * u - 1.;
    auto phi = where(phi_index == 1, -acos(cos_phi), acos(cos_phi));
    // 8 sqrt(-G4) = sqrt(lambda) * (s23_max - s23_min) * |sin(phi)|
    auto sqrt_neg_gram4_x8 =
        sqrt(max(kaellen<T>(s12, ma_2, t2), 0.)) * s23_width * 2. * sqrt(u * (1. - u));
    // The edges of the range (sin(phi) = 0) are a set of measure zero.
    auto det_2to3 = where(sqrt_neg_gram4_x8 > 0., 1. / sqrt_neg_gram4_x8, FVal<T>(0.));

    auto scatter_out =
        p1com_from_tabs_phi<T>(pa_com, s12, phi, t1_abs, m1, m2, ma_2, t2);
    auto p1_com = scatter_out.first;
    auto p3_p12 = boost<T>(load_mom<T>(p3), p_12, -1.);
    auto p1_rot = rotate_two_ref<T>(p1_com, pa_com, p3_p12);
    auto p1_lab = boost<T>(p1_rot, p_12, 1.);
    store_mom<T>(p1, p1_lab);
    for (int i = 0; i < 4; ++i) {
        p2[i] = p_12[i] - p1_lab[i];
    }
    det = det_2to3 / 2; // factor 1/2 as acos allows for two choices of phi
}

template <typename T>
KERNELSPEC void kernel_two_to_three_particle_scattering_inverse(
    FIn<T, 1> p1,
    FIn<T, 1> p2,
    FIn<T, 1> p3,
    FIn<T, 1> pa,
    FIn<T, 1> pb,
    FIn<T, 0> t1_abs,
    FIn<T, 0> s23,
    FOut<T, 0> m1,
    FOut<T, 0> m2,
    IOut<T, 0> phi_index,
    FOut<T, 0> det
) {
    FourMom<T> p_12, p_c;
    for (int i = 0; i < 4; ++i) {
        p_12[i] = pa[i] + pb[i] - p3[i];
        p_c[i] = pb[i] - p3[i];
    }
    auto ma_2 = lsquare<T>(load_mom<T>(pa));
    auto s12 = lsquare<T>(p_12);
    auto t2 = lsquare<T>(p_c);

    auto pa_com = boost<T>(load_mom<T>(pa), p_12, -1.);
    auto p1_com = boost<T>(load_mom<T>(p1), p_12, -1.);
    auto p3_p12 = boost<T>(load_mom<T>(p3), p_12, -1.);
    auto p1_rot = rotate_two_ref_inverse<T>(p1_com, pa_com, p3_p12);

    auto m1_2 = lsquare<T>(load_mom<T>(p1));
    auto m2_2 = lsquare<T>(load_mom<T>(p2));
    auto phi = atan2(p1_rot[2], p1_rot[1]);
    phi_index = where(
        phi < 0, IVal<T>(1), IVal<T>(0)
    ); // choose phi index based on the value of phi

    // 8 sqrt(-G4) = sqrt(lambda) * (s23_max - s23_min) * |sin(phi)|, see the
    // forward kernel; |sin(phi)| is read off the momenta directly.
    auto s23_range = s23_min_max<T>(
        load_mom<T>(pa),
        load_mom<T>(pb),
        load_mom<T>(p3),
        p_12,
        t1_abs,
        sqrt(max(m1_2, 0.)),
        sqrt(max(m2_2, 0.))
    );
    auto pt2_rot = p1_rot[1] * p1_rot[1] + p1_rot[2] * p1_rot[2];
    auto sin_phi_abs =
        where(pt2_rot > 0., fabs(p1_rot[2]) / sqrt(pt2_rot), FVal<T>(0.));
    auto det_2to3 = sqrt(max(kaellen<T>(s12, ma_2, t2), 0.)) *
        (s23_range.second - s23_range.first) * sin_phi_abs;
    m1 = sqrt(max(EPS2, m1_2));
    m2 = sqrt(max(EPS2, m2_2));
    det = 2 * det_2to3;
}

} // namespace kernels
} // namespace madspace

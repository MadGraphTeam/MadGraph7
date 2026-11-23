#pragma once

#include "definitions.hpp"
#include "kinematics.hpp"
#include "twoparticle.hpp"

namespace madspace {
namespace kernels {

// The largest rapidity of an outgoing particle of mass m and transverse mass^2
// mt2 with fixed beams of total energy e_cm, acosh(E_max / m_T), from its
// largest energy E_max = (s + m^2 - (mass_sum - m)^2) / (2 sqrt(s)) when all
// the others together have the least mass, mass_sum - m. 0 where m_T > E_max.
template <typename T>
KERNELSPEC FVal<T>
chili_fixed_y_max(FVal<T> e_cm, FVal<T> m, FVal<T> mass_sum, FVal<T> mt2) {
    auto rest = mass_sum - m;
    auto e_max = 0.5 * (e_cm + (m - rest) * (m + rest) / e_cm);
    auto ratio2 = e_max * e_max / max(mt2, FVal<T>(EPS2));
    return log(sqrt(max(ratio2, 1.)) + sqrt(max(ratio2 - 1., 0.)));
}

// The sums of the momenta chili has built so far, and the Jacobian.
template <typename T>
struct ChiliSums {
    FVal<T> e, px, py, pz, det;
};

// The first `count` outgoing momenta of chili, in p_ext[i + 2], generated in
// transverse momentum, rapidity and azimuth, the azimuth relative to the
// running recoil. The random numbers of particle i are r[pt_at + i],
// r[y_at + i] and r[phi_at + i]. The rapidity of a particle runs up to where
// its energy reaches sqrt(s) / 2, and with fixed beams up to its largest one,
// (s + m_i^2 - (mass_sum - m_i)^2) / (2 sqrt(s)), which is larger for a
// particle heavier than all the others together.
template <typename T, bool fixed = false>
KERNELSPEC ChiliSums<T> chili_head_forward(
    FIn<T, 1> r,
    std::size_t count,
    std::size_t pt_at,
    std::size_t y_at,
    std::size_t phi_at,
    FVal<T> _e_cm,
    FIn<T, 1> m_out,
    FSIn<T, 1> pt_min,
    FSIn<T, 1> y_max,
    FOut<T, 2> p_ext,
    FVal<T> mass_sum = 0.
) {
    FVal<T> e_sum(0.), px_sum(0.), py_sum(0.), pz_sum(0.);
    FVal<T> det_tmp(1.);
    auto e_cm2 = _e_cm * _e_cm;
    auto pt_max = _e_cm / 2; // loose bound; points with x1 or x2 >= 1 get zero weight
    auto pt2_max = pt_max * pt_max;
    for (std::size_t i = 0; i < count; ++i) {
        FVal<T> pt_min_i(pt_min[i]), y_max_i(y_max[i]), m_out_i(m_out[i]);
        auto p_i = p_ext[i + 2];
        FVal<T> r_pt(r[pt_at + i]), r_y(r[y_at + i]), r_phi(r[phi_at + i]);

        // ---- get the pts
        auto pt2_min_i = pt_min_i * pt_min_i;

        // with pt cut
        auto pt2_withcut = 1. / (r_pt / pt2_max + (1. - r_pt) / pt2_min_i);
        auto pt_withcut = sqrt(pt2_withcut);
        auto det_pt_withcut =
            pt2_withcut * pt2_withcut * (1. / pt2_min_i - 1. / pt2_max);

        // without pt cut
        auto mm_pt_i = where(m_out_i > EPS, m_out_i, 1.0);
        auto pt_nocut =
            2. * mm_pt_i * pt_max * r_pt / (2. * mm_pt_i + pt_max * (1. - r_pt));
        auto pt2_nocut = pt_nocut * pt_nocut;
        auto factor_pt = (2 * mm_pt_i + pt_nocut);
        auto denom_pt = mm_pt_i * (2 * mm_pt_i + pt_max);
        auto det_pt_nocut = pt_nocut * pt_max * factor_pt * factor_pt / denom_pt;

        auto has_pt_cut = pt2_min_i > 1e-9;
        auto pt = where(has_pt_cut, pt_withcut, pt_nocut);
        auto pt2 = where(has_pt_cut, pt2_withcut, pt2_nocut);
        auto det_pt = where(has_pt_cut, det_pt_withcut, det_pt_nocut);

        // ------ get first n-1 rapidities and phi
        auto mt2 = pt2 + m_out_i * m_out_i;
        auto mt = sqrt(mt2);
        FVal<T> y_max_calc;
        if constexpr (fixed) {
            y_max_calc = chili_fixed_y_max<T>(_e_cm, m_out_i, mass_sum, mt2);
        } else {
            y_max_calc = log(sqrt(e_cm2 / 4. / mt2) + sqrt(e_cm2 / 4. / mt2 - 1.0));
        }
        y_max_calc = min(y_max_i, y_max_calc);
        auto y = y_max_calc * (2. * r_y - 1.0);
        auto phi = 2. * PI * r_phi + atan2(py_sum, px_sum);

        det_tmp = det_tmp / 4. * det_pt * 2. * y_max_calc * 2. * PI;

        // calculate the momenta
        auto sinhy = sinh(y);
        auto coshy = sqrt(1. + sinhy * sinhy);
        auto e_i = mt * coshy;
        auto px_i = pt * cos(phi);
        auto py_i = pt * sin(phi);
        auto pz_i = mt * sinhy;
        p_i[0] = e_i;
        p_i[1] = px_i;
        p_i[2] = py_i;
        p_i[3] = pz_i;
        e_sum = e_sum + e_i;
        px_sum = px_sum + px_i;
        py_sum = py_sum + py_i;
        pz_sum = pz_sum + pz_i;
    }
    return {e_sum, px_sum, py_sum, pz_sum, det_tmp};
}

// The inverse of chili_head_forward: the random numbers of the first `count`
// outgoing momenta, the sums of these momenta and the inverse Jacobian. With
// fixed beams the rapidity is read off as sign(p_z) log((E + |p_z|) / m_T),
// which keeps its precision far forward, where E - |p_z| cancels.
template <typename T, bool fixed = false>
KERNELSPEC ChiliSums<T> chili_head_inverse(
    FIn<T, 2> p_ext,
    std::size_t count,
    std::size_t pt_at,
    std::size_t y_at,
    std::size_t phi_at,
    FVal<T> _e_cm,
    FIn<T, 1> m_out,
    FSIn<T, 1> pt_min,
    FSIn<T, 1> y_max,
    FOut<T, 1> r,
    FVal<T> mass_sum = 0.
) {
    // need same sums as in forward mapping
    FVal<T> e_sum(0.), px_sum(0.), py_sum(0.), pz_sum(0.);
    FVal<T> det_tmp(1.);
    auto e_cm2 = _e_cm * _e_cm;
    auto pt_max = _e_cm / 2;
    auto pt2_max = pt_max * pt_max;
    for (std::size_t i = 0; i < count; ++i) {
        FVal<T> pt_min_i(pt_min[i]), y_max_i(y_max[i]), m_out_i(m_out[i]);
        auto p_i = load_mom<T>(p_ext[i + 2]);
        auto E = p_i[0], px = p_i[1], py = p_i[2], pz = p_i[3];

        // ---- invert pt mapping
        auto pt2 = px * px + py * py;
        auto pt = sqrt(max(pt2, 0.0));
        auto pt2_min_i = pt_min_i * pt_min_i;

        // with pt cut inversion:
        auto inv_pt2 = 1.0 / max(pt2, pt2_min_i);
        auto r_pt_cut = pt2_max * inv_pt2 * (pt2 - pt2_min_i) / (pt2_max - pt2_min_i);

        // no-cut inversion:
        auto mm_pt_i = where(m_out_i > EPS, m_out_i, 1.0);
        auto r_pt_nocut =
            pt * (2.0 * mm_pt_i + pt_max) / (pt_max * (2.0 * mm_pt_i + pt));

        auto has_pt_cut = pt2_min_i > 1e-9;
        auto r_pt = where(has_pt_cut, r_pt_cut, r_pt_nocut);
        r[pt_at + i] = r_pt;

        // get determinant
        auto pt2_withcut = 1. / (r_pt / pt2_max + (1. - r_pt) / pt2_min_i);
        auto det_pt_withcut =
            pt2_withcut * pt2_withcut * (1. / pt2_min_i - 1. / pt2_max);
        auto pt_nocut =
            2. * mm_pt_i * pt_max * r_pt / (2. * mm_pt_i + pt_max * (1. - r_pt));
        auto factor_pt = (2 * mm_pt_i + pt_nocut);
        auto denom_pt = mm_pt_i * (2 * mm_pt_i + pt_max);
        auto det_pt_nocut = pt_nocut * pt_max * factor_pt * factor_pt / denom_pt;

        auto det_pt = where(has_pt_cut, det_pt_withcut, det_pt_nocut);

        // ---- invert phi
        // forward: phi = 2 pi r_phi + atan2(py_sum, px_sum)
        if (i == 0) {
            auto phi = atan2(py, px);
            phi = where(phi < 0, phi + 2. * PI, phi);
            auto r_phi = phi / (2. * PI);
            r[phi_at + i] = r_phi;
        } else {
            // for i>0, need to subtract previous sums
            auto base = atan2(py_sum, px_sum);
            auto phi = atan2(py, px);
            auto r_phi = (phi - base) / (2. * PI);
            r_phi = where(r_phi < 0.0, r_phi + 1.0, r_phi);
            r_phi = where(r_phi >= 1.0, r_phi - 1.0, r_phi);
            r[phi_at + i] = r_phi;
        }

        // ---- compute y from momentum (same rapidity used forward)
        auto mt2 = pt2 + m_out_i * m_out_i;
        auto mt = sqrt(mt2);
        FVal<T> y, y_max_calc;
        if constexpr (fixed) {
            auto pz_abs = fabs(pz);
            auto y_abs = log((E + pz_abs) / max(mt, FVal<T>(EPS)));
            y = where(pz < 0., -y_abs, y_abs);
            y_max_calc = chili_fixed_y_max<T>(_e_cm, m_out_i, mass_sum, mt2);
        } else {
            y = 0.5 * log((E + pz) / (E - pz));
            y_max_calc = log(sqrt(e_cm2 / 4. / mt2) + sqrt(e_cm2 / 4. / mt2 - 1.0));
        }
        y_max_calc = min(y_max_i, y_max_calc);

        // invert y = y_max_calc * (2 r_y - 1)
        auto r_y = (y / y_max_calc + 1.0) * 0.5;
        r[y_at + i] = r_y;

        det_tmp = det_tmp * 4.0 / (det_pt * (2.0 * y_max_calc) * (2.0 * PI));

        // update sums (must match forward)
        e_sum = e_sum + E;
        px_sum = px_sum + px;
        py_sum = py_sum + py;
        pz_sum = pz_sum + pz;
    }
    return {e_sum, px_sum, py_sum, pz_sum, det_tmp};
}

template <typename T>
KERNELSPEC void kernel_chili_forward(
    FIn<T, 1> r,
    FIn<T, 0> e_cm,
    FIn<T, 1> m_out,
    FSIn<T, 1> pt_min,
    FSIn<T, 1> y_max,
    FOut<T, 2> p_ext,
    FOut<T, 0> det
) {
    // Handle the first n-1 outgoing t-channel particles
    FVal<T> _e_cm(e_cm);
    auto e_cm2 = _e_cm * _e_cm;
    auto n_out = m_out.size();
    auto head = chili_head_forward<T>(
        r, n_out - 1, 0, n_out - 1, 2 * n_out - 1, _e_cm, m_out, pt_min, y_max, p_ext
    );
    auto e_sum = head.e, px_sum = head.px, py_sum = head.py, pz_sum = head.pz;
    auto det_tmp = head.det;

    // Build last t-channel momentum
    FVal<T> r_y(r[2 * n_out - 2]), y_max_n(y_max[n_out - 1]), m_out_n(m_out[n_out - 1]);
    auto p_n = p_ext[n_out + 1];
    auto mj2 = e_sum * e_sum - px_sum * px_sum - py_sum * py_sum - pz_sum * pz_sum;
    auto yj = 0.5 * log((e_sum + pz_sum) / (e_sum - pz_sum));
    yj = where(e_sum < EPS, 99.0, yj);
    auto ptj2 = px_sum * px_sum + py_sum * py_sum;
    auto m2 = m_out_n * m_out_n;
    auto qt = sqrt(ptj2 + m2);
    auto mt = sqrt(ptj2 + max(mj2, 0.));
    auto y_min_calc = -log(_e_cm / qt * (1.0 - mt / _e_cm * exp(-yj)));
    y_min_calc = max(y_min_calc, -y_max_n); // apply potential cuts
    auto y_max_calc = log(_e_cm / qt * (1.0 - mt / _e_cm * exp(yj)));
    y_max_calc = min(y_max_calc, y_max_n); // apply potential cuts
    auto dely = y_max_calc - y_min_calc;
    auto yn = y_min_calc + r_y * dely;
    det_tmp = det_tmp * dely / e_cm2;
    auto sinhyn = sinh(yn);
    auto coshyn = sqrt(1. + sinhyn * sinhyn);
    auto e_n = qt * coshyn;
    auto pz_n = qt * sinhyn;
    p_n[0] = e_n;
    p_n[1] = -px_sum;
    p_n[2] = -py_sum;
    p_n[3] = pz_n;
    e_sum = e_sum + e_n;
    pz_sum = pz_sum + pz_n;

    // Build incoming momenta
    auto pp = e_sum + pz_sum;
    auto pm = e_sum - pz_sum;
    auto p_a = p_ext[0], p_b = p_ext[1];
    p_a[0] = pp / 2.;
    p_a[1] = 0.;
    p_a[2] = 0.;
    p_a[3] = pp / 2.;
    p_b[0] = pm / 2.;
    p_b[1] = 0.;
    p_b[2] = 0.;
    p_b[3] = -pm / 2.;

    // Get the bjorken variables
    auto x1 = pp / e_cm;
    auto x2 = pm / e_cm;

    // keep invalid point but set det/weight to zero (important to obtain correct
    // integral)
    det = where((x1 < 1.) & (x2 < 1.), det_tmp, 0.);
}

template <typename T>
KERNELSPEC void kernel_chili_inverse(
    FIn<T, 2> p_ext,
    FIn<T, 0> e_cm,
    FIn<T, 1> m_out,
    FSIn<T, 1> pt_min,
    FSIn<T, 1> y_max,
    FOut<T, 1> r,
    FOut<T, 0> det
) {

    // Handle the first n-1 outgoing particles
    FVal<T> _e_cm(e_cm);
    auto e_cm2 = _e_cm * _e_cm;
    auto n_out = m_out.size();
    auto head = chili_head_inverse<T>(
        p_ext, n_out - 1, 0, n_out - 1, 2 * n_out - 1, _e_cm, m_out, pt_min, y_max, r
    );
    auto e_sum = head.e, px_sum = head.px, py_sum = head.py, pz_sum = head.pz;
    auto det_tmp = head.det;

    // -------------------------------------------------------------------------
    // Invert last outgoing momentum p_ext[n_out+1]
    auto p_n = load_mom<T>(p_ext[n_out + 1]);
    auto E_n = p_n[0];
    auto pz_n = p_n[3];

    auto mj2 = e_sum * e_sum - px_sum * px_sum - py_sum * py_sum - pz_sum * pz_sum;
    auto yj = 0.5 * log((e_sum + pz_sum) / (e_sum - pz_sum));
    yj = where(e_sum < EPS, 99.0, yj);
    auto ptj2 = px_sum * px_sum + py_sum * py_sum;
    auto m_n = m_out[n_out - 1];
    auto qt = sqrt(ptj2 + m_n * m_n);
    auto mt = sqrt(ptj2 + max(mj2, 0.0));
    auto y_max_n = y_max[n_out - 1];
    auto y_min_calc = -log(_e_cm / qt * (1.0 - mt / _e_cm * exp(-yj)));
    y_min_calc = max(y_min_calc, -y_max_n);
    auto y_max_calc = log(_e_cm / qt * (1.0 - mt / _e_cm * exp(yj)));
    y_max_calc = min(y_max_calc, y_max_n);
    auto dely = y_max_calc - y_min_calc;
    det_tmp = det_tmp * (e_cm2 / dely);

    // invert yn from last momentum
    auto yn = 0.5 * log((E_n + pz_n) / (E_n - pz_n));
    auto r_y = (yn - y_min_calc) / dely;
    r[2 * n_out - 2] = r_y;

    det = det_tmp;
}

// Chili with fixed beams (lepton colliders): the incoming momenta add up to
// P = (e_cm, 0, 0, 0), so the final state cannot fix them as in
// kernel_chili_forward, and its two longitudinal degrees of freedom are not
// free. The first n - 2 outgoing momenta are generated as there, and the last
// two as the two-body decay of what is left, Q = P - sum of the others, in the
// rest frame of Q (r[3 (n - 2)] for cos(theta), r[3 (n - 2) + 1] for phi):
//   dPhi_n = prod_{i < n-2} (dpt_i^2 dy_i dphi_i / 4) dPhi_2(Q),
//   dPhi_2(Q) = pi p* / M_Q per unit of the two random numbers,
// with p* the momentum of the decay in the rest frame of Q. A point with Q not
// time-like or below the threshold of the last two masses gets zero weight.
// p_ext[0], p_ext[1] are the massless beams.
template <typename T>
KERNELSPEC void kernel_chili_forward_fixed(
    FIn<T, 1> r,
    FIn<T, 0> e_cm,
    FIn<T, 1> m_out,
    FSIn<T, 1> pt_min,
    FSIn<T, 1> y_max,
    FOut<T, 2> p_ext,
    FOut<T, 0> det
) {
    FVal<T> _e_cm(e_cm);
    auto n_out = m_out.size();
    auto k = n_out - 2;
    FVal<T> mass_sum(0.);
    for (std::size_t i = 0; i < n_out; ++i) {
        mass_sum = mass_sum + FVal<T>(m_out[i]);
    }
    auto head = chili_head_forward<T, true>(
        r, k, 0, k, 2 * k, _e_cm, m_out, pt_min, y_max, p_ext, mass_sum
    );

    // the decay of Q into the last two
    FourMom<T> q{_e_cm - head.e, -head.px, -head.py, -head.pz};
    auto q_mass2 = lsquare<T>(q);
    FVal<T> m1(m_out[n_out - 2]), m2(m_out[n_out - 1]);
    auto physical = (q[0] > 0.) & (q_mass2 > 0.);
    // an unphysical Q is replaced by one at rest above the threshold, which
    // keeps the momenta finite
    FourMom<T> q_safe{
        where(physical, q[0], m1 + m2 + 1.),
        where(physical, q[1], FVal<T>(0.)),
        where(physical, q[2], FVal<T>(0.)),
        where(physical, q[3], FVal<T>(0.))
    };
    auto q_safe_mass2 = lsquare<T>(q_safe);
    auto m0 = sqrt(q_safe_mass2);
    auto decay = two_body_decay<T>(r[3 * k + 1], r[3 * k], m0, m1, m2);
    auto p1_rest = decay.first;
    auto e2_rest = 0.5 * (m0 - (m1 - m2) * (m1 + m2) / max(m0, FVal<T>(EPS)));
    FourMom<T> p2_rest{max(e2_rest, 0.), -p1_rest[1], -p1_rest[2], -p1_rest[3]};
    auto p_out =
        boost_two_body<T>(p1_rest, m1 * m1, p2_rest, m2 * m2, q_safe, q_safe_mass2);
    store_mom<T>(p_ext[n_out], p_out.first);
    store_mom<T>(p_ext[n_out + 1], p_out.second);

    auto p_a = p_ext[0], p_b = p_ext[1];
    p_a[0] = 0.5 * _e_cm;
    p_a[1] = 0.;
    p_a[2] = 0.;
    p_a[3] = 0.5 * _e_cm;
    p_b[0] = 0.5 * _e_cm;
    p_b[1] = 0.;
    p_b[2] = 0.;
    p_b[3] = -0.5 * _e_cm;

    det = where(physical, head.det * decay.second, FVal<T>(0.));
}

template <typename T>
KERNELSPEC void kernel_chili_inverse_fixed(
    FIn<T, 2> p_ext,
    FIn<T, 0> e_cm,
    FIn<T, 1> m_out,
    FSIn<T, 1> pt_min,
    FSIn<T, 1> y_max,
    FOut<T, 1> r,
    FOut<T, 0> det
) {
    FVal<T> _e_cm(e_cm);
    auto n_out = m_out.size();
    auto k = n_out - 2;
    FVal<T> mass_sum(0.);
    for (std::size_t i = 0; i < n_out; ++i) {
        mass_sum = mass_sum + FVal<T>(m_out[i]);
    }
    auto head = chili_head_inverse<T, true>(
        p_ext, k, 0, k, 2 * k, _e_cm, m_out, pt_min, y_max, r, mass_sum
    );

    // the decay of Q = p1 + p2 into the last two, in its rest frame
    auto p1 = load_mom<T>(p_ext[n_out]), p2 = load_mom<T>(p_ext[n_out + 1]);
    FourMom<T> q;
    for (int i = 0; i < 4; ++i) {
        q[i] = p1[i] + p2[i];
    }
    auto decay = two_body_decay_inverse<T>(q, boost<T>(p1, q, -1.), p2);
    r[3 * k] = decay.second;
    r[3 * k + 1] = decay.first;
    det = head.det * decay.sixth;
}

} // namespace kernels
} // namespace madspace

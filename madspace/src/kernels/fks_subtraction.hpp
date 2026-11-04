#pragma once

#include "kinematics.hpp"

namespace madspace {
namespace kernels {

// The ingredients of the FKS subtraction of fks_Sij.f and fks_singular.f of
// MadGraph (arXiv:hep-ph/9512328, arXiv:0908.4272): sector functions, the
// factors of the soft and collinear limits and the prefactors of the
// counterterms. Each event takes the tables of its FKS configuration, and the
// momenta can be in any frame.

// The FKS sector functions S_ij of fks_Sij.f. d_kl = (E_k E_l / (sqrt_s / 2)^2)^a
// (1 - beta_k beta_l cos theta_kl)^b, energies and angles in the partonic
// center-of-mass frame of the momenta (written with invariants, so that the
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
// i), masses m1, m2 and color octets o1, o2. With both, the energy of 1 is
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
// with t = z s / 4. z = 1 - E_i / (E_i + E_j) in the partonic center-of-mass
// frame of the momenta for a final-state j, 1 - xi = 1 - 2 E_i / sqrt(s) for
// an initial-state one.
// kind selects the splitting function of AP_reduced (0: g -> g g, 1: g/a ->
// q q~, 2: q -> q g/a, 3: q -> g/a q) and factors holds, per configuration,
// the QCD and QED factors of the kernel and of the azimuthal terms.
// (1 - z) P(z) of AP_reduced without its color factor, for the splittings
// 0: g -> g g, 1: g/a -> q q~, 2: q -> q g/a, 3: q -> g/a q
template <typename T>
KERNELSPEC FVal<T> fks_ap_shape(std::size_t kind, FVal<T> z) {
    auto w = 1. - z;
    switch (kind) {
    case 0:
        return w * w / z + z + z * w * w;
    case 1:
        return (z * z + w * w) * w;
    case 2:
        return 1. + z * z;
    default:
        return (1. + w * w) * w / z;
    }
}

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
    // energies in the center-of-mass frame, up to the common 1 / sqrt(s)
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
    FVal<T> f_ap = fks_ap_shape<T>(single_index(kind[c]), zz);
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
// partonic center-of-mass frame it is (0, cos phi, sin phi, 0) in the frame of
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

// The prefactors of the real emission and its counterterms of
// compute_prefactors_n1body (fks_singular.f), for the Jacobians of the
// events without the factor xi and without the symmetry factors: the
// plus distributions in xi and 1 - y with their cuts xicut, xiScut, delta and
// deltaS. The collinear ones vanish for a massive emitter.
template <typename T>
KERNELSPEC void kernel_fks_prefactors(
    FIn<T, 0> xi,
    FIn<T, 0> y,
    FIn<T, 0> xi_max,
    FIn<T, 0> xi_c,
    FIn<T, 0> xi_max_c,
    FIn<T, 2> p_c,
    FIn<T, 2> p_sc,
    IIn<T, 0> index,
    IIn<T, 1> collinear,
    FIn<T, 1> delta,
    FIn<T, 1> options,
    FOut<T, 0> f_r,
    FOut<T, 0> f_s,
    FOut<T, 0> f_c,
    FOut<T, 0> f_dc,
    FOut<T, 0> f_sc,
    FOut<T, 1> f_dsc,
    FOut<T, 1> f_pdf
) {
    constexpr double pi = 3.14159265358979323846;
    std::size_t c = single_index(index);
    FVal<T> xicut = options[0], xi_s_cut = options[1], delta_s = options[2];
    FVal<T> omy = 1. - FVal<T>(y);
    FVal<T> x = xi;
    f_r = 1. / (x * omy);
    auto xi_lim = min(FVal<T>(xi_max), xi_s_cut);
    f_s = (1. / x + log(xicut / xi_lim) / xi_lim) / omy;
    if (single_index(collinear[c]) == 0) {
        f_c = 0.;
        f_dc = 0.;
        f_sc = 0.;
        for (std::size_t k = 0; k < 4; ++k) {
            f_dsc[k] = 0.;
        }
        for (std::size_t k = 0; k < 3; ++k) {
            f_pdf[k] = 0.;
        }
        return;
    }
    FVal<T> x_c = xi_c;
    auto l_delta = log(FVal<T>(delta[c]) / delta_s) / delta_s;
    auto in_y = 1. / omy + l_delta;
    f_c = in_y / x_c;
    auto pc1 = load_mom<T>(p_c[0]), pc2 = load_mom<T>(p_c[1]);
    auto s_c = lsquare<T>(
        FourMom<T>{pc1[0] + pc2[0], pc1[1] + pc2[1], pc1[2] + pc2[2], pc1[3] + pc2[3]}
    );
    f_dc = 32. * pi * pi / (s_c * x_c * delta_s);
    auto xi_lim_c = min(FVal<T>(xi_max_c), xi_s_cut);
    auto l_xi = log(xicut / xi_lim_c);
    f_sc = in_y * (1. / x_c + l_xi / xi_lim_c);
    auto ps1 = load_mom<T>(p_sc[0]), ps2 = load_mom<T>(p_sc[1]);
    auto s_sc = lsquare<T>(
        FourMom<T>{ps1[0] + ps2[0], ps1[1] + ps2[1], ps1[2] + ps2[2], ps1[3] + ps2[3]}
    );
    auto g = 32. * pi * pi / (s_sc * delta_s);
    auto log_cut = log(xicut), log_lim = log(xi_lim_c);
    f_dsc[0] = g / x_c;
    f_dsc[1] = g * l_xi / xi_lim_c;
    // as a product, which vanishes exactly for xi_lim_c = xicut
    f_dsc[2] = g * (log_cut - log_lim) * (log_cut + log_lim) / (2. * xi_lim_c);
    f_dsc[3] = g * (1. / x_c + l_xi / xi_lim_c);
    // the delta, plus and log-plus parts of a change of the PDF scheme
    auto log_s_cut = log(xi_s_cut);
    f_pdf[0] = g / xi_s_cut;
    f_pdf[1] = g * log_s_cut / xi_s_cut;
    f_pdf[2] = g * log_s_cut * log_s_cut / (2. * xi_s_cut);
}

// The soft limit of xi^2 (1 - y) times the real matrix element of sbornsoft
// (fks_singular.f): 4 pi alpha_s sum_(m, n) f_mn E_mn <T_m . T_n> B over the
// pairs of the configuration, positions in the Born, with f_mn = -2 iden_comp
// and half that for m = n (the color-linked Born of a particle with itself has
// a factor 1/2 in MadFKS). A configuration without a soft limit has no pairs.
template <typename T>
KERNELSPEC void kernel_fks_soft_limit(
    FIn<T, 1> eik,
    FIn<T, 2> color_correlated,
    FIn<T, 0> alpha_s,
    IIn<T, 0> index,
    IIn<T, 1> pair_count,
    IIn<T, 2> pair_m,
    IIn<T, 2> pair_n,
    FIn<T, 2> pair_factor,
    FOut<T, 0> soft
) {
    constexpr double pi = 3.14159265358979323846;
    std::size_t c = single_index(index);
    std::size_t count = single_index(pair_count[c]);
    FVal<T> sum = 0.;
    for (std::size_t k = 0; k < count; ++k) {
        std::size_t m = single_index(pair_m[c][k]), n = single_index(pair_n[c][k]);
        FVal<T> factor = pair_factor[c][k], e = eik[k], b = color_correlated[m][n];
        sum = sum + factor * e * b;
    }
    soft = 4. * pi * FVal<T>(alpha_s) * sum;
}

// The collinear limit of xi^2 (1 - y) times the real matrix element of
// sborncol_fsr and sborncol_isr (fks_singular.f) from the QCD kernels:
// 4 pi alpha_s iden_comp (AP B + Q (B / 2 - B_kperp)), with the Born B_kperp
// whose emitter has the polarization vector k_perp. In MadFKS the Born that Q
// multiplies is the helicity-flipped Born times a phase of the azimuth of the
// emission, which is B / 2 - B_kperp. B_kperp enters only where Q is not zero.
template <typename T>
KERNELSPEC void kernel_fks_collinear_limit(
    FIn<T, 0> born,
    FIn<T, 0> spin_correlated,
    FIn<T, 1> ap,
    FIn<T, 1> q,
    FIn<T, 0> alpha_s,
    IIn<T, 0> index,
    FIn<T, 1> factor,
    FOut<T, 0> collinear
) {
    constexpr double pi = 3.14159265358979323846;
    std::size_t c = single_index(index);
    FVal<T> f = factor[c];
    if (f == 0.) {
        // no collinear limit (a massive emitter)
        collinear = 0.;
        return;
    }
    FVal<T> b = born, kernel = ap[0], azimuthal = q[0];
    FVal<T> value = kernel * b;
    if (azimuthal != 0.) {
        value = value + azimuthal * (0.5 * b - FVal<T>(spin_correlated));
    }
    collinear = 4. * pi * FVal<T>(alpha_s) * f * value;
}

// (1 - z) P'(z) of AP_reduced_prime (the O(epsilon) part) without its color
// factor, and the plus (p), log-plus (l) and delta (d) parts of the change
// of the PDF scheme from MSbar to DIS of xkplus, xklog and xkdelta, with
// their color factors and n_f, for the splittings of fks_ap_shape
template <typename T>
KERNELSPEC FVal<T> fks_ap_prime_shape(std::size_t kind, FVal<T> z) {
    auto w = 1. - z;
    switch (kind) {
    case 0:
        return 0.;
    case 1:
        return -2. * z * w * w;
    case 2:
        return -w * w;
    default:
        return -z * w;
    }
}

template <typename T>
KERNELSPEC void fks_dis_kernels(
    std::size_t kind, FVal<T> x, FVal<T> n_f, FVal<T>& k_p, FVal<T>& k_l, FVal<T>& k_d
) {
    constexpr double pi = 3.14159265358979323846;
    constexpr double tr = 0.5, cf = 4. / 3.;
    auto w = 1. - x;
    auto lx = log(x);
    auto pqg = x * x + w * w;
    auto pgq = 1. + x * x;
    switch (kind) {
    case 0:
        k_p = -2. * n_f * tr * w * (-pqg * lx + 8. * x * w - 1.);
        k_l = -2. * n_f * tr * w * pqg;
        k_d = 0.;
        break;
    case 1:
        k_p = tr * w * (-pqg * lx + 8. * x * w - 1.);
        k_l = tr * w * pqg;
        k_d = 0.;
        break;
    case 2:
        k_p = cf * (-1.5 - pgq * lx + w * (3. + 2. * x));
        k_l = cf * pgq;
        k_d = -cf * (4.5 + pi * pi / 3.);
        break;
    default:
        k_p = -cf * (-1.5 - pgq * lx + w * (3. + 2. * x));
        k_l = -cf * pgq;
        k_d = cf * (4.5 + pi * pi / 3.);
        break;
    }
}

// The collinear remnants of an initial-state emitter of sreal_deg and
// compute_collinear_counter_term / compute_soft_collinear_counter_term
// (fks_singular.f), with the factorization scale of the beam of the
// emitter and the QCD kernels: with N = B iden_comp / (8 pi^2 z),
//   D_xi = N g^2 ((1 - z) P log(s delta_I / (2 mu_F^2)) - (1 - z) P'),
//   D_lxi = 2 N g^2 (1 - z) P,   K_p,l,d = -N g^2 k_p,l,d (zero in MSbar),
// the collinear one f_dc (D_xi + K_p + (D_lxi + K_l) log xi_c) at z = 1 - xi_c
// and s of the collinear counter-event, the soft-collinear one
// -f_dsc1 (D_xi + K_p + (D_lxi + K_l) log xi_c) - f_dsc2 D_xi - f_dsc3 D_lxi
// + f_pdf_d K_d + f_pdf_p K_p + f_pdf_l K_l at z = 1 and s of the
// soft-collinear counter-event. Each is zero where its sector function is
// (options = pdf_scheme, n_f).
template <typename T>
KERNELSPEC void kernel_fks_collinear_remnants(
    FIn<T, 0> born,
    FIn<T, 0> alpha_s,
    FIn<T, 0> s_c,
    FIn<T, 0> s_sc,
    FIn<T, 2> p_c,
    FIn<T, 2> p_sc,
    FIn<T, 0> xi_c,
    FIn<T, 0> f_dc,
    FIn<T, 1> f_dsc,
    FIn<T, 1> f_pdf,
    FIn<T, 0> fact_scale1,
    FIn<T, 0> fact_scale2,
    IIn<T, 0> index,
    IIn<T, 1> remnant,
    IIn<T, 1> beam,
    IIn<T, 1> kind,
    FIn<T, 1> factor,
    FIn<T, 1> delta,
    FIn<T, 1> options,
    FOut<T, 0> collinear,
    FOut<T, 0> soft_collinear
) {
    constexpr double pi = 3.14159265358979323846;
    std::size_t c = single_index(index);
    collinear = 0.;
    soft_collinear = 0.;
    if (single_index(remnant[c]) == 0) {
        return;
    }
    std::size_t k = single_index(kind[c]);
    bool dis = FVal<T>(options[0]) == 1.;
    FVal<T> n_f = options[1];
    FVal<T> mu_f =
        single_index(beam[c]) == 0 ? FVal<T>(fact_scale1) : FVal<T>(fact_scale2);
    FVal<T> d = delta[c];
    // N g^2 for z = 1
    FVal<T> norm = FVal<T>(born) * FVal<T>(factor[c]) * 4. * pi * FVal<T>(alpha_s) /
        (8. * pi * pi);
    auto pieces =
        [&](FVal<T> z,
            FVal<T> s,
            FVal<T>& d_xi,
            FVal<T>& d_lxi,
            FVal<T>& k_p,
            FVal<T>& k_l,
            FVal<T>& k_d) {
            auto n = norm / z;
            auto ap = fks_ap_shape<T>(k, z), ap_prime = fks_ap_prime_shape<T>(k, z);
            // the color factor of AP_reduced
            FVal<T> color = k == 0 ? 6. : (k == 1 ? 0.5 : 4. / 3.);
            d_xi = n * color * (ap * log(s * d / (2. * mu_f * mu_f)) - ap_prime);
            d_lxi = 2. * n * color * ap;
            k_p = 0.;
            k_l = 0.;
            k_d = 0.;
            if (dis) {
                fks_dis_kernels<T>(k, z, n_f, k_p, k_l, k_d);
                k_p = -n * k_p;
                k_l = -n * k_l;
                k_d = -n * k_d;
            }
        };
    auto invariant_mass = [](FIn<T, 2> p) {
        auto p1 = load_mom<T>(p[0]), p2 = load_mom<T>(p[1]);
        return lsquare<T>(
            FourMom<T>{p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2], p1[3] + p2[3]}
        );
    };
    FVal<T> x_c = xi_c, log_xi = log(x_c);
    FVal<T> d_xi, d_lxi, k_p, k_l, k_d;
    if (FVal<T>(s_c) > 0.) {
        pieces(1. - x_c, invariant_mass(p_c), d_xi, d_lxi, k_p, k_l, k_d);
        collinear = FVal<T>(f_dc) * (d_xi + k_p + (d_lxi + k_l) * log_xi);
    }
    if (FVal<T>(s_sc) > 0.) {
        pieces(1., invariant_mass(p_sc), d_xi, d_lxi, k_p, k_l, k_d);
        soft_collinear = -FVal<T>(f_dsc[0]) * (d_xi + k_p + (d_lxi + k_l) * log_xi) -
            FVal<T>(f_dsc[1]) * d_xi - FVal<T>(f_dsc[2]) * d_lxi +
            FVal<T>(f_pdf[0]) * k_d + FVal<T>(f_pdf[1]) * k_p + FVal<T>(f_pdf[2]) * k_l;
    }
}

// The real matrix element times the sector function at the event, with the
// limits of sreal and fks_Sij (fks_singular.f, fks_Sij.f) below tiny
// (options[0], zero for none): below tiny in 1 - y the collinear limit at the
// event over xi^2 (1 - y), else below tiny in xi the soft one, in place of the
// matrix element, which is then not used. The limits are zero without a soft
// (or collinear) singularity. For xi <= tiny the sector function has the
// direction of i at the soft counter-event, at the soft-collinear one where
// also 1 - y < tiny and j is massless. Without the counter-event (zero det)
// the event keeps its own direction and matrix element, as fks_Sij.
template <typename T>
KERNELSPEC void kernel_fks_real_emission(
    FIn<T, 0> matrix_element,
    FIn<T, 0> s,
    FIn<T, 0> s_soft,
    FIn<T, 0> s_soft_collinear,
    FIn<T, 0> soft,
    FIn<T, 0> collinear,
    FIn<T, 0> xi,
    FIn<T, 0> y,
    FIn<T, 0> det_s,
    FIn<T, 0> det_sc,
    IIn<T, 0> index,
    IIn<T, 1> massless,
    FIn<T, 1> options,
    FOut<T, 0> real
) {
    std::size_t c = single_index(index);
    FVal<T> tiny = options[0], x = xi, omy = 1. - FVal<T>(y);
    bool on = tiny > 0.;
    bool with_soft = FVal<T>(det_s) > 0.;
    bool with_soft_collinear = FVal<T>(det_sc) > 0. && single_index(massless[c]) != 0;
    FVal<T> sector = s;
    if (on && x <= tiny) {
        if (omy < tiny && with_soft_collinear) {
            sector = s_soft_collinear;
        } else if (with_soft) {
            sector = s_soft;
        }
    }
    FVal<T> value = matrix_element;
    if (on && omy < tiny) {
        value = FVal<T>(collinear) / (x * x * omy);
    } else if (on && x < tiny && with_soft) {
        value = FVal<T>(soft) / (x * x * omy);
    }
    real = sector * value;
}

// The real emission and its counterterms of compute_real_emission and
// compute_soft_counter_term and its siblings (fks_singular.f), from the
// products of the matrix element and its limits with the sector functions
// and prefactors and the collinear remnants: signs, symmetry factors and the
// conditions of each counterterm (options = xiScut, deltaS), zero where its
// momenta are not generated.
template <typename T>
KERNELSPEC void kernel_fks_subtraction(
    FIn<T, 0> real,
    FIn<T, 0> soft,
    FIn<T, 0> collinear,
    FIn<T, 0> soft_collinear,
    FIn<T, 0> remnant_c,
    FIn<T, 0> remnant_sc,
    FIn<T, 0> xi,
    FIn<T, 0> xi_hat,
    FIn<T, 0> y,
    FIn<T, 0> xi_max_s,
    FIn<T, 0> xi_c,
    FIn<T, 0> xi_max_c,
    FIn<T, 0> weight,
    FIn<T, 0> det_s,
    FIn<T, 0> det_c,
    FIn<T, 0> det_sc,
    IIn<T, 0> index,
    FIn<T, 1> symmetry,
    FIn<T, 1> options,
    FOut<T, 0> real_out,
    FOut<T, 0> soft_out,
    FOut<T, 0> collinear_out,
    FOut<T, 0> soft_collinear_out
) {
    std::size_t c = single_index(index);
    FVal<T> sym = symmetry[c], xi_s_cut = options[0], delta_s = options[1];
    FVal<T> x = xi, x_hat = xi_hat, x_c = xi_c, yy = y;
    bool in_y = yy > 1. - delta_s;
    real_out = FVal<T>(weight) > 0. ? sym * FVal<T>(real) : FVal<T>(0.);
    bool with_soft =
        FVal<T>(det_s) > 0. && x_hat * FVal<T>(xi_max_s) <= xi_s_cut && x <= xi_s_cut;
    soft_out = with_soft ? -sym * FVal<T>(soft) : FVal<T>(0.);
    bool with_collinear = FVal<T>(det_c) > 0. && in_y;
    collinear_out =
        with_collinear ? sym * (FVal<T>(remnant_c) - FVal<T>(collinear)) : FVal<T>(0.);
    bool with_soft_collinear = FVal<T>(det_sc) > 0. && in_y &&
        x_hat * FVal<T>(xi_max_c) < xi_s_cut && x_c < xi_s_cut;
    soft_collinear_out = with_soft_collinear
        ? sym * (FVal<T>(soft_collinear) + FVal<T>(remnant_sc))
        : FVal<T>(0.);
}

// The dilogarithm of a real argument as DDILOG of CERNLIB (fks_singular.f),
// with its Chebyshev series
template <typename T>
KERNELSPEC FVal<T> fks_dilog(FVal<T> x) {
    constexpr double pi = 3.14159265358979323846;
    constexpr double pi3 = pi * pi / 3., pi6 = pi * pi / 6., pi12 = pi * pi / 12.;
    constexpr double c[20] = {
        0.42996693560813697,
        0.40975987533077105,
        -0.01858843665014592,
        0.00145751084062268,
        -0.00014304184442340,
        0.00001588415541880,
        -0.00000190784959387,
        0.00000024195180854,
        -0.00000003193341274,
        0.00000000434545063,
        -0.00000000060578480,
        0.00000000008612098,
        -0.00000000001244332,
        0.00000000000182256,
        -0.00000000000027007,
        0.00000000000004042,
        -0.00000000000000610,
        0.00000000000000093,
        -0.00000000000000014,
        0.00000000000000002,
    };
    if (x == 1.) {
        return pi6;
    }
    if (x == -1.) {
        return -pi12;
    }
    FVal<T> t = -x, y, s, a;
    if (t <= -2.) {
        y = -1. / (1. + t);
        s = 1.;
        FVal<T> l1 = log(-t), l2 = log(1. + 1. / t);
        a = -pi3 + 0.5 * (l1 * l1 - l2 * l2);
    } else if (t < -1.) {
        y = -1. - t;
        s = -1.;
        a = log(-t);
        a = -pi6 + a * (a + log(1. + 1. / t));
    } else if (t <= -0.5) {
        y = -(1. + t) / t;
        s = 1.;
        a = log(-t);
        a = -pi6 + a * (-0.5 * a + log(1. + t));
    } else if (t < 0.) {
        y = -t / (1. + t);
        s = -1.;
        FVal<T> l = log(1. + t);
        a = 0.5 * l * l;
    } else if (t <= 1.) {
        y = t;
        s = 1.;
        a = 0.;
    } else {
        y = 1. / t;
        s = -1.;
        FVal<T> l = log(t);
        a = pi6 + 0.5 * l * l;
    }
    FVal<T> h = y + y - 1., alpha = h + h, b0 = 0., b1 = 0., b2 = 0.;
    for (int k = 19; k >= 0; --k) {
        b0 = c[k] + alpha * b1 - b2;
        b2 = b1;
        b1 = b0;
    }
    return -(s * (b0 - h * b2) + a);
}

// The integrated eikonal I_mn of eikonal_Ireg (fks_singular.f) of the
// particles m and n with energies e_m, e_n in the partonic center-of-mass
// frame, p_m . p_n = k and masses mass_m, mass_n, for l_s = log(s / mu^2) and
// the cut xicut, m = n (same) only for a massive particle. For a massive and
// a massless particle the massless one is i of eikonal_Ireg, for two massive
// ones n is.
template <typename T>
KERNELSPEC FVal<T> fks_integrated_eikonal(
    FVal<T> e_m,
    FVal<T> e_n,
    FVal<T> k,
    FVal<T> mass_m,
    FVal<T> mass_n,
    bool same,
    FVal<T> l_s,
    FVal<T> xicut
) {
    constexpr double pi = 3.14159265358979323846;
    FVal<T> log_xicut = log(xicut);
    // log(xicut^2 s / mu^2)
    FVal<T> l = 2. * log_xicut + l_s;
    if (mass_m == 0. && mass_n == 0.) {
        FVal<T> r = k / (2. * e_m * e_n);
        if (fabs(r - 1.) > 1e-6) {
            FVal<T> lr = log(r);
            return 0.5 * l * l + l * lr - fks_dilog<T>(r) + 0.5 * lr * lr -
                log(1. - r) * lr;
        }
        return 0.5 * l * l - pi * pi / 6.;
    }
    if (mass_m == 0. || mass_n == 0.) {
        bool m_massless = mass_m == 0.;
        FVal<T> e_i = m_massless ? e_m : e_n, e_j = m_massless ? e_n : e_m;
        FVal<T> mass = m_massless ? mass_n : mass_m;
        FVal<T> beta = sqrt(1. - mass * mass / (e_j * e_j));
        FVal<T> r = k / (2. * e_i * e_j);
        FVal<T> lk = log(k / (mass * e_i));
        FVal<T> lr = log(2. * r / (1. - beta));
        FVal<T> lb = log((1. + beta) / (1. - beta));
        return log_xicut * (log_xicut + l_s + 2. * lk) -
            fks_dilog<T>(1. - (1. + beta) / (2. * r)) +
            fks_dilog<T>(1. - 2. * r / (1. - beta)) + 0.5 * lr * lr + l_s * lk -
            pi * pi / 12. + 0.25 * l_s * l_s - 0.25 * lb * lb;
    }
    if (same) {
        FVal<T> beta = sqrt(1. - mass_n * mass_n / (e_n * e_n));
        if (beta > 1e-6) {
            return l - log((1. + beta) / (1. - beta)) / beta;
        }
        FVal<T> b2 = beta * beta;
        return l - 2. * (1. + b2 / 3. + b2 * b2 / 5.);
    }
    FVal<T> e_i = e_n, e_j = e_m;
    FVal<T> mi2 = mass_n * mass_n, mj2 = mass_m * mass_m;
    FVal<T> beta_i = sqrt(1. - mi2 / (e_i * e_i));
    FVal<T> beta_j = sqrt(1. - mj2 / (e_j * e_j));
    FVal<T> v = sqrt(1. - mi2 * mj2 / (k * k));
    FVal<T> al = k * (1. + v) / mi2;
    FVal<T> t_vl = (al * al * mi2 - mj2) / 2.;
    FVal<T> t_v = t_vl / (al * e_i - e_j);
    FVal<T> xi0 = v < 1. ? log((1. + v) / (1. - v)) / v : log(4. * k * k / (mi2 * mj2));
    auto j1a = [&](FVal<T> x, FVal<T> y) {
        FVal<T> lxy = log((x - y) / (x + y));
        return (lxy * lxy + 4. * fks_dilog<T>(1. - (x + y) / t_v) +
                4. * fks_dilog<T>(1. - (x - y) / t_v)) /
            (2. * t_vl);
    };
    FVal<T> arg1 = al * e_i, arg3 = e_j;
    FVal<T> xi1a =
        k * k * (1. + v) / mi2 * (j1a(arg1, arg1 * beta_i) - j1a(arg3, arg3 * beta_j));
    return 0.5 * xi0 * l + 0.5 * xi1a;
}

// The n-body part of bornsoftvirtual (fks_singular.f) without the virtual,
// for QCD and with the Ellis-Sexton scale at the renormalization scale mu_R:
// with a = alpha_s / (2 pi), a B sum_j Q_j over the massless gluons and quarks
// plus a sum_(m, n) f_mn <T_m . T_n> B I_mn over the pairs of color-linked
// particles, f_mn = -2 and -1 for a massive particle with itself (the
// color-linked Born of MadFKS has a factor 1/2 there), with the integrated
// eikonals I_mn of eikonal_Ireg. The Q term of a final-state parton j is
// gamma'_j - log(s delta_O / (2 mu_R^2)) (gamma_j - 2 C_j log(2 E_j / (xicut
// sqrt_s))) + 2 C_j (log^2(2 E_j / sqrt_s) - log^2 xicut) - 2 gamma_j log(2 E_j
// / sqrt_s), with its energy E_j in the partonic center-of-mass frame
// (written with invariants), that of an incoming one -log(mu_F^2 / mu_R^2)
// (gamma_j + 2 C_j log xicut) with the factorization scale of its beam. Also
// the coefficients of log mu_R^2, a b gamma_g B with the power b of alpha_s of
// the Born, and of log mu_F^2 of each beam, -a (gamma_j + 2 C_j log xicut) B,
// at a fixed virtual (options = xicut, delta_O, b gamma_g).
template <typename T>
KERNELSPEC void kernel_fks_soft_virtual(
    FIn<T, 2> momenta,
    FIn<T, 0> born,
    FIn<T, 2> color_correlated,
    FIn<T, 0> alpha_s,
    FIn<T, 0> ren_scale,
    FIn<T, 0> fact_scale1,
    FIn<T, 0> fact_scale2,
    FIn<T, 1> casimir,
    FIn<T, 1> gamma,
    FIn<T, 1> gamma_prime,
    FIn<T, 1> masses,
    IIn<T, 1> pair_m,
    IIn<T, 1> pair_n,
    FIn<T, 1> pair_factor,
    FIn<T, 1> options,
    FOut<T, 0> soft_virtual,
    FOut<T, 0> ren_coefficient,
    FOut<T, 1> fact_coefficients
) {
    constexpr double pi = 3.14159265358979323846;
    FVal<T> xicut = options[0], delta_o = options[1], beta = options[2];
    auto p1 = load_mom<T>(momenta[0]), p2 = load_mom<T>(momenta[1]);
    FourMom<T> p_tot{p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2], p1[3] + p2[3]};
    FVal<T> s = lsquare<T>(p_tot), rs = sqrt(s);
    FVal<T> mu_r = ren_scale, mu_r2 = mu_r * mu_r;
    FVal<T> l_s = log(s / mu_r2), log_xicut = log(xicut);
    FVal<T> a = FVal<T>(alpha_s) / (2. * pi), b = born;
    FVal<T> q = 0.;
    for (std::size_t j = 0; j < 2; ++j) {
        FVal<T> mu_f = j == 0 ? FVal<T>(fact_scale1) : FVal<T>(fact_scale2);
        FVal<T> coefficient =
            -(FVal<T>(gamma[j]) + 2. * FVal<T>(casimir[j]) * log_xicut);
        fact_coefficients[j] = a * b * coefficient;
        q = q + coefficient * log(mu_f * mu_f / mu_r2);
    }
    FVal<T> l_o = log(s * delta_o / (2. * mu_r2));
    for (std::size_t j = 2; j < momenta.size(); ++j) {
        FVal<T> c = casimir[j], g = gamma[j], g_prime = gamma_prime[j];
        if (c == 0.) {
            continue;
        }
        auto p = load_mom<T>(momenta[j]);
        FVal<T> l_e = log(2. * fks_frame_energy<T>(p, p_tot, rs) / rs);
        q = q + g_prime - l_o * (g - 2. * c * (l_e - log_xicut)) +
            2. * c * (l_e * l_e - log_xicut * log_xicut) - 2. * g * l_e;
    }
    FVal<T> eikonal = 0.;
    for (std::size_t k = 0; k < pair_factor.size(); ++k) {
        FVal<T> factor = pair_factor[k];
        if (factor == 0.) {
            continue;
        }
        std::size_t m = single_index(pair_m[k]), n = single_index(pair_n[k]);
        auto p_m = load_mom<T>(momenta[m]), p_n = load_mom<T>(momenta[n]);
        FVal<T> e_m = fks_frame_energy<T>(p_m, p_tot, rs);
        FVal<T> e_n = fks_frame_energy<T>(p_n, p_tot, rs);
        FVal<T> i_mn = fks_integrated_eikonal<T>(
            e_m, e_n, ldot<T>(p_m, p_n), masses[m], masses[n], m == n, l_s, xicut
        );
        eikonal = eikonal + factor * FVal<T>(color_correlated[m][n]) * i_mn;
    }
    soft_virtual = a * (b * q + eikonal);
    ren_coefficient = a * b * beta;
}

} // namespace kernels
} // namespace madspace

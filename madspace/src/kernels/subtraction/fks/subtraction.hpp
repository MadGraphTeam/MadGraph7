#pragma once

#include "../../kinematics.hpp"
#include "../common.hpp"

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

// the sum of the incoming momenta of p, the first incoming (two, or one for a
// decay), whose rest frame is the partonic center-of-mass frame
template <typename T>
KERNELSPEC FourMom<T> fks_incoming_sum(FIn<T, 2> p, std::size_t incoming) {
    auto p1 = load_mom<T>(p[0]);
    if (incoming == 1) {
        return p1;
    }
    auto p2 = load_mom<T>(p[1]);
    return {p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2], p1[3] + p2[3]};
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
    IIn<T, 0> incoming,
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

    std::size_t n_in = single_index(incoming);
    auto p1 = load_mom<T>(p[0]);
    auto p_tot = fks_incoming_sum<T>(p, n_in);
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
                if (single_index(ag[k]) != 0 && single_index(ag[l]) != 0 &&
                    jj >= n_in) {
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

// the momentum k of the resonance of the resonance-local recoil where the event
// has one (k_0 > 0), q where it has none (k = 0)
template <typename T>
KERNELSPEC FourMom<T> fks_frame_of(FIn<T, 1> k, FourMom<T> q) {
    auto res = load_mom<T>(k);
    auto resonant = res[0] > 0.;
    FourMom<T> frame;
    for (int mu = 0; mu < 4; ++mu) {
        frame[mu] = where(resonant, res[mu], q[mu]);
    }
    return frame;
}

// Reduced eikonal factors of eikonal_reduced (fks_singular.f) for the pairs
// (m, n) of the configuration: (p_m p_n) (1 - y) / ((p_m p_i_hat) (p_n
// p_i_hat)), with (p_j p_i_hat) / (1 - y) = (P p_j) / 2 for a massless j,
// which is finite at y = 1. P is the sum of the incoming momenta, sqrt_s E_j
// in the partonic center-of-mass frame, or the momentum of the resonance of
// the resonance-local recoil (frame). Entries past the pairs of the
// configuration are zero.
template <typename T>
KERNELSPEC void fks_eikonals_in_frame(
    FIn<T, 2> p,
    FIn<T, 1> p_i_hat,
    FIn<T, 0> y,
    FourMom<T> p_tot,
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
    IIn<T, 0> incoming,
    FOut<T, 1> eik
) {
    fks_eikonals_in_frame<T>(
        p,
        p_i_hat,
        y,
        fks_incoming_sum<T>(p, single_index(incoming)),
        index,
        fks_j,
        pair_count,
        pair_m,
        pair_n,
        masses,
        eik
    );
}

// the same with the momentum k of the resonance of the resonance-local recoil
// in place of the sum of the incoming momenta where the event has one (k_0 > 0)
template <typename T>
KERNELSPEC void kernel_fks_eikonals_resonance(
    FIn<T, 2> p,
    FIn<T, 1> p_i_hat,
    FIn<T, 0> y,
    FIn<T, 1> k,
    IIn<T, 0> index,
    IIn<T, 1> fks_j,
    IIn<T, 1> pair_count,
    IIn<T, 2> pair_m,
    IIn<T, 2> pair_n,
    FIn<T, 2> masses,
    IIn<T, 0> incoming,
    FOut<T, 1> eik
) {
    fks_eikonals_in_frame<T>(
        p,
        p_i_hat,
        y,
        fks_frame_of<T>(k, fks_incoming_sum<T>(p, single_index(incoming))),
        index,
        fks_j,
        pair_count,
        pair_m,
        pair_n,
        masses,
        eik
    );
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

// The kernels with energies and energy squared s in the rest frame of p_tot,
// the sum of the incoming momenta or, for a final-state j, the momentum of the
// resonance of the resonance-local recoil (sborncol_fsr of its MadGraph7
// branch: E = (K p) / M_K and t = z M_K^2 / 4).
template <typename T>
KERNELSPEC void fks_splitting_kernels_in_frame(
    FIn<T, 2> p,
    FourMom<T> p_tot,
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
    auto s = lsquare<T>(p_tot);
    // energies in the rest frame of p_tot, up to the common 1 / sqrt(s)
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

template <typename T>
KERNELSPEC void kernel_fks_splitting_kernels(
    FIn<T, 2> p,
    IIn<T, 0> index,
    IIn<T, 1> fks_i,
    IIn<T, 1> fks_j,
    IIn<T, 1> kind,
    FIn<T, 2> factors,
    IIn<T, 0> incoming,
    FOut<T, 0> z,
    FOut<T, 1> ap,
    FOut<T, 1> q
) {
    fks_splitting_kernels_in_frame<T>(
        p,
        fks_incoming_sum<T>(p, single_index(incoming)),
        index,
        fks_i,
        fks_j,
        kind,
        factors,
        z,
        ap,
        q
    );
}

// the same with the momentum k of the resonance of the resonance-local recoil
// where the event has one (k_0 > 0)
template <typename T>
KERNELSPEC void kernel_fks_splitting_kernels_resonance(
    FIn<T, 2> p,
    FIn<T, 1> k,
    IIn<T, 0> index,
    IIn<T, 1> fks_i,
    IIn<T, 1> fks_j,
    IIn<T, 1> kind,
    FIn<T, 2> factors,
    IIn<T, 0> incoming,
    FOut<T, 0> z,
    FOut<T, 1> ap,
    FOut<T, 1> q
) {
    fks_splitting_kernels_in_frame<T>(
        p,
        fks_frame_of<T>(k, fks_incoming_sum<T>(p, single_index(incoming))),
        index,
        fks_i,
        fks_j,
        kind,
        factors,
        z,
        ap,
        q
    );
}

// The transverse direction of the emission k_perp (xij_kperp of genps_fks.f):
// the momentum of i per unit xi made orthogonal to the Born emitter and to P,
// normalized to k_perp^2 = -1. With P the sum of the incoming Born momenta it
// is (0, cos phi, sin phi, 0) in the frame of the emitter (the beam axis for
// an initial-state one) in the partonic center-of-mass frame, up to its sign.
// With P the momentum of the resonance of the resonance-local recoil it is that
// vector in the rest frame of the resonance, boosted with the momenta
// (resonance_collinear_phase of the MadGraph7 branch).
template <typename T>
KERNELSPEC void fks_transverse_vector_in_frame(
    FIn<T, 1> p_i_hat,
    FIn<T, 2> p_born,
    FourMom<T> P,
    IIn<T, 0> index,
    IIn<T, 1> born_j,
    FOut<T, 1> k_perp
) {
    std::size_t c = single_index(index);
    auto q = load_mom<T>(p_born[single_index(born_j[c])]);
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

template <typename T>
KERNELSPEC void kernel_fks_transverse_vector(
    FIn<T, 1> p_i_hat,
    FIn<T, 2> p_born,
    IIn<T, 0> index,
    IIn<T, 1> born_j,
    IIn<T, 0> incoming,
    FOut<T, 1> k_perp
) {
    fks_transverse_vector_in_frame<T>(
        p_i_hat,
        p_born,
        fks_incoming_sum<T>(p_born, single_index(incoming)),
        index,
        born_j,
        k_perp
    );
}

// the same with the momentum k of the resonance of the resonance-local recoil
// where the event has one (k_0 > 0)
template <typename T>
KERNELSPEC void kernel_fks_transverse_vector_resonance(
    FIn<T, 1> p_i_hat,
    FIn<T, 2> p_born,
    FIn<T, 1> k,
    IIn<T, 0> index,
    IIn<T, 1> born_j,
    IIn<T, 0> incoming,
    FOut<T, 1> k_perp
) {
    fks_transverse_vector_in_frame<T>(
        p_i_hat,
        p_born,
        fks_frame_of<T>(k, fks_incoming_sum<T>(p_born, single_index(incoming))),
        index,
        born_j,
        k_perp
    );
}

// The prefactors of the real emission and its counterterms of
// compute_prefactors_n1body (fks_singular.f), for the Jacobians of the
// events without the factor xi and without the symmetry factors: the
// plus distributions in xi and 1 - y with their cuts xicut, xiScut, delta and
// deltaS. The collinear ones vanish for a massive emitter. soft_cut and
// coll_cut are the cuts of the plus distribution in xi of the soft and of the
// collinear counter-events, delta_c that in 1 - y: xicut and delta, or those
// of the resonance-local recoil.
template <typename T>
KERNELSPEC void fks_prefactors_with_cuts(
    FIn<T, 0> xi,
    FIn<T, 0> y,
    FIn<T, 0> xi_max,
    FIn<T, 0> xi_c,
    FIn<T, 0> xi_max_c,
    FIn<T, 2> p_c,
    FIn<T, 2> p_sc,
    FVal<T> soft_cut,
    FVal<T> coll_cut,
    FVal<T> delta_c,
    bool collinear,
    FIn<T, 1> options,
    std::size_t n_in,
    FOut<T, 0> f_r,
    FOut<T, 0> f_s,
    FOut<T, 0> f_c,
    FOut<T, 0> f_dc,
    FOut<T, 0> f_sc,
    FOut<T, 1> f_dsc,
    FOut<T, 1> f_pdf
) {
    constexpr double pi = 3.14159265358979323846;
    FVal<T> xicut = options[0], xi_s_cut = options[1], delta_s = options[2];
    FVal<T> omy = 1. - FVal<T>(y);
    FVal<T> x = xi;
    f_r = 1. / (x * omy);
    auto xi_lim = min(FVal<T>(xi_max), xi_s_cut);
    f_s = (1. / x + log(soft_cut / xi_lim) / xi_lim) / omy;
    if (!collinear) {
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
    auto l_delta = log(delta_c / delta_s) / delta_s;
    auto in_y = 1. / omy + l_delta;
    f_c = in_y / x_c;
    auto s_c = lsquare<T>(fks_incoming_sum<T>(p_c, n_in));
    f_dc = 32. * pi * pi / (s_c * x_c * delta_s);
    auto xi_lim_c = min(FVal<T>(xi_max_c), xi_s_cut);
    auto l_xi = log(coll_cut / xi_lim_c);
    f_sc = in_y * (1. / x_c + l_xi / xi_lim_c);
    auto s_sc = lsquare<T>(fks_incoming_sum<T>(p_sc, n_in));
    auto g = 32. * pi * pi / (s_sc * delta_s);
    auto log_cut = log(xicut), log_lim = log(xi_lim_c);
    f_dsc[0] = g / x_c;
    f_dsc[1] = g * l_xi / xi_lim_c;
    // as a product, which vanishes exactly for xi_lim_c = xicut
    f_dsc[2] = g * (log_cut - log_lim) * (log_cut + log_lim) / (2. * xi_lim_c);
    f_dsc[3] = g * (1. / x_c + l_xi / xi_lim_c);
    // the delta, plus and log-plus parts of a change of the PDF scheme:
    // delta(xi) spread over the range of the soft-collinear counterterm, up to
    // xi_lim_c, and log(xi_lim_c) and log^2(xi_lim_c) / 2 that turn its
    // [1/xi] and [log(xi)/xi] into [1/(1 - z)]_+ and [log(1 - z)/(1 - z)]_+.
    // MadFKS takes xiScut for xi_lim_c, which misses part of them where the
    // range of xi ends below xiScut.
    f_pdf[0] = g / xi_lim_c;
    f_pdf[1] = g * log_lim / xi_lim_c;
    f_pdf[2] = g * log_lim * log_lim / (2. * xi_lim_c);
}

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
    IIn<T, 0> incoming,
    FOut<T, 0> f_r,
    FOut<T, 0> f_s,
    FOut<T, 0> f_c,
    FOut<T, 0> f_dc,
    FOut<T, 0> f_sc,
    FOut<T, 1> f_dsc,
    FOut<T, 1> f_pdf
) {
    std::size_t c = single_index(index);
    FVal<T> xicut = options[0];
    fks_prefactors_with_cuts<T>(
        xi,
        y,
        xi_max,
        xi_c,
        xi_max_c,
        p_c,
        p_sc,
        xicut,
        xicut,
        FVal<T>(delta[c]),
        single_index(collinear[c]) != 0,
        options,
        single_index(incoming),
        f_r,
        f_s,
        f_c,
        f_dc,
        f_sc,
        f_dsc,
        f_pdf
    );
}

// The same with the cuts of the resonance-local recoil where the event has a
// resonance of momentum k (k_0 > 0), as compute_prefactors_n1body of the
// MadGraph7 branch takes them from resonance_subtraction_scales at the soft
// counter-event (momenta p_s, i per unit xi p_i_hat_s): with the sum P of its
// incoming momenta, s = P^2, the emitter j and a(q) = s (k q) / (k^2 (P q)),
// the ratio of xi in the rest frame of k to xi in that of P for a soft
// momentum q, the soft cut xicut a(p_i_hat_s), the collinear one
// xicut a(p_j) and delta s / (k^2 a(p_j)^2), delta / D_j^2 with D_j the ratio
// of the energies of j in the two frames. The local counterterms then match
// the integrated ones of the partonic center-of-mass frame.
template <typename T>
KERNELSPEC void kernel_fks_prefactors_resonance(
    FIn<T, 0> xi,
    FIn<T, 0> y,
    FIn<T, 0> xi_max,
    FIn<T, 0> xi_c,
    FIn<T, 0> xi_max_c,
    FIn<T, 2> p_c,
    FIn<T, 2> p_sc,
    FIn<T, 2> p_s,
    FIn<T, 1> p_i_hat_s,
    FIn<T, 1> k,
    IIn<T, 0> index,
    IIn<T, 1> fks_j,
    IIn<T, 1> collinear,
    FIn<T, 1> delta,
    FIn<T, 1> options,
    IIn<T, 0> incoming,
    FOut<T, 0> f_r,
    FOut<T, 0> f_s,
    FOut<T, 0> f_c,
    FOut<T, 0> f_dc,
    FOut<T, 0> f_sc,
    FOut<T, 1> f_dsc,
    FOut<T, 1> f_pdf
) {
    std::size_t c = single_index(index);
    FVal<T> xicut = options[0], delta_c = delta[c];
    auto res = load_mom<T>(k);
    auto resonant = res[0] > 0.;
    auto total = fks_incoming_sum<T>(p_s, single_index(incoming));
    auto phat = load_mom<T>(p_i_hat_s);
    auto pj = load_mom<T>(p_s[single_index(fks_j[c])]);
    // as resonance_subtraction_scales, at a pass-through k without a resonance
    FourMom<T> q;
    for (int mu = 0; mu < 4; ++mu) {
        q[mu] = where(resonant, res[mu], total[mu]);
    }
    auto s = lsquare<T>(total), mass2 = lsquare<T>(q);
    auto kq = ldot<T>(phat, q), kt = ldot<T>(phat, total);
    auto jq = ldot<T>(pj, q), jt = ldot<T>(pj, total);
    auto soft_scale = (s / mass2) * (kq / kt);
    auto coll_scale = (s / mass2) * (jq / jt);
    auto angular_scale = (s / mass2) / (coll_scale * coll_scale);
    fks_prefactors_with_cuts<T>(
        xi,
        y,
        xi_max,
        xi_c,
        xi_max_c,
        p_c,
        p_sc,
        where(resonant, xicut * soft_scale, xicut),
        where(resonant, xicut * coll_scale, xicut),
        where(resonant, delta_c * angular_scale, delta_c),
        single_index(collinear[c]) != 0,
        options,
        single_index(incoming),
        f_r,
        f_s,
        f_c,
        f_dc,
        f_sc,
        f_dsc,
        f_pdf
    );
}

// The soft limit of xi^2 (1 - y) times the real matrix element of sbornsoft
// (fks_singular.f) per NLO order: for a gluon i (coupling 0)
// 4 pi alpha_s sum_(m, n) f_mn E_mn <T_m . T_n> B, for a photon i (coupling 1)
// 4 pi alpha sum_(m, n) f_mn E_mn B, over the pairs of the configuration,
// positions in the Born, with the Born of the order that born_index gives for
// each NLO order (none for -1). f_mn = -2 iden_comp, half that for m = n (the
// color-linked Born of a particle with itself has a factor 1/2 in MadFKS),
// times eta_m Q_m eta_n Q_n for a photon i, with the charges Q and eta = -1
// for an incoming particle. A configuration without a soft limit has no pairs.
template <typename T>
KERNELSPEC void kernel_fks_soft_limit(
    FIn<T, 1> eik,
    FIn<T, 1> born,
    FIn<T, 3> color_correlated,
    FIn<T, 0> alpha_s,
    FIn<T, 0> alpha,
    IIn<T, 0> index,
    IIn<T, 1> pair_count,
    IIn<T, 2> pair_m,
    IIn<T, 2> pair_n,
    FIn<T, 2> pair_factor,
    IIn<T, 2> coupling,
    IIn<T, 3> born_index,
    FOut<T, 1> soft
) {
    constexpr double pi = 3.14159265358979323846;
    std::size_t c = single_index(index);
    std::size_t count = single_index(pair_count[c]);
    // the soft limit has the splitting of the Born only
    bool qed = single_index(coupling[c][0]) == 1;
    FVal<T> g2 = 4. * pi * (qed ? FVal<T>(alpha) : FVal<T>(alpha_s));
    // the sum without the Born of the charge links
    FVal<T> charge_sum = 0.;
    if (qed) {
        for (std::size_t k = 0; k < count; ++k) {
            charge_sum = charge_sum + FVal<T>(pair_factor[c][k]) * FVal<T>(eik[k]);
        }
    }
    for (std::size_t o = 0; o < soft.size(); ++o) {
        int b = born_index[c][0][o];
        if (b < 0 || count == 0) {
            soft[o] = 0.;
            continue;
        }
        if (qed) {
            soft[o] = g2 * charge_sum * FVal<T>(born[b]);
            continue;
        }
        FVal<T> sum = 0.;
        for (std::size_t k = 0; k < count; ++k) {
            std::size_t m = single_index(pair_m[c][k]), n = single_index(pair_n[c][k]);
            sum = sum +
                FVal<T>(pair_factor[c][k]) * FVal<T>(eik[k]) *
                    FVal<T>(color_correlated[b][m][n]);
        }
        soft[o] = g2 * sum;
    }
}

// The collinear limit of xi^2 (1 - y) times the real matrix element of
// sborncol_fsr and sborncol_isr (fks_singular.f) per NLO order from the
// kernels of the couplings of the splitting types of the configuration (0
// QCD, 1 QED, -1 none; the second for the extra counterterm): the sum over
// them of 4 pi alpha iden_comp (AP B + Q (B / 2 - B_kperp)), with the Born
// B and the Born B_kperp whose emitter has the polarization vector k_perp
// of the Born input that born_index gives for each NLO order (none for -1)
// and alpha the strong or the QED coupling. In MadFKS the Born that Q
// multiplies is the helicity-flipped Born times a phase of the azimuth of the
// emission, which is B / 2 - B_kperp. B_kperp enters only where Q is not
// zero.
template <typename T>
KERNELSPEC void kernel_fks_collinear_limit(
    FIn<T, 1> born,
    FIn<T, 1> spin_correlated,
    FIn<T, 1> ap,
    FIn<T, 1> q,
    FIn<T, 0> alpha_s,
    FIn<T, 0> alpha,
    IIn<T, 0> index,
    FIn<T, 2> factor,
    IIn<T, 2> coupling,
    IIn<T, 3> born_index,
    FOut<T, 1> collinear
) {
    constexpr double pi = 3.14159265358979323846;
    std::size_t c = single_index(index);
    for (std::size_t o = 0; o < collinear.size(); ++o) {
        collinear[o] = 0.;
    }
    for (std::size_t entry = 0; entry < 2; ++entry) {
        int type = coupling[c][entry];
        FVal<T> f = factor[c][entry];
        if (type < 0) {
            continue;
        }
        std::size_t t = static_cast<std::size_t>(type);
        FVal<T> g2 = 4. * pi * (t == 1 ? FVal<T>(alpha) : FVal<T>(alpha_s));
        FVal<T> kernel = ap[t], azimuthal = q[t];
        for (std::size_t o = 0; o < collinear.size(); ++o) {
            int b = born_index[c][entry][o];
            if (b < 0 || f == 0.) {
                // no collinear limit (a massive emitter) or no counterterm
                continue;
            }
            FVal<T> born_b = born[b];
            FVal<T> value = kernel * born_b;
            if (azimuthal != 0.) {
                value =
                    value + azimuthal * (0.5 * born_b - FVal<T>(spin_correlated[b]));
            }
            collinear[o] = FVal<T>(collinear[o]) + g2 * f * value;
        }
    }
}

// (1 - z) P'(z) of AP_reduced_prime (the O(epsilon) part) without its color
// or charge factor, for the splittings of fks_ap_shape
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

// The plus (p), log-plus (l) and delta (d) parts of the change of the PDF
// scheme from MSbar of xkplus, xklog and xkdelta (fks_singular.f) without
// their factor, for the splittings of fks_ap_shape and the schemes of
// pdfscheme: 1 DIS, 2 eta, 3 beta, 4 mixed, 5 nobeta, 6 delta (the leptonic
// ones of eMELA), 7 UPC. The factor is
// - DIS: -2 n_f T_R for g -> g g, T_R (QCD) or N_c Q^2 (QED, N_c = 1 for a
//   lepton) for g/a -> q q~, C_F or Q^2 for q -> q g/a and minus that for
//   q -> g/a q;
// - eta to delta: C_F or Q^2 of the mother for q -> q g/a, and for delta also
//   C_F or Q^2 of i for q -> g/a q, zero for the other splittings;
// - UPC: N_c Q^2 for a -> q q~ (QED), zero for the other splittings.
// upc_log is log(mu^2 xi_A^2 R_A^2) of UPC, with the Ellis-Sexton scale mu and
// the nucleus of the beam of the emitter.
template <typename T>
KERNELSPEC void fks_scheme_shapes(
    int scheme,
    std::size_t kind,
    FVal<T> x,
    FVal<T> upc_log,
    FVal<T>& k_p,
    FVal<T>& k_l,
    FVal<T>& k_d
) {
    constexpr double pi = 3.14159265358979323846;
    auto w = 1. - x;
    auto pgq = 1. + x * x;
    k_p = 0.;
    k_l = 0.;
    k_d = 0.;
    if (scheme == 1) {
        auto lx = log(x);
        if (kind < 2) {
            auto pqg = x * x + w * w;
            k_p = w * (-pqg * lx + 8. * x * w - 1.);
            k_l = w * pqg;
        } else {
            k_p = -1.5 - pgq * lx + w * (3. + 2. * x);
            k_l = pgq;
            k_d = -(4.5 + pi * pi / 3.);
        }
    } else if (scheme >= 2 && scheme <= 6 && kind == 2) {
        // (1 - x) (1 + x) as a product, exact at x = 1
        auto w_x = w * (1. + x);
        k_l = 2. * pgq;
        switch (scheme) {
        case 2:
            k_p = -w_x;
            k_d = -3.5;
            break;
        case 3:
            k_d = -3.5;
            break;
        case 4:
            k_p = -w_x;
            k_d = -2.;
            break;
        case 5:
            k_p = 2. - w_x;
            k_d = -2.;
            break;
        default:
            k_p = pgq;
            k_d = -2.;
        }
    } else if (scheme == 6 && kind == 3) {
        k_p = w * (1. + w * w) / x * (2. * log(x) + 1.);
    } else if (scheme == 7 && kind == 1) {
        k_p = -w * (x * x + w * w) * upc_log;
    }
}

// The collinear remnants of an initial-state emitter of sreal_deg and
// compute_collinear_counter_term / compute_soft_collinear_counter_term
// (fks_singular.f) per NLO order, with the factorization scale of the beam
// of the emitter, summed over the splitting types of the configuration (the
// second for the extra counterterm) with the kernels of their couplings
// (0 QCD with alpha_s, 1 QED with alpha, -1 none): with g^2 = 4 pi alpha and
// N = B iden_comp / (8 pi^2 z), the Born input B that born_index gives for
// each NLO order (none for -1), at the collinear counter-event that of
// collinear_born (p_born_coll without the projection of the event),
//   D_xi = N g^2 ((1 - z) P log(s delta_I / (2 mu_F^2)) - (1 - z) P'),
//   D_lxi = 2 N g^2 (1 - z) P,   K_p,l,d = -N g^2 k_p,l,d (zero in MSbar),
// with the color or charge factor ap_factor of P and P' and scheme_factor of
// k_p,l,d (fks_scheme_shapes), the collinear one
// f_dc (D_xi + K_p + (D_lxi + K_l) log xi_c) at z = 1 - xi_c and s of the
// collinear counter-event, the soft-collinear one
// -f_dsc1 (D_xi + K_p + (D_lxi + K_l) log xi_c) - f_dsc2 D_xi - f_dsc3 D_lxi
// + f_pdf_d K_d + f_pdf_p K_p + f_pdf_l K_l at z = 1 and s of the
// soft-collinear counter-event. Each is zero where its sector function is
// (options = pdf_scheme, then xi_A^2 R_A^2 of the nuclei of the two beams for
// UPC, with the renormalization scale as the Ellis-Sexton scale).
template <typename T>
KERNELSPEC void kernel_fks_collinear_remnants(
    FIn<T, 1> born,
    FIn<T, 1> collinear_born,
    FIn<T, 0> alpha_s,
    FIn<T, 0> alpha,
    FIn<T, 0> s_c,
    FIn<T, 0> s_sc,
    FIn<T, 2> p_c,
    FIn<T, 2> p_sc,
    FIn<T, 0> xi_c,
    FIn<T, 0> f_dc,
    FIn<T, 1> f_dsc,
    FIn<T, 1> f_pdf,
    FIn<T, 0> ren_scale,
    FIn<T, 0> fact_scale1,
    FIn<T, 0> fact_scale2,
    IIn<T, 0> index,
    IIn<T, 1> remnant,
    IIn<T, 1> beam,
    IIn<T, 1> kind,
    IIn<T, 2> coupling,
    IIn<T, 3> born_index,
    FIn<T, 2> factor,
    FIn<T, 2> ap_factor,
    FIn<T, 2> scheme_factor,
    FIn<T, 1> delta,
    FIn<T, 1> options,
    FOut<T, 1> collinear,
    FOut<T, 1> soft_collinear
) {
    constexpr double pi = 3.14159265358979323846;
    std::size_t c = single_index(index);
    for (std::size_t o = 0; o < collinear.size(); ++o) {
        collinear[o] = 0.;
        soft_collinear[o] = 0.;
    }
    if (single_index(remnant[c]) == 0) {
        return;
    }
    std::size_t k = single_index(kind[c]);
    int scheme = static_cast<int>(FVal<T>(options[0]));
    std::size_t b = single_index(beam[c]);
    FVal<T> mu_f = b == 0 ? FVal<T>(fact_scale1) : FVal<T>(fact_scale2);
    FVal<T> upc_log = 0.;
    if (scheme == 7) {
        FVal<T> mu_r = ren_scale;
        upc_log = log(mu_r * mu_r * FVal<T>(options[1 + b]));
    }
    FVal<T> d = delta[c];
    FVal<T> norm, color, scheme_color;
    // the remnants per unit Born
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
            d_xi = n * color * (ap * log(s * d / (2. * mu_f * mu_f)) - ap_prime);
            d_lxi = 2. * n * color * ap;
            k_p = 0.;
            k_l = 0.;
            k_d = 0.;
            if (scheme_color != 0.) {
                fks_scheme_shapes<T>(scheme, k, z, upc_log, k_p, k_l, k_d);
                k_p = -n * scheme_color * k_p;
                k_l = -n * scheme_color * k_l;
                k_d = -n * scheme_color * k_d;
            }
        };
    auto invariant_mass = [](FIn<T, 2> p) {
        auto p1 = load_mom<T>(p[0]), p2 = load_mom<T>(p[1]);
        return lsquare<T>(
            FourMom<T>{p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2], p1[3] + p2[3]}
        );
    };
    FVal<T> x_c = xi_c, log_xi = log(x_c);
    for (std::size_t entry = 0; entry < 2; ++entry) {
        int type = coupling[c][entry];
        if (type < 0) {
            continue;
        }
        FVal<T> g2 = 4. * pi * (type == 1 ? FVal<T>(alpha) : FVal<T>(alpha_s));
        // N g^2 for z = 1 and a unit Born
        norm = FVal<T>(factor[c][entry]) * g2 / (8. * pi * pi);
        color = ap_factor[c][entry];
        scheme_color = scheme_factor[c][entry];
        FVal<T> d_xi, d_lxi, k_p, k_l, k_d;
        FVal<T> unit_c = 0., unit_sc = 0.;
        if (FVal<T>(s_c) > 0.) {
            pieces(1. - x_c, invariant_mass(p_c), d_xi, d_lxi, k_p, k_l, k_d);
            unit_c = FVal<T>(f_dc) * (d_xi + k_p + (d_lxi + k_l) * log_xi);
        }
        if (FVal<T>(s_sc) > 0.) {
            pieces(1., invariant_mass(p_sc), d_xi, d_lxi, k_p, k_l, k_d);
            unit_sc = -FVal<T>(f_dsc[0]) * (d_xi + k_p + (d_lxi + k_l) * log_xi) -
                FVal<T>(f_dsc[1]) * d_xi - FVal<T>(f_dsc[2]) * d_lxi +
                FVal<T>(f_pdf[0]) * k_d + FVal<T>(f_pdf[1]) * k_p +
                FVal<T>(f_pdf[2]) * k_l;
        }
        for (std::size_t o = 0; o < collinear.size(); ++o) {
            int b = born_index[c][entry][o];
            if (b < 0) {
                continue;
            }
            FVal<T> born_c = collinear_born[b], born_sc = born[b];
            collinear[o] = FVal<T>(collinear[o]) + unit_c * born_c;
            soft_collinear[o] = FVal<T>(soft_collinear[o]) + unit_sc * born_sc;
        }
    }
}

// The real matrix element per NLO order times the sector function at the
// event, with the limits of sreal and fks_Sij (fks_singular.f, fks_Sij.f)
// below tiny (options[0], zero for none): below tiny in 1 - y the collinear
// limit at the event over xi^2 (1 - y), else below tiny in xi the soft one, in
// place of the matrix element, which is then not used. The limits are zero
// without a soft (or collinear) singularity. For xi <= tiny the sector
// function has the direction of i at the soft counter-event, at the
// soft-collinear one where also 1 - y < tiny and j is massless. Without the
// counter-event (zero det) the event keeps its own direction and matrix
// element, as fks_Sij.
template <typename T>
KERNELSPEC void kernel_fks_real_emission(
    FIn<T, 1> matrix_element,
    FIn<T, 0> s,
    FIn<T, 0> s_soft,
    FIn<T, 0> s_soft_collinear,
    FIn<T, 1> soft,
    FIn<T, 1> collinear,
    FIn<T, 0> xi,
    FIn<T, 0> y,
    FIn<T, 0> det_s,
    FIn<T, 0> det_sc,
    IIn<T, 0> index,
    IIn<T, 1> massless,
    FIn<T, 1> options,
    FOut<T, 1> real
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
    for (std::size_t o = 0; o < real.size(); ++o) {
        FVal<T> value = matrix_element[o];
        if (on && omy < tiny) {
            value = FVal<T>(collinear[o]) / (x * x * omy);
        } else if (on && x < tiny && with_soft) {
            value = FVal<T>(soft[o]) / (x * x * omy);
        }
        real[o] = sector * value;
    }
}

// The real emission and its counterterms of compute_real_emission and
// compute_soft_counter_term and its siblings (fks_singular.f) per NLO order,
// from the real emission times the sector function, the limits at the
// counter-events with their sector functions and prefactors, and the
// collinear remnants: signs, symmetry factors and the conditions of each
// counterterm (options = xiScut, deltaS), zero where its momenta are not
// generated.
template <typename T>
KERNELSPEC void kernel_fks_subtraction(
    FIn<T, 1> real,
    FIn<T, 1> soft,
    FIn<T, 1> collinear,
    FIn<T, 1> soft_collinear,
    FIn<T, 0> s_s,
    FIn<T, 0> s_c,
    FIn<T, 0> s_sc,
    FIn<T, 0> f_s,
    FIn<T, 0> f_c,
    FIn<T, 0> f_sc,
    FIn<T, 1> remnant_c,
    FIn<T, 1> remnant_sc,
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
    FOut<T, 1> real_out,
    FOut<T, 1> soft_out,
    FOut<T, 1> collinear_out,
    FOut<T, 1> soft_collinear_out
) {
    std::size_t c = single_index(index);
    FVal<T> sym = symmetry[c], xi_s_cut = options[0], delta_s = options[1];
    FVal<T> x = xi, x_hat = xi_hat, x_c = xi_c, yy = y;
    bool in_y = yy > 1. - delta_s;
    bool with_real = FVal<T>(weight) > 0.;
    bool with_soft =
        FVal<T>(det_s) > 0. && x_hat * FVal<T>(xi_max_s) <= xi_s_cut && x <= xi_s_cut;
    bool with_collinear = FVal<T>(det_c) > 0. && in_y;
    bool with_soft_collinear = FVal<T>(det_sc) > 0. && in_y &&
        x_hat * FVal<T>(xi_max_c) < xi_s_cut && x_c < xi_s_cut;
    FVal<T> soft_factor = FVal<T>(s_s) * FVal<T>(f_s);
    FVal<T> collinear_factor = FVal<T>(s_c) * FVal<T>(f_c);
    FVal<T> soft_collinear_factor = FVal<T>(s_sc) * FVal<T>(f_sc);
    for (std::size_t o = 0; o < real_out.size(); ++o) {
        real_out[o] = with_real ? sym * FVal<T>(real[o]) : FVal<T>(0.);
        soft_out[o] = with_soft ? -sym * soft_factor * FVal<T>(soft[o]) : FVal<T>(0.);
        collinear_out[o] = with_collinear
            ? sym * (FVal<T>(remnant_c[o]) - collinear_factor * FVal<T>(collinear[o]))
            : FVal<T>(0.);
        soft_collinear_out[o] = with_soft_collinear
            ? sym *
                (soft_collinear_factor * FVal<T>(soft_collinear[o]) +
                 FVal<T>(remnant_sc[o]))
            : FVal<T>(0.);
    }
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
            return 0.5 * l * l + l * lr - dilog<T>(r) + 0.5 * lr * lr -
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
            dilog<T>(1. - (1. + beta) / (2. * r)) +
            dilog<T>(1. - 2. * r / (1. - beta)) + 0.5 * lr * lr + l_s * lk -
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
        return (lxy * lxy + 4. * dilog<T>(1. - (x + y) / t_v) +
                4. * dilog<T>(1. - (x - y) / t_v)) /
            (2. * t_vl);
    };
    FVal<T> arg1 = al * e_i, arg3 = e_j;
    FVal<T> xi1a =
        k * k * (1. + v) / mi2 * (j1a(arg1, arg1 * beta_i) - j1a(arg3, arg3 * beta_j));
    return 0.5 * xi0 * l + 0.5 * xi1a;
}

// The n-body part of bornsoftvirtual (fks_singular.f) without the virtual
// per NLO order, with the Ellis-Sexton scale at the renormalization scale
// mu_R, for QCD (t = 0, a = alpha_s / (2 pi)) and QED (t = 1,
// a = alpha / (2 pi)): sum_t a_t (B_t sum_j Q_tj + sum_(m, n) f_tmn L_tmn
// I_mn), with the Born B_t of the order that born_index[t] gives for each NLO
// order (none for -1), the Q terms of the massless particles j of type t and
// the pairs of linked particles of type t with the integrated eikonals I_mn of
// eikonal_Ireg, L_0mn = <T_m . T_n> B_0 and L_1mn = B_1. f_0mn = -2 and -1 for
// a massive particle with itself (the color-linked Born of MadFKS has a factor
// 1/2 there), f_1mn the same times eta_m Q_m eta_n Q_n. The Q term of a
// final-state particle j is gamma'_j - log(s delta_O / (2 mu_R^2)) (gamma_j -
// 2 C_j log(2 E_j / (xicut sqrt_s))) + 2 C_j (log^2(2 E_j / sqrt_s) - log^2
// xicut) - 2 gamma_j log(2 E_j / sqrt_s), with its energy E_j in the partonic
// center-of-mass frame (written with invariants), that of an incoming one
// -log(mu_F^2 / mu_R^2) (gamma_j + 2 C_j log xicut) with the factorization
// scale of its beam. Also the coefficients of log mu_R^2, a_0 ren_factor B_0
// with ren_factor = b gamma_g for the power b of alpha_s of the Born, and of
// log mu_F^2 of each beam, -sum_t a_t (gamma_tj + 2 C_tj log xicut) B_t, at a
// fixed virtual (options = xicut, delta_O).
template <typename T>
KERNELSPEC void kernel_fks_soft_virtual(
    FIn<T, 2> momenta,
    FIn<T, 1> born,
    FIn<T, 3> color_correlated,
    FIn<T, 0> alpha_s,
    FIn<T, 0> alpha,
    FIn<T, 0> ren_scale,
    FIn<T, 0> fact_scale1,
    FIn<T, 0> fact_scale2,
    FIn<T, 2> casimir,
    FIn<T, 2> gamma,
    FIn<T, 2> gamma_prime,
    FIn<T, 1> masses,
    IIn<T, 2> pair_m,
    IIn<T, 2> pair_n,
    FIn<T, 2> pair_factor,
    IIn<T, 2> born_index,
    FIn<T, 1> ren_factor,
    FIn<T, 1> options,
    IIn<T, 0> incoming,
    FOut<T, 1> soft_virtual,
    FOut<T, 1> ren_coefficient,
    FOut<T, 2> fact_coefficients
) {
    constexpr double pi = 3.14159265358979323846;
    FVal<T> xicut = options[0], delta_o = options[1];
    std::size_t n_in = single_index(incoming);
    auto p_tot = fks_incoming_sum<T>(momenta, n_in);
    FVal<T> s = lsquare<T>(p_tot), rs = sqrt(s);
    FVal<T> mu_r = ren_scale, mu_r2 = mu_r * mu_r;
    FVal<T> l_s = log(s / mu_r2), log_xicut = log(xicut);
    FVal<T> l_o = log(s * delta_o / (2. * mu_r2));
    for (std::size_t o = 0; o < soft_virtual.size(); ++o) {
        soft_virtual[o] = 0.;
        ren_coefficient[o] = 0.;
        fact_coefficients[o][0] = 0.;
        fact_coefficients[o][1] = 0.;
    }
    for (std::size_t t = 0; t < 2; ++t) {
        bool used = false;
        for (std::size_t o = 0; o < soft_virtual.size(); ++o) {
            used = used || int(born_index[t][o]) >= 0;
        }
        if (!used) {
            continue;
        }
        FVal<T> a = (t == 0 ? FVal<T>(alpha_s) : FVal<T>(alpha)) / (2. * pi);
        // the Q terms and the coefficients of log mu_F^2 per unit Born
        FVal<T> q = 0.;
        // (a decay has no beams, and its incoming particle no Q term)
        FVal<T> fact[2] = {0., 0.};
        for (std::size_t j = 0; j < n_in; ++j) {
            FVal<T> mu_f = j == 0 ? FVal<T>(fact_scale1) : FVal<T>(fact_scale2);
            fact[j] = -(FVal<T>(gamma[t][j]) + 2. * FVal<T>(casimir[t][j]) * log_xicut);
            q = q + fact[j] * log(mu_f * mu_f / mu_r2);
        }
        for (std::size_t j = n_in; j < momenta.size(); ++j) {
            FVal<T> c = casimir[t][j], g = gamma[t][j], g_prime = gamma_prime[t][j];
            if (c == 0. && g == 0. && g_prime == 0.) {
                continue;
            }
            auto p = load_mom<T>(momenta[j]);
            FVal<T> l_e = log(2. * fks_frame_energy<T>(p, p_tot, rs) / rs);
            q = q + g_prime - l_o * (g - 2. * c * (l_e - log_xicut)) +
                2. * c * (l_e * l_e - log_xicut * log_xicut) - 2. * g * l_e;
        }
        for (std::size_t o = 0; o < soft_virtual.size(); ++o) {
            int b = born_index[t][o];
            if (b < 0) {
                continue;
            }
            FVal<T> born_b = born[b];
            soft_virtual[o] = soft_virtual[o] + a * born_b * q;
            fact_coefficients[o][0] = fact_coefficients[o][0] + a * born_b * fact[0];
            fact_coefficients[o][1] = fact_coefficients[o][1] + a * born_b * fact[1];
            if (t == 0) {
                ren_coefficient[o] = a * born_b * FVal<T>(ren_factor[o]);
            }
        }
        // the integrated eikonals of the pairs, with the color-linked Born
        // (QCD) or the Born (QED)
        for (std::size_t k = 0; k < pair_factor[t].size(); ++k) {
            FVal<T> factor = pair_factor[t][k];
            if (factor == 0.) {
                continue;
            }
            std::size_t m = single_index(pair_m[t][k]);
            std::size_t n = single_index(pair_n[t][k]);
            auto p_m = load_mom<T>(momenta[m]), p_n = load_mom<T>(momenta[n]);
            FVal<T> e_m = fks_frame_energy<T>(p_m, p_tot, rs);
            FVal<T> e_n = fks_frame_energy<T>(p_n, p_tot, rs);
            FVal<T> i_mn = fks_integrated_eikonal<T>(
                e_m, e_n, ldot<T>(p_m, p_n), masses[m], masses[n], m == n, l_s, xicut
            );
            for (std::size_t o = 0; o < soft_virtual.size(); ++o) {
                int b = born_index[t][o];
                if (b < 0) {
                    continue;
                }
                FVal<T> link =
                    t == 0 ? FVal<T>(color_correlated[b][m][n]) : FVal<T>(born[b]);
                soft_virtual[o] = soft_virtual[o] + a * factor * link * i_mn;
            }
        }
    }
}

// The weight in pb of one contribution of the FKS integrand at its own
// kinematics: INV_GEV2_TO_PB luminosity factor det / (2 s) sum_o g_s^p_o c_o
// with g_s = sqrt(4 pi alpha_s), the power p_o of g_s of order o and
// s = (p_1 + p_2)^2, as sigint of driver_mintFO.f multiplies the weight lines
// (without g_s) by g_strong^QCDpower and the luminosity of their kinematics.
// For a decay (one incoming particle of mass M = sqrt(p_1^2)) the flux is
// 1 / (2 M) and the weight a width in GeV. Zero where det is zero, where the
// kinematics may be unset.
template <typename T>
KERNELSPEC void kernel_fks_contribution_weight(
    FIn<T, 1> coefficients,
    FIn<T, 0> det,
    FIn<T, 2> momenta,
    FIn<T, 0> alpha_s,
    FIn<T, 0> luminosity,
    FIn<T, 0> factor,
    IIn<T, 1> powers,
    IIn<T, 0> incoming,
    FOut<T, 0> weight
) {
    constexpr double pi = 3.14159265358979323846;
    bool decay = single_index(incoming) == 1;
    FVal<T> s = lsquare<T>(fks_incoming_sum<T>(momenta, single_index(incoming)));
    FVal<T> g_s = sqrt(4. * pi * FVal<T>(alpha_s));
    FVal<T> sum = 0.;
    for (std::size_t o = 0; o < coefficients.size(); ++o) {
        FVal<T> coupling = 1.;
        for (int k = 0; k < int(powers[o]); ++k) {
            coupling = coupling * g_s;
        }
        sum = sum + coupling * FVal<T>(coefficients[o]);
    }
    FVal<T> d = det;
    FVal<T> safe_s = where(d == 0., 1., s);
    FVal<T> product = FVal<T>(luminosity) * FVal<T>(factor) * d * sum;
    weight = where(
        d == 0.,
        0.,
        decay ? FVal<T>(product / (2. * sqrt(safe_s)))
              : FVal<T>(INV_GEV2_TO_PB * product / (2. * safe_s))
    );
}

} // namespace kernels
} // namespace madspace

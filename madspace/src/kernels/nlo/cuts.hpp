#pragma once

#include "../kinematics.hpp"

namespace madspace {
namespace kernels {

// The generation cuts of an NLO integrand, those of cuts.f of MadFKS
// (passcuts_leptons and passcuts_jets of passcuts_user), on the momenta of one
// kinematics in the lab frame, and the jets of the clustering they share with
// the histograms. The particles are in the order of the flavor layout of the
// event. The kernels vectorize: what depends on the flavor layout runs once
// per layout among the lanes, what depends on the momenta selects with where,
// and the lists of each lane (the pseudojets of the clustering, the particles
// around a photon) are slots with masks.

// most final-state QCD partons that a kinematics clusters into jets
// (NLOCuts::jet_parton_max, NLOJets::parton_max)
constexpr std::size_t NLO_JET_PARTONS_MAX = 16;

// p where mask, else q
template <typename T>
KERNELSPEC FourMom<T>
nlo_where(BVal<T> mask, const FourMom<T>& p, const FourMom<T>& q) {
    return {
        where(mask, p[0], q[0]),
        where(mask, p[1], q[1]),
        where(mask, p[2], q[2]),
        where(mask, p[3], q[3])
    };
}

// pseudorapidity as eta of kin_functions.f, +-25 within 1e-5 of the beam axis
template <typename T>
KERNELSPEC FVal<T> nlo_cut_eta(const FourMom<T>& p) {
    constexpr double pi = 3.14159265358979323846;
    FVal<T> theta = acos(p[3] / sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3]));
    FVal<T> eta = -log(tan(theta / 2.));
    return where(
        fabs(theta) < 1e-5,
        FVal<T>(25.),
        where(fabs(theta - pi) < 1e-5, FVal<T>(-25.), eta)
    );
}

// Delta phi as DELTA_PHI of kin_functions.f, from the cosine clamped to
// +-0.99999999
template <typename T>
KERNELSPEC FVal<T> nlo_cut_delta_phi(const FourMom<T>& p1, const FourMom<T>& p2) {
    FVal<T> denom =
        sqrt(p1[1] * p1[1] + p1[2] * p1[2]) * sqrt(p2[1] * p2[1] + p2[2] * p2[2]);
    FVal<T> cosine = (p1[1] * p2[1] + p1[2] * p2[2]) / denom;
    return acos(min(max(cosine, -0.99999999), 0.99999999));
}

// pseudorapidity of a jet as PseudoJet::pseudorapidity of fastjet
template <typename T>
KERNELSPEC FVal<T> nlo_jet_eta(const FourMom<T>& p) {
    constexpr double pi = 3.14159265358979323846;
    FVal<T> pt = sqrt(p[1] * p[1] + p[2] * p[2]);
    FVal<T> theta = atan(pt / p[3]);
    theta = where(theta < 0., theta + pi, theta);
    return where(p[3] == 0., FVal<T>(0.), FVal<T>(-log(tan(theta / 2.))));
}

// The rapidity and azimuth of a parton for the clustering, as PseudoJet of
// fastjet: phi in [0, 2 pi).
template <typename T>
KERNELSPEC void nlo_jet_rap_phi(const FourMom<T>& p, FVal<T>& rapidity, FVal<T>& phi) {
    constexpr double pi = 3.14159265358979323846;
    rapidity = 0.5 * log((p[0] + p[3]) / (p[0] - p[3]));
    phi = atan2(p[2], p[1]);
    phi = where(phi < 0., phi + 2. * pi, phi);
}

// The jets of partons (count of them, each where member) with the pp
// generalized kT algorithm of fastjet (power jetalgo, radius jetradius,
// rapidity and azimuth, E scheme), those with pt >= ptj and |eta| <= etaj
// (etaj <= 0 for none), in jets in the order the clustering finds them, and
// their number. Partons without transverse momentum make no pseudojet (with
// ptj > 0 they never make a jet). The pseudojets of each lane take the first
// slots, at most NLO_JET_PARTONS_MAX of them, and every step takes the
// closest distance of the slots in use, makes a jet or merges the pair into
// the first of the two, and moves the last slot into the one that empties.
template <typename T>
KERNELSPEC IVal<T> nlo_cluster_jets(
    const FourMom<T>* partons,
    const BVal<T>* member,
    std::size_t count,
    double jetalgo,
    double jetradius,
    double ptj,
    double etaj,
    FourMom<T>* jets
) {
    constexpr double pi = 3.14159265358979323846;
    constexpr std::size_t max_slots = NLO_JET_PARTONS_MAX;
    std::size_t slots = count < max_slots ? count : max_slots;
    FourMom<T> zero{0., 0., 0., 0.};
    FourMom<T> pseudo[max_slots];
    FVal<T> kt2[max_slots], rap[max_slots], phi[max_slots];
    for (std::size_t s = 0; s < max_slots; ++s) {
        pseudo[s] = zero;
        kt2[s] = 0.;
        rap[s] = 0.;
        phi[s] = 0.;
        jets[s] = zero;
    }
    IVal<T> active = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const auto& q = partons[i];
        FVal<T> k2 = q[1] * q[1] + q[2] * q[2];
        BVal<T> keep =
            member[i] & (k2 > 0.) & (active < IVal<T>(static_cast<int>(slots)));
        FVal<T> r, f;
        nlo_jet_rap_phi<T>(q, r, f);
        // the next free slot, one of the first i + 1
        for (std::size_t s = 0; s < slots && s <= i; ++s) {
            BVal<T> here = keep & (active == IVal<T>(static_cast<int>(s)));
            pseudo[s] = nlo_where<T>(here, q, pseudo[s]);
            kt2[s] = where(here, k2, kt2[s]);
            rap[s] = where(here, r, rap[s]);
            phi[s] = where(here, f, phi[s]);
        }
        active = active + where(keep, IVal<T>(1), IVal<T>(0));
    }
    double r2_jet = jetradius * jetradius;
    IVal<T> jet_count = 0;
    for (std::size_t step = 0; step < slots; ++step) {
        BVal<T> running = active > IVal<T>(0);
        // the smallest of the beam distances and the pair distances
        FVal<T> d_beam[max_slots];
        for (std::size_t s = 0; s < slots; ++s) {
            d_beam[s] = pow(kt2[s], jetalgo);
        }
        FVal<T> best = d_beam[0];
        IVal<T> best_i = 0, best_j = 0;
        for (std::size_t i = 0; i < slots; ++i) {
            IVal<T> index_i = static_cast<int>(i);
            BVal<T> better = (index_i < active) & (d_beam[i] < best);
            best = where(better, d_beam[i], best);
            best_i = where(better, index_i, best_i);
            best_j = where(better, index_i, best_j);
            for (std::size_t j = i + 1; j < slots; ++j) {
                IVal<T> index_j = static_cast<int>(j);
                FVal<T> dphi = fabs(phi[i] - phi[j]);
                dphi = where(dphi > pi, 2. * pi - dphi, dphi);
                FVal<T> drap = rap[i] - rap[j];
                FVal<T> d_ij =
                    min(d_beam[i], d_beam[j]) * (drap * drap + dphi * dphi) / r2_jet;
                better = (index_j < active) & (d_ij < best);
                best = where(better, d_ij, best);
                best_i = where(better, index_i, best_i);
                best_j = where(better, index_j, best_j);
            }
        }
        FourMom<T> p_i = zero, p_j = zero;
        FVal<T> kt2_i = 0.;
        for (std::size_t s = 0; s < slots; ++s) {
            IVal<T> index = static_cast<int>(s);
            p_i = nlo_where<T>(best_i == index, pseudo[s], p_i);
            kt2_i = where(best_i == index, kt2[s], kt2_i);
            p_j = nlo_where<T>(best_j == index, pseudo[s], p_j);
        }
        // a jet: pt and |eta| of the selection
        BVal<T> beam = best_i == best_j;
        BVal<T> jet = running & beam & (kt2_i >= ptj * ptj);
        if (etaj > 0.) {
            jet = jet & (fabs(nlo_jet_eta<T>(p_i)) <= etaj);
        }
        for (std::size_t s = 0; s < slots; ++s) {
            BVal<T> here = jet & (jet_count == IVal<T>(static_cast<int>(s)));
            jets[s] = nlo_where<T>(here, p_i, jets[s]);
        }
        jet_count = jet_count + where(jet, IVal<T>(1), IVal<T>(0));
        // E-scheme recombination of the pair into best_i
        BVal<T> merge = running & !beam;
        FourMom<T> sum{
            p_i[0] + p_j[0], p_i[1] + p_j[1], p_i[2] + p_j[2], p_i[3] + p_j[3]
        };
        FVal<T> kt2_sum = sum[1] * sum[1] + sum[2] * sum[2], rap_sum, phi_sum;
        nlo_jet_rap_phi<T>(sum, rap_sum, phi_sum);
        for (std::size_t s = 0; s < slots; ++s) {
            BVal<T> here = merge & (best_i == IVal<T>(static_cast<int>(s)));
            pseudo[s] = nlo_where<T>(here, sum, pseudo[s]);
            kt2[s] = where(here, kt2_sum, kt2[s]);
            rap[s] = where(here, rap_sum, rap[s]);
            phi[s] = where(here, phi_sum, phi[s]);
        }
        // drop best_j, the jet or the pseudojet merged into best_i: the last
        // slot in use moves there
        IVal<T> last = active - IVal<T>(1);
        FourMom<T> p_last = zero;
        FVal<T> kt2_last = 0., rap_last = 0., phi_last = 0.;
        for (std::size_t s = 0; s < slots; ++s) {
            BVal<T> here = last == IVal<T>(static_cast<int>(s));
            p_last = nlo_where<T>(here, pseudo[s], p_last);
            kt2_last = where(here, kt2[s], kt2_last);
            rap_last = where(here, rap[s], rap_last);
            phi_last = where(here, phi[s], phi_last);
        }
        for (std::size_t s = 0; s < slots; ++s) {
            BVal<T> here = running & (best_j == IVal<T>(static_cast<int>(s)));
            pseudo[s] = nlo_where<T>(here, p_last, pseudo[s]);
            kt2[s] = where(here, kt2_last, kt2[s]);
            rap[s] = where(here, rap_last, rap[s]);
            phi[s] = where(here, phi_last, phi[s]);
        }
        active = active - where(running, IVal<T>(1), IVal<T>(0));
    }
    return jet_count;
}

// most particles of a kinematics that the photon cuts and the recombination
// take (NLOCuts::particle_max)
constexpr std::size_t NLO_CUT_PARTICLES_MAX = 32;

// pseudorapidity as iso_getpseudorap of cuts.f
template <typename T>
KERNELSPEC FVal<T> nlo_iso_eta(const FourMom<T>& p) {
    FVal<T> pt = sqrt(p[1] * p[1] + p[2] * p[2]);
    FVal<T> eta = -log(tan(atan2(pt, p[3]) / 2.));
    return where(
        (pt < 1e-5) & (fabs(p[3]) < 1e-5),
        where(p[3] < 0., FVal<T>(-1e8), FVal<T>(1e8)),
        eta
    );
}

// Delta phi as iso_getdelphi of cuts.f, 1e8 where a transverse momentum
// vanishes (cuts.f stops beyond a cosine of 1 + 1e-5, here clamped)
template <typename T>
KERNELSPEC FVal<T> nlo_iso_delta_phi(const FourMom<T>& p1, const FourMom<T>& p2) {
    FVal<T> pt1 = sqrt(p1[1] * p1[1] + p1[2] * p1[2]);
    FVal<T> pt2 = sqrt(p2[1] * p2[1] + p2[2] * p2[2]);
    FVal<T> cosine = (p1[1] * p2[1] + p1[2] * p2[2]) / (pt1 * pt2);
    return where(
        (pt1 == 0.) | (pt2 == 0.),
        FVal<T>(1e8),
        acos(min(max(cosine, FVal<T>(-1.)), FVal<T>(1.)))
    );
}

// Delta R of the photon isolation (iso_getdrv40 of cuts.f), rounded to single
// precision as its drlist
template <typename T>
KERNELSPEC FVal<T> nlo_iso_dr(const FourMom<T>& p1, const FourMom<T>& p2) {
    FVal<T> deta = nlo_iso_eta<T>(p1) - nlo_iso_eta<T>(p2);
    FVal<T> dphi = nlo_iso_delta_phi<T>(p1, p2);
    return round_to_float32(sqrt(dphi * dphi + deta * deta));
}

// Whether a photon of transverse momentum ptg passes the isolation of
// Frixione (hep-ph/9801442, eq. 3.4, chi_gamma_iso of cuts.f) against the
// particles others (count of them, skip the photon itself): for every
// distance dr <= r0 the summed transverse momenta within dr stay at or below
// epsgamma ptg ((1 - cos dr) / (1 - cos r0))^|xn|. The particles go by
// ascending distance, those beyond r0 last, in a stable sort as the insertion
// of cuts.f.
template <typename T>
KERNELSPEC BVal<T> nlo_photon_isolated(
    const FourMom<T>& photon,
    const FourMom<T>* others,
    std::size_t count,
    std::size_t skip,
    FVal<T> ptg,
    double r0,
    double xn,
    double epsgamma
) {
    FVal<T> dr[NLO_CUT_PARTICLES_MAX], pt[NLO_CUT_PARTICLES_MAX];
    std::size_t size = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (i == skip) {
            continue;
        }
        FVal<T> d = nlo_iso_dr<T>(photon, others[i]);
        dr[size] = where(d <= r0, d, FVal<T>(1e300));
        pt[size] = sqrt(others[i][1] * others[i][1] + others[i][2] * others[i][2]);
        ++size;
    }
    for (std::size_t end = size; end > 1; --end) {
        for (std::size_t k = 0; k + 1 < end; ++k) {
            BVal<T> swap = dr[k] > dr[k + 1];
            FVal<T> dr_k = dr[k], pt_k = pt[k];
            dr[k] = where(swap, dr[k + 1], dr_k);
            dr[k + 1] = where(swap, dr_k, dr[k + 1]);
            pt[k] = where(swap, pt[k + 1], pt_k);
            pt[k + 1] = where(swap, pt_k, pt[k + 1]);
        }
    }
    BVal<T> isolated = true;
    FVal<T> sum = 0.;
    double axn = fabs(xn), norm = 1. - cos(r0);
    for (std::size_t k = 0; k < size; ++k) {
        BVal<T> inside = dr[k] <= r0;
        sum = sum + where(inside, pt[k], FVal<T>(0.));
        FVal<T> chi = epsgamma * ptg;
        if (axn != 0.) {
            FVal<T> d = where(inside, dr[k], FVal<T>(0.));
            chi = chi * pow((1. - cos(d)) / norm, axn);
        }
        isolated = isolated & !(inside & (sum > chi));
    }
    return isolated;
}

template <typename T>
KERNELSPEC bool nlo_is_lepton(int pdg) {
    int a = pdg < 0 ? -pdg : pdg;
    return a == 11 || a == 13 || a == 15;
}

// The generation cuts of cuts.f of MadFKS on one kinematics (passcuts_user
// without the cuts per PDG code): 1 where it passes, else 0. pdg holds the
// PDG codes of each layout and index the layout of the event.
//
// With ptgmin > 0 (options[12]) the photons, unless gamma_is_j (options[18]),
// pass pt >= ptgmin and |eta| <= etagamma (options[13], none below 0) and
// are isolated as Frixione (radius r0gamma, xn, epsgamma, options[14] to
// [16]) from the QCD partons and, with isoem (options[17]), from the other
// photons and the charged leptons, and at least photons[c] of them must be
// isolated (all with -1), as passcuts_photons. Then a photon that is not
// isolated and within etaphreco of the beam axis (options[20], none below 0)
// joins the closest charged lepton (with lepphreco, options[21]) or quark
// (with quarkphreco, options[22], flavors up to 5) within rphreco
// (options[19]) and goes, as recombine_momenta (where a second such photon
// stops cuts.f, it fails here). The charged leptons that the recombination
// leaves pass ptl and etal, each pair of opposite charge drll and mll, and
// drll_sf and mll_sf for the same flavor, with the pseudorapidity and Delta
// phi of kin_functions.f. With ptj > 0 the QCD partons (quarks up to
// maxjetflavor and gluons) and the photons that are not isolated (all with
// gamma_is_j) are clustered with the pp generalized kT algorithm of fastjet
// (power jetalgo, radius jetradius, rapidity and azimuth, E scheme) into jets
// with pt >= ptj and |eta| <= etaj (etaj <= 0 for none), and with more of
// them than the unresolved ones u (options[11]), at most u may be soft (E
// below 1e-8 of the first incoming energy) and all but at most u must make
// jets, as passcuts_jets for u = 1 on the real emission. Kinematics with a
// first incoming energy that is not positive or with momenta that are not
// finite fail.
template <typename T>
KERNELSPEC void kernel_nlo_cuts(
    FIn<T, 2> p,
    IIn<T, 0> index,
    ISIn<T, 2> pdg,
    ISIn<T, 1> photons,
    FSIn<T, 1> options,
    ISIn<T, 0> incoming,
    FOut<T, 0> mask
) {
    std::size_t n_in = single_index(incoming);
    double jetalgo = single_value(options[0]), jetradius = single_value(options[1]);
    double ptj = single_value(options[2]), etaj = single_value(options[3]);
    double ptl = single_value(options[5]), etal = single_value(options[6]);
    double drll = single_value(options[7]), drll_sf = single_value(options[8]);
    double mll = single_value(options[9]), mll_sf = single_value(options[10]);
    int maxjetflavor = static_cast<int>(single_value(options[4]));
    auto unresolved = static_cast<std::size_t>(single_value(options[11]));
    double ptgmin = single_value(options[12]), etagamma = single_value(options[13]);
    double r0gamma = single_value(options[14]), xn = single_value(options[15]);
    double epsgamma = single_value(options[16]);
    bool isoem = single_value(options[17]) != 0.;
    bool gamma_is_j = single_value(options[18]) != 0.;
    double rphreco = single_value(options[19]), etaphreco = single_value(options[20]);
    bool lepphreco = single_value(options[21]) != 0.;
    bool quarkphreco = single_value(options[22]) != 0.;
    std::size_t n = p.size();
    if (n > NLO_CUT_PARTICLES_MAX) {
        mask = 0.;
        return;
    }
    // a positive first incoming energy and finite momenta
    FVal<T> e1 = p[0][0];
    BVal<T> valid = e1 > 0.;
    FourMom<T> momenta[NLO_CUT_PARTICLES_MAX];
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < 4; ++k) {
            valid = valid & (fabs(FVal<T>(p[i][k])) <= 1e32);
        }
        momenta[i] = load_mom<T>(p[i]);
    }
    FourMom<T> zero{0., 0., 0., 0.};
    for_each_value(IVal<T>(index), [&](std::size_t c, BVal<T> lanes) {
        auto pdgs = pdg[c];
        // the lanes that pass the cuts so far
        BVal<T> pass = valid;
        FourMom<T> mom[NLO_CUT_PARTICLES_MAX];
        int ids[NLO_CUT_PARTICLES_MAX];
        // isolated photons, and photons recombined with a fermion
        BVal<T> iso[NLO_CUT_PARTICLES_MAX], gone[NLO_CUT_PARTICLES_MAX];
        for (std::size_t i = 0; i < n; ++i) {
            mom[i] = momenta[i];
            ids[i] = single_index(pdgs[i]);
            iso[i] = false;
            gone[i] = false;
        }

        // photons, isolated from the bare QCD partons and the electromagnetic
        // particles
        if (ptgmin != 0.) {
            FourMom<T> partons[NLO_CUT_PARTICLES_MAX], em[NLO_CUT_PARTICLES_MAX];
            std::size_t parton_count = 0, photon_count = 0, em_count = 0;
            std::size_t em_of[NLO_CUT_PARTICLES_MAX];
            for (std::size_t i = n_in; i < n; ++i) {
                int flavor = ids[i] < 0 ? -ids[i] : ids[i];
                if ((flavor >= 1 && flavor <= maxjetflavor) || flavor == 21) {
                    partons[parton_count++] = mom[i];
                }
                if (ids[i] == 22 && !gamma_is_j) {
                    em_of[i] = em_count;
                    em[em_count++] = mom[i];
                    ++photon_count;
                }
            }
            for (std::size_t i = n_in; i < n && isoem; ++i) {
                if (nlo_is_lepton<T>(ids[i])) {
                    em[em_count++] = mom[i];
                }
            }
            if (photon_count > 0) {
                IVal<T> isolated = 0;
                for (std::size_t i = n_in; i < n; ++i) {
                    if (ids[i] != 22 || gamma_is_j) {
                        continue;
                    }
                    const auto& g = mom[i];
                    FVal<T> ptg = sqrt(g[1] * g[1] + g[2] * g[2]);
                    BVal<T> ok = !(ptg < ptgmin);
                    if (etagamma > 0.) {
                        ok = ok & !(fabs(nlo_cut_eta<T>(g)) > etagamma);
                    }
                    ok =
                        ok &
                        nlo_photon_isolated<T>(
                            g,
                            partons,
                            parton_count,
                            parton_count,
                            ptg,
                            r0gamma,
                            xn,
                            epsgamma
                        );
                    if (isoem && em_count > 1) {
                        ok = ok &
                            nlo_photon_isolated<T>(
                                 g, em, em_count, em_of[i], ptg, r0gamma, xn, epsgamma
                            );
                    }
                    iso[i] = ok;
                    isolated = isolated + where(ok, IVal<T>(1), IVal<T>(0));
                }
                int needed = static_cast<int>(single_index(photons[c]));
                int required = needed < 0 ? static_cast<int>(photon_count) : needed;
                pass = pass & !(isolated < IVal<T>(required));
            }
        }

        // recombination of a photon that is not isolated with the closest
        // light charged fermion
        if (lepphreco || quarkphreco) {
            IVal<T> photon = static_cast<int>(n), candidates = 0;
            for (std::size_t i = n_in; i < n; ++i) {
                if (ids[i] != 22) {
                    continue;
                }
                const auto& g = mom[i];
                BVal<T> candidate = (sqrt(g[1] * g[1] + g[2] * g[2]) != 0.) & !iso[i];
                if (etaphreco >= 0.) {
                    candidate = candidate & (fabs(nlo_cut_eta<T>(g)) < etaphreco);
                }
                photon = where(candidate, IVal<T>(static_cast<int>(i)), photon);
                candidates = candidates + where(candidate, IVal<T>(1), IVal<T>(0));
            }
            pass = pass & !(candidates > IVal<T>(1));
            FourMom<T> g = zero;
            for (std::size_t i = n_in; i < n; ++i) {
                if (ids[i] == 22) {
                    g = nlo_where<T>(photon == IVal<T>(static_cast<int>(i)), mom[i], g);
                }
            }
            FVal<T> eta_g = nlo_cut_eta<T>(g);
            IVal<T> closest = static_cast<int>(n);
            FVal<T> distance = rphreco;
            for (std::size_t i = n_in; i < n; ++i) {
                int flavor = ids[i] < 0 ? -ids[i] : ids[i];
                bool fermion = (quarkphreco && flavor >= 1 && flavor <= 5) ||
                    (lepphreco && nlo_is_lepton<T>(ids[i]));
                if (!fermion) {
                    continue;
                }
                FVal<T> deta = eta_g - nlo_cut_eta<T>(mom[i]);
                FVal<T> dphi = nlo_cut_delta_phi<T>(g, mom[i]);
                FVal<T> d = sqrt(dphi * dphi + deta * deta);
                BVal<T> closer = d <= distance;
                distance = where(closer, d, distance);
                closest = where(closer, IVal<T>(static_cast<int>(i)), closest);
            }
            BVal<T> recombine =
                (candidates == IVal<T>(1)) & (closest < IVal<T>(static_cast<int>(n)));
            for (std::size_t i = n_in; i < n; ++i) {
                IVal<T> index_i = static_cast<int>(i);
                FourMom<T> sum{
                    mom[i][0] + g[0],
                    mom[i][1] + g[1],
                    mom[i][2] + g[2],
                    mom[i][3] + g[3]
                };
                mom[i] = nlo_where<T>(recombine & (closest == index_i), sum, mom[i]);
                if (ids[i] == 22) {
                    gone[i] = recombine & (photon == index_i);
                }
            }
        }

        // charged leptons
        for (std::size_t i = n_in; i < n; ++i) {
            int pdg_i = ids[i];
            if (!nlo_is_lepton<T>(pdg_i)) {
                continue;
            }
            const auto& p_i = mom[i];
            FVal<T> pt_i = sqrt(p_i[1] * p_i[1] + p_i[2] * p_i[2]);
            if (ptl > 0.) {
                pass = pass & !(pt_i < ptl);
            }
            if (etal > 0.) {
                pass = pass & !(fabs(nlo_cut_eta<T>(p_i)) > etal);
            }
            if (pdg_i > 0) {
                continue;
            }
            // a positive lepton with each negative one
            for (std::size_t j = n_in; j < n; ++j) {
                int pdg_j = ids[j];
                if (pdg_j != 11 && pdg_j != 13 && pdg_j != 15) {
                    continue;
                }
                const auto& p_j = mom[j];
                FVal<T> deta = nlo_cut_eta<T>(p_i) - nlo_cut_eta<T>(p_j);
                FVal<T> dphi = nlo_cut_delta_phi<T>(p_i, p_j);
                FVal<T> r2 = dphi * dphi + deta * deta;
                FourMom<T> pair{
                    p_i[0] + p_j[0], p_i[1] + p_j[1], p_i[2] + p_j[2], p_i[3] + p_j[3]
                };
                FVal<T> m2 = lsquare<T>(pair);
                if (drll > 0.) {
                    pass = pass & !(r2 < drll * drll);
                }
                if (mll > 0.) {
                    pass = pass & !(m2 < mll * mll);
                }
                if (pdg_i == -pdg_j) {
                    if (drll_sf > 0.) {
                        pass = pass & !(r2 < drll_sf * drll_sf);
                    }
                    if (mll_sf > 0.) {
                        pass = pass & !(m2 < mll_sf * mll_sf);
                    }
                }
            }
        }

        // jets, of the QCD partons and the photons that are not isolated
        if (ptj > 0.) {
            FourMom<T> partons[NLO_CUT_PARTICLES_MAX];
            BVal<T> member[NLO_CUT_PARTICLES_MAX];
            std::size_t candidate_count = 0;
            IVal<T> parton_count = 0, soft_count = 0;
            for (std::size_t i = n_in; i < n; ++i) {
                int flavor = ids[i] < 0 ? -ids[i] : ids[i];
                bool qcd = (flavor >= 1 && flavor <= maxjetflavor) || flavor == 21;
                if (!qcd && flavor != 22) {
                    continue;
                }
                BVal<T> jet = qcd || gamma_is_j ? BVal<T>(true) : BVal<T>(!iso[i]);
                jet = jet & !gone[i];
                const auto& p_i = mom[i];
                BVal<T> soft = jet & (fabs(p_i[0] / e1) < 1e-8);
                soft_count = soft_count + where(soft, IVal<T>(1), IVal<T>(0));
                parton_count = parton_count + where(jet, IVal<T>(1), IVal<T>(0));
                partons[candidate_count] = p_i;
                member[candidate_count] = jet;
                ++candidate_count;
            }
            IVal<T> u = static_cast<int>(unresolved);
            IVal<T> most = static_cast<int>(NLO_JET_PARTONS_MAX);
            BVal<T> clustered = parton_count > u;
            pass = pass & !(clustered & ((soft_count > u) | (parton_count > most)));
            if (candidate_count > unresolved) {
                FourMom<T> jets[NLO_JET_PARTONS_MAX];
                IVal<T> jet_count = nlo_cluster_jets<T>(
                    partons,
                    member,
                    candidate_count,
                    jetalgo,
                    jetradius,
                    ptj,
                    etaj,
                    jets
                );
                pass = pass & !(clustered & (jet_count + u < parton_count));
            }
        }
        masked_store(mask, lanes, where(pass, FVal<T>(1.), FVal<T>(0.)));
    });
}

} // namespace kernels
} // namespace madspace

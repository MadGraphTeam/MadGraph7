#pragma once

#include "kinematics.hpp"

namespace madspace {
namespace kernels {

// The generation cuts of the FKS integrand, those of cuts.f of MadFKS
// (passcuts_leptons and passcuts_jets of passcuts_user), on the momenta of the
// event or of a counter-event in the lab frame. The particles keep their
// order in the real process of the configuration of the event.

// most final-state QCD partons that a kinematics clusters into jets
// (FKSCuts::jet_parton_max)
constexpr std::size_t FKS_JET_PARTONS_MAX = 16;

// pseudorapidity as eta of kin_functions.f, +-25 within 1e-5 of the beam axis
template <typename T>
KERNELSPEC FVal<T> fks_cut_eta(const FourMom<T>& p) {
    constexpr double pi = 3.14159265358979323846;
    FVal<T> theta = acos(p[3] / sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3]));
    if (fabs(theta) < 1e-5) {
        return 25.;
    }
    if (fabs(theta - pi) < 1e-5) {
        return -25.;
    }
    return -log(tan(theta / 2.));
}

// Delta phi as DELTA_PHI of kin_functions.f, from the cosine clamped to
// +-0.99999999
template <typename T>
KERNELSPEC FVal<T> fks_cut_delta_phi(const FourMom<T>& p1, const FourMom<T>& p2) {
    FVal<T> denom =
        sqrt(p1[1] * p1[1] + p1[2] * p1[2]) * sqrt(p2[1] * p2[1] + p2[2] * p2[2]);
    FVal<T> cosine = (p1[1] * p2[1] + p1[2] * p2[2]) / denom;
    return acos(min(max(cosine, -0.99999999), 0.99999999));
}

// pseudorapidity of a jet as PseudoJet::pseudorapidity of fastjet
template <typename T>
KERNELSPEC FVal<T> fks_jet_eta(const FourMom<T>& p) {
    constexpr double pi = 3.14159265358979323846;
    FVal<T> pt = sqrt(p[1] * p[1] + p[2] * p[2]);
    if (p[3] == 0.) {
        return 0.;
    }
    FVal<T> theta = atan(pt / p[3]);
    if (theta < 0.) {
        theta = theta + pi;
    }
    return -log(tan(theta / 2.));
}

// The rapidity and azimuth of a parton for the clustering, as PseudoJet of
// fastjet: phi in [0, 2 pi).
template <typename T>
KERNELSPEC void fks_jet_rap_phi(const FourMom<T>& p, FVal<T>& rapidity, FVal<T>& phi) {
    constexpr double pi = 3.14159265358979323846;
    rapidity = 0.5 * log((p[0] + p[3]) / (p[0] - p[3]));
    phi = atan2(p[2], p[1]);
    if (phi < 0.) {
        phi = phi + 2. * pi;
    }
}

// The generation cuts of cuts.f of MadFKS on one kinematics: 1 where it
// passes, else 0. The leptons (charged, of either sign) pass ptl and etal,
// and each pair of opposite charge drll and mll, and drll_sf and mll_sf for
// the same flavor, with the pseudorapidity and Delta phi of kin_functions.f. The QCD
// partons (quarks up to maxjetflavor and gluons) are clustered with the pp generalized
// kT algorithm of fastjet (power jetalgo, radius jetradius, rapidity and azimuth, E
// scheme) into jets with pt >= ptj and |eta| <= etaj (etaj <= 0 for none). With ptj > 0
// and more than one parton, at most one parton may be soft (E below 1e-8 of the first
// incoming energy) and all partons, or all but one, must make jets, as
// passcuts_jets. Kinematics with a first incoming energy that is not positive
// or with momenta that are not finite fail.
template <typename T>
KERNELSPEC void kernel_fks_cuts(
    FIn<T, 2> p,
    IIn<T, 0> index,
    IIn<T, 2> pdg,
    FIn<T, 1> options,
    IIn<T, 0> incoming,
    FOut<T, 0> mask
) {
    std::size_t c = single_index(index);
    std::size_t n_in = single_index(incoming);
    auto pdgs = pdg[c];
    FVal<T> jetalgo = options[0], jetradius = options[1], ptj = options[2],
            etaj = options[3], ptl = options[5], etal = options[6], drll = options[7],
            drll_sf = options[8], mll = options[9], mll_sf = options[10];
    int maxjetflavor = static_cast<int>(FVal<T>(options[4]));
    std::size_t n = p.size();
    mask = 0.;
    FVal<T> e1 = p[0][0];
    if (!(e1 > 0.)) {
        return;
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < 4; ++k) {
            FVal<T> v = p[i][k];
            if (!(fabs(v) <= 1e32)) {
                return;
            }
        }
    }

    // leptons
    for (std::size_t i = n_in; i < n; ++i) {
        int pdg_i = single_index(pdgs[i]);
        int lepton_i = pdg_i < 0 ? -pdg_i : pdg_i;
        if (lepton_i != 11 && lepton_i != 13 && lepton_i != 15) {
            continue;
        }
        auto p_i = load_mom<T>(p[i]);
        FVal<T> pt_i = sqrt(p_i[1] * p_i[1] + p_i[2] * p_i[2]);
        if (ptl > 0. && pt_i < ptl) {
            return;
        }
        if (etal > 0. && fabs(fks_cut_eta<T>(p_i)) > etal) {
            return;
        }
        if (pdg_i > 0) {
            continue;
        }
        // a positive lepton with each negative one
        for (std::size_t j = n_in; j < n; ++j) {
            int pdg_j = single_index(pdgs[j]);
            if (pdg_j != 11 && pdg_j != 13 && pdg_j != 15) {
                continue;
            }
            auto p_j = load_mom<T>(p[j]);
            FVal<T> deta = fks_cut_eta<T>(p_i) - fks_cut_eta<T>(p_j);
            FVal<T> dphi = fks_cut_delta_phi<T>(p_i, p_j);
            FVal<T> r2 = dphi * dphi + deta * deta;
            FourMom<T> pair{
                p_i[0] + p_j[0], p_i[1] + p_j[1], p_i[2] + p_j[2], p_i[3] + p_j[3]
            };
            FVal<T> m2 = lsquare<T>(pair);
            if (drll > 0. && r2 < drll * drll) {
                return;
            }
            if (mll > 0. && m2 < mll * mll) {
                return;
            }
            if (pdg_i == -pdg_j) {
                if (drll_sf > 0. && r2 < drll_sf * drll_sf) {
                    return;
                }
                if (mll_sf > 0. && m2 < mll_sf * mll_sf) {
                    return;
                }
            }
        }
    }

    // jets
    FourMom<T> partons[FKS_JET_PARTONS_MAX];
    std::size_t parton_count = 0, soft_count = 0;
    for (std::size_t i = n_in; i < n; ++i) {
        int pdg_i = single_index(pdgs[i]);
        int flavor = pdg_i < 0 ? -pdg_i : pdg_i;
        if (!((flavor >= 1 && flavor <= maxjetflavor) || flavor == 21)) {
            continue;
        }
        auto p_i = load_mom<T>(p[i]);
        if (fabs(p_i[0] / e1) < 1e-8) {
            ++soft_count;
        }
        if (parton_count < FKS_JET_PARTONS_MAX) {
            partons[parton_count] = p_i;
        }
        ++parton_count;
    }
    if (ptj > 0. && parton_count > 1) {
        if (soft_count > 1 || parton_count > FKS_JET_PARTONS_MAX) {
            return;
        }
        // the pseudojets, without those of zero pt, which never make a jet
        // above ptj
        FourMom<T> jets[FKS_JET_PARTONS_MAX];
        FVal<T> kt2[FKS_JET_PARTONS_MAX], rap[FKS_JET_PARTONS_MAX],
            phi[FKS_JET_PARTONS_MAX];
        std::size_t active = 0;
        for (std::size_t i = 0; i < parton_count; ++i) {
            FVal<T> k2 = partons[i][1] * partons[i][1] + partons[i][2] * partons[i][2];
            if (!(k2 > 0.)) {
                continue;
            }
            jets[active] = partons[i];
            kt2[active] = k2;
            fks_jet_rap_phi<T>(partons[i], rap[active], phi[active]);
            ++active;
        }
        constexpr double pi = 3.14159265358979323846;
        FVal<T> r2_jet = jetradius * jetradius;
        std::size_t jet_count = 0;
        while (active > 0) {
            // the smallest of the beam distances and the pair distances
            std::size_t best_i = 0, best_j = 0;
            FVal<T> best = pow(kt2[0], jetalgo);
            for (std::size_t i = 0; i < active; ++i) {
                FVal<T> d_ib = pow(kt2[i], jetalgo);
                if (d_ib < best) {
                    best = d_ib;
                    best_i = i;
                    best_j = i;
                }
                for (std::size_t j = i + 1; j < active; ++j) {
                    FVal<T> dphi = fabs(phi[i] - phi[j]);
                    if (dphi > pi) {
                        dphi = 2. * pi - dphi;
                    }
                    FVal<T> drap = rap[i] - rap[j];
                    FVal<T> d_ij = min(d_ib, pow(kt2[j], jetalgo)) *
                        (drap * drap + dphi * dphi) / r2_jet;
                    if (d_ij < best) {
                        best = d_ij;
                        best_i = i;
                        best_j = j;
                    }
                }
            }
            if (best_i == best_j) {
                // a jet: pt and |eta| of the selection
                auto jet = jets[best_i];
                FVal<T> jet_pt2 = kt2[best_i];
                if (jet_pt2 >= ptj * ptj &&
                    (etaj <= 0. || fabs(fks_jet_eta<T>(jet)) <= etaj)) {
                    ++jet_count;
                }
            } else {
                // E-scheme recombination of the pair into best_i
                for (std::size_t k = 0; k < 4; ++k) {
                    jets[best_i][k] = jets[best_i][k] + jets[best_j][k];
                }
                kt2[best_i] = jets[best_i][1] * jets[best_i][1] +
                    jets[best_i][2] * jets[best_i][2];
                fks_jet_rap_phi<T>(jets[best_i], rap[best_i], phi[best_i]);
            }
            // drop best_j: the jet, or the pseudojet merged into best_i
            std::size_t drop = best_j;
            --active;
            jets[drop] = jets[active];
            kt2[drop] = kt2[active];
            rap[drop] = rap[active];
            phi[drop] = phi[active];
        }
        if (jet_count != parton_count && jet_count + 1 != parton_count) {
            return;
        }
    }
    mask = 1.;
}

} // namespace kernels
} // namespace madspace

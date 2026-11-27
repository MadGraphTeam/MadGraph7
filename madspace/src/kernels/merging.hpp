#pragma once

#include "definitions.hpp"
#include "kinematics.hpp"

namespace madspace {
namespace kernels {

// Merging scale of an event with nothing to measure: above any cut, and float32-safe.
constexpr double MERGING_SCALE_NONE = 1e30;

// Bits of beam_flags: which beam is a hadron (madevent's lpp != 0).
constexpr int MERGING_BEAM1_HADRONIC = 1;
constexpr int MERGING_BEAM2_HADRONIC = 2;

// Values of the mode input, set by MergingScale from the legs alone.
constexpr int MERGING_MODE_OFF = 0;
// ptlund only: exactly two massless outgoing legs, cut on their pT
constexpr int MERGING_MODE_TWO_LEGS = 1;
constexpr int MERGING_MODE_FULL = 2;

// Port of the DURHAM_KT CUT block of Template/LO/SubProcesses/cuts.f: smallest pT
// (hadronic) and pair kt min(pT) sqrt(dy^2 + dphi^2) / D, with the true rapidity.
template <typename T>
KERNELSPEC void kernel_merging_scale_kt_durham(
    FIn<T, 2> momenta,
    IIn<T, 1> merging_legs,
    FIn<T, 0> d_parameter,
    IIn<T, 0> beam_flags,
    IIn<T, 0> mode,
    FOut<T, 0> scale
) {
    FVal<T> scale_min = MERGING_SCALE_NONE;
    int mode_val = mode, flags = beam_flags;
    FVal<T> d = d_parameter;
    if (mode_val == MERGING_MODE_OFF) {
        scale = scale_min;
        return;
    }
    bool hadronic = (flags & (MERGING_BEAM1_HADRONIC | MERGING_BEAM2_HADRONIC)) != 0;
    std::size_t n = momenta.size();
    for (std::size_t i = 2; i < n; ++i) {
        if (int(merging_legs[i]) == 0) {
            continue;
        }
        auto p_i = load_mom<T>(momenta[i]);
        FVal<T> pt2_i = p_i[1] * p_i[1] + p_i[2] * p_i[2];
        if (hadronic) {
            scale_min = min(scale_min, sqrt(pt2_i));
        }
        for (std::size_t j = i + 1; j < n; ++j) {
            if (int(merging_legs[j]) == 0) {
                continue;
            }
            auto p_j = load_mom<T>(momenta[j]);
            FVal<T> kt;
            if (!hadronic) {
                // 2 min(E^2) (1 - cos theta) as 4 min(E^2) sin^2(theta/2), stable
                // for collinear pairs
                auto cross_x = p_i[2] * p_j[3] - p_i[3] * p_j[2];
                auto cross_y = p_i[3] * p_j[1] - p_i[1] * p_j[3];
                auto cross_z = p_i[1] * p_j[2] - p_i[2] * p_j[1];
                auto theta = atan2(
                    sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z),
                    edot<T>(p_i, p_j)
                );
                auto sin_half = sin(0.5 * theta);
                kt = 2. * sqrt(min(p_i[0] * p_i[0], p_j[0] * p_j[0])) * fabs(sin_half);
            } else {
                FVal<T> pt2_j = p_j[1] * p_j[1] + p_j[2] * p_j[2];
                if (pt2_i == 0. || pt2_j == 0.) {
                    kt = 0.;
                } else {
                    auto y_i = 0.5 * log((p_i[0] + p_i[3]) / (p_i[0] - p_i[3]));
                    auto y_j = 0.5 * log((p_j[0] + p_j[3]) / (p_j[0] - p_j[3]));
                    auto dy = y_i - y_j;
                    auto dphi = atan2(
                        p_i[1] * p_j[2] - p_i[2] * p_j[1],
                        p_i[1] * p_j[1] + p_i[2] * p_j[2]
                    );
                    kt = sqrt(min(pt2_i, pt2_j) * (dy * dy + dphi * dphi)) / d;
                }
            }
            scale_min = min(scale_min, kt);
        }
    }
    scale = scale_min;
}

// Pythia8 evolution pT of rad + emt with recoiler rec (types: 1 final, -1 initial).
// Port of RHOPYTHIA in Template/LO/SubProcesses/cuts.f, with its fixed masses.
template <typename T>
KERNELSPEC FVal<T> rho_pythia(
    const FourMom<T>& p_rad,
    const FourMom<T>& p_emt,
    const FourMom<T>& p_rec,
    int flav_rad,
    int flav_emt,
    int rad_type,
    int rec_type
) {
    constexpr double m0c = 1.5, m0b = 4.7, m0t = 172.5, m0x = 400.0;
    int abs_rad = flav_rad < 0 ? -flav_rad : flav_rad;
    int abs_emt = flav_emt < 0 ? -flav_emt : flav_emt;

    double emt_sign = rad_type == -1 ? -1. : 1.;
    FourMom<T> q;
    for (int k = 0; k < 4; ++k) {
        q[k] = p_rad[k] + emt_sign * p_emt[k];
    }
    FVal<T> q2 = emt_sign * lsquare<T>(q);
    bool allowed = !(q2 < 0.);

    // flavour of the radiator before the emission
    int id_rad_bef = 0;
    if (abs_emt == 21 || abs_emt == 22) {
        id_rad_bef = flav_rad;
    }
    if (rad_type == 1 && flav_emt == -flav_rad) {
        id_rad_bef = 21;
    }
    if (rad_type == 1 && abs_emt < 10 && flav_rad == 21) {
        id_rad_bef = flav_emt;
    }
    if (rad_type == -1 && abs_emt < 10 && flav_rad == 21) {
        id_rad_bef = -flav_emt;
    }
    if (rad_type == -1 && abs_emt < 10 && flav_rad == flav_emt) {
        id_rad_bef = 21;
    }
    if (flav_emt == 24) {
        id_rad_bef = flav_rad + 1;
    }
    if (flav_emt == -24) {
        id_rad_bef = flav_rad - 1;
    }

    FVal<T> m2_rad_aft = lsquare<T>(p_rad);
    FVal<T> m2_emt_aft = lsquare<T>(p_emt);
    FVal<T> m2_rec = lsquare<T>(p_rec);
    if (m2_rad_aft < 1e-4) {
        m2_rad_aft = 0.;
    }
    if (m2_emt_aft < 1e-4) {
        m2_emt_aft = 0.;
    }
    if (m2_rec < 1e-4) {
        m2_rec = 0.;
    }

    FVal<T> m2_rad_bef = 0.;
    if (abs_rad != 21 && abs_rad != 22 && abs_emt != 24 && abs_rad != abs_emt) {
        m2_rad_bef = m2_rad_aft;
    } else if (abs_emt == 24) {
        int abs_bef = id_rad_bef < 0 ? -id_rad_bef : id_rad_bef;
        if (abs_bef == 4) {
            m2_rad_bef = m0c * m0c;
        }
        if (abs_bef == 5) {
            m2_rad_bef = m0b * m0b;
        }
        if (abs_bef == 6) {
            m2_rad_bef = m0t * m0t;
        }
        if (abs_bef == 9000001) {
            m2_rad_bef = m0x * m0x;
        }
    } else if (rad_type == -1) {
        if (abs_rad == 21 && abs_emt == 21) {
            m2_rad_bef = m2_emt_aft;
        }
    }

    FourMom<T> p_final;
    for (int k = 0; k < 4; ++k) {
        p_final[k] = p_rad[k] + p_rec[k] + p_emt[k];
    }
    FVal<T> m2_final = lsquare<T>(p_final);
    if (rad_type == 1 && m2_final < 0.) {
        allowed = false;
    }

    // final-initial splittings rescale the recoiler
    FVal<T> rescale = 1.;
    FourMom<T> p_rec_aft = p_rec;
    if (rad_type == 1 && rec_type == -1) {
        FVal<T> m2_ar = m2_final - 2. * q2 + 2. * m2_rad_bef;
        FVal<T> ratio = (q2 - m2_rad_bef) / (m2_ar - m2_rad_bef);
        rescale = (1. - ratio) / (1. + ratio);
        for (int k = 0; k < 4; ++k) {
            p_rec_aft[k] = p_rec_aft[k] * rescale;
        }
    }
    if (rescale < 0.) {
        allowed = false;
    }

    FourMom<T> sum;
    for (int k = 0; k < 4; ++k) {
        sum[k] = p_rad[k] + p_rec_aft[k] + p_emt[k];
    }
    FVal<T> m2_dip = lsquare<T>(sum);
    FVal<T> x1 = 2. * ldot<T>(sum, p_rad) / m2_dip;
    FVal<T> x2 = 2. * ldot<T>(sum, p_rec_aft) / m2_dip;
    if (rad_type == 1 && (x1 < 0. || x1 > 1. || x2 < 0. || x2 > 1.)) {
        allowed = false;
    }

    // massive FSR
    FVal<T> lambda13 = sqrt(max(
        (q2 - m2_rad_aft - m2_emt_aft) * (q2 - m2_rad_aft - m2_emt_aft) -
            4. * m2_rad_aft * m2_emt_aft,
        0.
    ));
    FVal<T> k1 = (q2 - lambda13 + (m2_emt_aft - m2_rad_aft)) / (2. * q2);
    FVal<T> k3 = (q2 - lambda13 - (m2_emt_aft - m2_rad_aft)) / (2. * q2);

    FVal<T> z;
    if (rad_type == -1) {
        FourMom<T> q_br, q_ar;
        for (int k = 0; k < 4; ++k) {
            q_br[k] = p_rad[k] + p_rec[k] - p_emt[k];
            q_ar[k] = p_rad[k] + p_rec[k];
        }
        z = lsquare<T>(q_br) / lsquare<T>(q_ar);
    } else {
        z = 1. / (1. - k1 - k3) * (x1 / (2. - x2) - k3);
    }
    if (z < 0. || z > 1.) {
        allowed = false;
    }

    FVal<T> temp = rad_type == 1 ? z * (1. - z) * (q2 - m2_rad_bef) : (1. - z) * q2;

    // ISR heavy-quark threshold, with the cuts.f comparison of pT to 2 m^2
    if (rad_type != 1) {
        if ((abs_rad == 4 || abs_emt == 4) && temp >= 0. &&
            sqrt(temp) <= 2. * m0c * m0c) {
            temp = (1. - z) * (q2 + m0c * m0c);
        }
        if ((abs_rad == 5 || abs_emt == 5) && temp >= 0. &&
            sqrt(temp) <= 2. * m0b * m0b) {
            temp = (1. - z) * (q2 + m0b * m0b);
        }
    }

    if (!allowed) {
        temp = 1e15;
    }
    if (temp < 0.) {
        temp = 0.;
    }
    return sqrt(temp);
}

// Port of the PTLUND CUT block of Template/LO/SubProcesses/cuts.f: smallest
// Pythia8 evolution pT over the ISR and FSR splittings of the merging legs.
template <typename T>
KERNELSPEC void kernel_merging_scale_pt_lund(
    FIn<T, 2> momenta,
    IIn<T, 1> merging_legs,
    IIn<T, 1> pdg_ids,
    IIn<T, 0> beam_flags,
    IIn<T, 0> mode,
    FOut<T, 0> scale
) {
    FVal<T> scale_min = MERGING_SCALE_NONE;
    int mode_val = mode, flags = beam_flags;
    bool hadronic1 = (flags & MERGING_BEAM1_HADRONIC) != 0;
    bool hadronic2 = (flags & MERGING_BEAM2_HADRONIC) != 0;
    bool hadronic = hadronic1 || hadronic2;
    std::size_t n = momenta.size();

    if (mode_val == MERGING_MODE_TWO_LEGS) {
        // no energy-sharing variable: only the pT to the beam
        if (hadronic) {
            for (std::size_t i = 2; i < n; ++i) {
                if (int(merging_legs[i]) == 0) {
                    continue;
                }
                auto p_i = load_mom<T>(momenta[i]);
                scale_min = min(scale_min, sqrt(p_i[1] * p_i[1] + p_i[2] * p_i[2]));
            }
        }
        scale = scale_min;
        return;
    }
    if (mode_val != MERGING_MODE_FULL) {
        scale = scale_min;
        return;
    }

    auto p_inc1 = load_mom<T>(momenta[0]);
    auto p_inc2 = load_mom<T>(momenta[1]);
    int flav_inc1 = pdg_ids[0];
    int flav_inc2 = pdg_ids[1];
    for (std::size_t i = 2; i < n; ++i) {
        if (int(merging_legs[i]) == 0) {
            continue;
        }
        auto p_i = load_mom<T>(momenta[i]);
        int flav_i = pdg_ids[i];
        int abs_i = flav_i < 0 ? -flav_i : flav_i;

        // ISR: only SM-like emissions off the beams
        if (hadronic && abs_i < 30) {
            scale_min = min(
                scale_min, rho_pythia<T>(p_inc1, p_i, p_inc2, flav_inc1, flav_i, -1, -1)
            );
            scale_min = min(
                scale_min, rho_pythia<T>(p_inc2, p_i, p_inc1, flav_inc2, flav_i, -1, -1)
            );
        }

        // FSR between merging legs, with a merging leg as spectator
        for (std::size_t j = 2; j < n; ++j) {
            if (j == i || int(merging_legs[j]) == 0) {
                continue;
            }
            auto p_j = load_mom<T>(momenta[j]);
            int flav_j = pdg_ids[j];
            int abs_j = flav_j < 0 ? -flav_j : flav_j;
            for (std::size_t k = 2; k < n; ++k) {
                if (k == i || k == j || int(merging_legs[k]) == 0) {
                    continue;
                }
                auto p_k = load_mom<T>(momenta[k]);
                if (abs_i < 30) {
                    scale_min = min(
                        scale_min, rho_pythia<T>(p_j, p_i, p_k, flav_j, flav_i, 1, 1)
                    );
                }
                if (abs_j < 30) {
                    scale_min = min(
                        scale_min, rho_pythia<T>(p_i, p_j, p_k, flav_i, flav_j, 1, 1)
                    );
                }
            }
        }

        // FSR between merging legs with a beam spectator; BSM particles only radiate
        if (hadronic && abs_i <= 1000000) {
            for (std::size_t j = 2; j < n; ++j) {
                if (j == i || int(merging_legs[j]) == 0) {
                    continue;
                }
                auto p_j = load_mom<T>(momenta[j]);
                int flav_j = pdg_ids[j];
                if (hadronic1) {
                    scale_min =
                        min(scale_min,
                            rho_pythia<T>(p_j, p_i, p_inc1, flav_j, flav_i, 1, -1));
                }
                if (hadronic2) {
                    scale_min =
                        min(scale_min,
                            rho_pythia<T>(p_j, p_i, p_inc2, flav_j, flav_i, 1, -1));
                }
            }
        }
    }
    scale = scale_min;
}

} // namespace kernels
} // namespace madspace

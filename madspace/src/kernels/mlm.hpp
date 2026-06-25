#pragma once

#include "definitions.hpp"
#include "kinematics.hpp"

#include <cmath>

namespace madspace {
namespace kernels {

constexpr int N_EXT_MAX = 12;

// --- Low-level kinematic helpers (mirror boost<T> and rotate_inverse<T> from
// kinematics.hpp
//     but operating on plain double[4] so they can be used with the local scratch
//     arrays) ---

// Lorentz boost of k by the frame defined by p_boost (sign=+1 forward, -1 inverse).
// Mirrors boost<T> from kinematics.hpp.
KERNELSPEC void
boost4(const double k[4], const double p_boost[4], double sign, double out[4]) {
    double p2 = p_boost[0] * p_boost[0] - p_boost[1] * p_boost[1] -
        p_boost[2] * p_boost[2] - p_boost[3] * p_boost[3];
    double rsq = sqrt(p2 > EPS2 ? p2 : EPS2);
    double k_dot_p = k[1] * p_boost[1] + k[2] * p_boost[2] + k[3] * p_boost[3];
    double e = (k[0] * p_boost[0] + sign * k_dot_p) / rsq;
    double c1 = sign * (k[0] + e) / (rsq + p_boost[0]);
    out[0] = e;
    out[1] = k[1] + c1 * p_boost[1];
    out[2] = k[2] + c1 * p_boost[2];
    out[3] = k[3] + c1 * p_boost[3];
}

// Rotate p into the canonical frame where q's spatial direction is the z-axis.
// Mirrors rotate_inverse<T> from kinematics.hpp.
KERNELSPEC void rotate_inv4(const double p[4], const double q[4], double out[4]) {
    double qt2 = q[1] * q[1] + q[2] * q[2];
    double qq2 = qt2 + q[3] * q[3];
    double qt = sqrt(qt2 > EPS2 ? qt2 : EPS2);
    double qq = sqrt(qq2 > EPS2 ? qq2 : EPS2);
    out[0] = p[0];
    if (qt2 < EPS2) {
        double s = (q[3] < 0.0) ? -1.0 : 1.0;
        out[1] = s * p[1];
        out[2] = p[2];
        out[3] = s * p[3];
        return;
    }
    out[1] = q[1] * q[3] / (qq * qt) * p[1] + q[2] * q[3] / (qq * qt) * p[2] -
        p[3] * qt / qq;
    out[2] = -q[2] / qt * p[1] + q[1] / qt * p[2];
    out[3] = q[1] / qq * p[1] + q[2] / qq * p[2] + q[3] / qq * p[3];
}

// --- Mass tracking ---

// Update momenta and separately-tracked masses after one clustering step.
//
// i_remove : index of the final-state particle absorbed into the combination (always >=
// 2) j_keep   : index of the particle kept with the combined momentum (0/1 = initial
// state, else final) is_bw    : Breit-Wigner resonance — combined mass = invariant
// mass; else max of daughters alive    : alive[i] is true while particle i has not yet
// been removed n_part   : total number of particles (bound for the alive[] loop)
//
// Mass update rules from Fortran update_momenta:
//   Final-state non-BW  : mass[j_keep] = max(mass[j_keep], mass[i_remove])
//   Final-state BW      : mass[j_keep] = sqrt((p_i + p_j)^2)
//   Initial-state (XOR) : mass[j_keep] = max(mass[j_keep], mass[i_remove])   (exactly
//   one massive) Initial-state (same): mass[j_keep] = 0 (both massive or both massless)
//
// Momentum update rules:
//   Final-state  : momenta[j_keep] += momenta[i_remove]
//   Initial-state: momenta[j_keep] -= momenta[i_remove]; then if CM s > 100 GeV^2,
//                  boost all alive particles to the new CM frame and rotate so that
//                  the new j_keep momentum is along z.
KERNELSPEC void update_momenta(
    int n_part,
    double momenta[N_EXT_MAX][4],
    double masses[N_EXT_MAX],
    bool alive[N_EXT_MAX],
    int i_remove,
    int j_keep,
    bool is_bw
) {
    alive[i_remove] = false;

    if (j_keep < 2) {
        // initial-state clustering
        int j_other = 1 - j_keep; // the other beam particle (0-indexed)
        for (int k = 0; k < 4; ++k) {
            momenta[j_keep][k] -= momenta[i_remove][k];
        }

        // mass: max if exactly one daughter is massive, else 0
        bool m0 = (masses[j_keep] > 0.0);
        bool m1 = (masses[i_remove] > 0.0);
        masses[j_keep] = (m0 != m1) ? fmax(masses[j_keep], masses[i_remove]) : 0.0;

        // CM boost vector: (E_tot, -px_tot, -py_tot, -pz_tot) of the two beam particles
        double pcmsp[4];
        pcmsp[0] = momenta[j_keep][0] + momenta[j_other][0];
        pcmsp[1] = -(momenta[j_keep][1] + momenta[j_other][1]);
        pcmsp[2] = -(momenta[j_keep][2] + momenta[j_other][2]);
        pcmsp[3] = -(momenta[j_keep][3] + momenta[j_other][3]);

        double inv_sq = pcmsp[0] * pcmsp[0] - pcmsp[1] * pcmsp[1] -
            pcmsp[2] * pcmsp[2] - pcmsp[3] * pcmsp[3];
        if (inv_sq > 100.0) {
            // boost j_keep to CM frame to define the rotation axis
            double jkeep_cm[4];
            boost4(momenta[j_keep], pcmsp, 1.0, jkeep_cm);

            // boost all alive particles and rotate so that j_keep is along z
            for (int j = 0; j < n_part; ++j) {
                if (!alive[j]) {
                    continue;
                }
                double p_cm[4], p_rot[4];
                boost4(momenta[j], pcmsp, 1.0, p_cm);
                rotate_inv4(p_cm, jkeep_cm, p_rot);
                for (int k = 0; k < 4; ++k) {
                    momenta[j][k] = p_rot[k];
                }
            }
        }
        // if inv_sq <= 100: no boost needed, momenta already correct after subtraction

    } else {
        // final-state clustering: sum momenta, update mass
        for (int k = 0; k < 4; ++k) {
            momenta[j_keep][k] += momenta[i_remove][k];
        }

        if (is_bw) {
            const double* p = momenta[j_keep];
            double m2 = p[0] * p[0] - p[1] * p[1] - p[2] * p[2] - p[3] * p[3];
            masses[j_keep] = sqrt(m2 > 0.0 ? m2 : 0.0);
        } else {
            masses[j_keep] = fmax(masses[j_keep], masses[i_remove]);
        }
    }
}

// --- Clustering scale helpers ---

KERNELSPEC double minkowski_dot(const double* p1, const double* p2) {
    return p1[0] * p2[0] - p1[1] * p2[1] - p1[2] * p2[2] - p1[3] * p2[3];
}

// mT^2 = E^2 - pz^2 for one parton (hadronic), or E^2 (lepton collider).
// Replaces Fortran DJB_clus from cluster.f:2201.
KERNELSPEC double djb_clus(const double* p, bool hadronic) {
    double r = hadronic ? (p[0] - p[3]) * (p[0] + p[3]) : p[0] * p[0];
    return r < 0.0 ? 0.0 : r;
}

// kt/Durham clustering measure for two partons.
// mass1/mass2 are the tracked clustering masses from masses[] (set by update_momenta),
// NOT the Lorentz-invariant masses computed from the 4-momentum.
// D is the jet-radius parameter from Fortran common /to_dj/D.
// Replaces Fortran dj_clus from cluster.f:2144.
KERNELSPEC double dj_clus(
    const double* p1,
    const double* p2,
    double mass1,
    double mass2,
    bool hadronic,
    double D
) {
    constexpr double one_plus_tiny = 1.0 + 1e-6;
    if (!hadronic) {
        // Durham e+e- measure: 2*min(E1^2,E2^2)*(1-cos_theta)
        double p1a = sqrt(p1[1] * p1[1] + p1[2] * p1[2] + p1[3] * p1[3]);
        double p2a = sqrt(p2[1] * p2[1] + p2[2] * p2[2] + p2[3] * p2[3]);
        if (p1a * p2a == 0.0) {
            return 0.0;
        }
        double costh = (p1[1] * p2[1] + p1[2] * p2[2] + p1[3] * p2[3]) / (p1a * p2a);
        return 2.0 * fmin(p1[0] * p1[0], p2[0] * p2[0]) * fmax(1.0 - costh, 0.0);
    }
    // hadronic: massless+massive pair clusters to the lighter parton's mT^2
    bool massive1 = (mass1 > 0.0);
    bool massive2 = (mass2 > 0.0);
    if (!massive1 && massive2) {
        return djb_clus(p1, true) * one_plus_tiny;
    }
    if (massive1 && !massive2) {
        return djb_clus(p2, true) * one_plus_tiny;
    }
    // both massless or both massive: generalised kt measure in (eta, phi)
    double pt1_sq = p1[1] * p1[1] + p1[2] * p1[2];
    double pt2_sq = p2[1] * p2[1] + p2[2] * p2[2];
    if (pt1_sq == 0.0 || pt2_sq == 0.0) {
        return 0.0;
    }
    double p1a = sqrt(pt1_sq + p1[3] * p1[3]);
    double p2a = sqrt(pt2_sq + p2[3] * p2[3]);
    double eta1 = 0.5 * log((p1a + p1[3]) / (p1a - p1[3]));
    double eta2 = 0.5 * log((p2a + p2[3]) / (p2a - p2[3]));
    double m_max_sq = fmax(mass1 * mass1, mass2 * mass2);
    double dphi_cos = (p1[1] * p2[1] + p1[2] * p2[2]) / sqrt(pt1_sq * pt2_sq);
    double r = m_max_sq +
        fmin(pt1_sq, pt2_sq) * 2.0 * (cosh(eta1 - eta2) - dphi_cos) / (D * D);
    return r < 0.0 ? 0.0 : r;
}

// Clustering scale for the pair (momentum1=pi, momentum2=pj).
// mass1/mass2 are the tracked clustering masses for momentum1/momentum2.
//
// Parameters replacing Fortran globals:
//   is_initial : momentum2 is a beam particle (Fortran: j<=2 in cluster_one_step)
//   hadronic   : hadronic collider (Fortran: lpp[] from run.inc)
//   D          : jet-radius parameter (Fortran: common /to_dj/D)
//
// massive_in/out1/out2 replace the Fortran get_clustering_type cl[0:2] bit-array:
//   massive_in   = cl[0] has bit 2 or 4  (intermediate/mother)
//   massive_out1 = cl[1] has bit 2 or 4  (momentum1 = final-state particle pi)
//   massive_out2 = cl[2] has bit 2 or 4  (momentum2 = pj)
//
// resonant replaces the iBWlist lookup in Fortran cluster_scale.
// mass_in/width_in are carried for future use; not consumed in the scale formula.
//
// Replaces Fortran cluster_scale from cluster.f:1159.
KERNELSPEC double compute_scale(
    const double* momentum1,
    const double* momentum2,
    double mass1,
    double mass2,
    double mass_in,
    double width_in,
    bool resonant,
    bool is_initial,
    bool massive_in,
    bool massive_out1,
    bool massive_out2,
    bool hadronic,
    double D
) {
    constexpr double one_plus_tiny = 1.000001;

    if (is_initial) {
        // scale = mT of the final-state parton; small penalty when it goes against the
        // beam
        double scale = sqrt(djb_clus(momentum1, hadronic));
        if ((momentum1[3] < 0.0) != (momentum2[3] < 0.0)) {
            scale *= one_plus_tiny;
        }
        return scale;
    }

    double sum[4] = {
        momentum1[0] + momentum2[0],
        momentum1[1] + momentum2[1],
        momentum1[2] + momentum2[2],
        momentum1[3] + momentum2[3]
    };

    if (resonant) {
        return sqrt(fmax(minkowski_dot(sum, sum), 0.0));
    }

    // Map massive_in/out1/out2 booleans to Fortran get_clustering_type itypes:
    //   type 4: massless mother -> massive out1 + massless out2
    if (!massive_in && massive_out1 && !massive_out2) {
        return sqrt(fabs(minkowski_dot(momentum2, sum))) / 2.0;
    }
    //   type 5: massless mother -> massless out1 + massive out2
    if (!massive_in && !massive_out1 && massive_out2) {
        return sqrt(fabs(minkowski_dot(momentum1, sum))) / 2.0;
    }
    //   type 8: massive mother -> massless out1 + massless out2
    if (massive_in && !massive_out1 && !massive_out2) {
        return sqrt(fmax(minkowski_dot(sum, sum), 0.0));
    }
    // types 1,2,3,6,7: all-massless, massive-emitting-massless, all-massive
    return sqrt(dj_clus(momentum1, momentum2, mass1, mass2, hadronic, D));
}

template <typename T>
KERNELSPEC void mlm_clustering_hadronic(
    FIn<T, 2> momenta,
    FIn<T, 0> random,
    IIn<T, 1> state_machine,
    FIn<T, 1> masses,
    FIn<T, 1> widths,
    FIn<T, 0> D,
    FOut<T, 0> ren_scale,
    FOut<T, 0> fact_scale1,
    FOut<T, 0> fact_scale2,
    FOut<T, 1> outgoing_scales,
    IOut<T, 0> diagram_index
) {
    static_assert(std::is_same_v<IVal<T>, int>);
    static_assert(std::is_same_v<FVal<T>, double>);
    int state = 0, cluster_count = 0;
    int cluster_max = momenta.size() - 3;
    int n_part = momenta.size();
    double momenta_tmp[N_EXT_MAX][4];
    double masses_tmp[N_EXT_MAX];
    bool alive[N_EXT_MAX];
    int cluster_history[N_EXT_MAX - 3];
    double cluster_scales[N_EXT_MAX - 3];

    for (int i = 0; i < n_part; ++i) {
        for (int j = 0; j < 4; ++j) {
            momenta_tmp[i][j] = momenta[i][j];
        }
        masses_tmp[i] = masses[i];
        alive[i] = true;
    }

    int win_next_state = -1, win_data = 0;
    double win_scale = 1e308;
    while (cluster_count < cluster_max) {
        int data = state_machine[state];
        int next_state = state_machine[state + 1];
        int particle1 = data & 0xFF;
        int particle2 = (data >> 8) & 0xFF;
        int mass_index = (data >> 16) & 0xFF;
        bool massive_in = (data >> 24) & 1;
        bool massive_out1 = (data >> 25) & 1;
        bool massive_out2 = (data >> 26) & 1;
        bool is_last = (data >> 28) & 1;
        bool is_initial = (particle2 < 2);
        bool resonant =
            (mass_index != 0) && (static_cast<double>(widths[mass_index]) > 0.0);

        double scale = compute_scale(
            momenta_tmp[particle1],
            momenta_tmp[particle2],
            masses_tmp[particle1],
            masses_tmp[particle2],
            static_cast<double>(masses[mass_index]),
            static_cast<double>(widths[mass_index]),
            resonant,
            is_initial,
            massive_in,
            massive_out1,
            massive_out2,
            true, // hadronic
            static_cast<double>(D)
        );

        if (scale < win_scale) {
            win_next_state = next_state;
            win_scale = scale;
            win_data = data;
        }
        if (is_last) {
            int p1_win = win_data & 0xFF;
            int p2_win = (win_data >> 8) & 0xFF;
            bool win_resonant = ((win_data >> 16) & 0xFF) != 0 &&
                (static_cast<double>(widths[(win_data >> 16) & 0xFF]) > 0.0);
            update_momenta(
                n_part, momenta_tmp, masses_tmp, alive, p1_win, p2_win, win_resonant
            );
            state = win_next_state;
            cluster_history[cluster_count] = win_data;
            cluster_scales[cluster_count] = win_scale;
            ++cluster_count;
            win_scale = 1e308;
        } else {
            ++state;
        }
    }

    // Renormalization scale: geometric mean of QCD clustering scales
    // (non-QCD entries replaced by the max scale).
    // Factorization scale: smallest QCD clustering scale.
    double fac_scale = 1e308, max_scale = 0.0;
    for (int i = 0; i < cluster_max; ++i) {
        double scale = cluster_scales[i];
        bool is_qcd = (cluster_history[i] >> 27) & 1;
        if (is_qcd && scale < fac_scale) {
            fac_scale = scale;
        }
        if (scale > max_scale) {
            max_scale = scale;
        }
    }

    double ren_scale_val = 1.0;
    for (int i = 0; i < cluster_max; ++i) {
        double scale = cluster_scales[i];
        bool is_qcd = (cluster_history[i] >> 27) & 1;
        ren_scale_val *= is_qcd ? scale : max_scale;
    }
    ren_scale_val = pow(ren_scale_val, 1.0 / cluster_max);
    if (fac_scale > ren_scale_val) {
        fac_scale = ren_scale_val;
    }

    ren_scale = ren_scale_val;
    fact_scale1 = fac_scale;
    fact_scale2 = fac_scale;

    // if multiple diagrams exist, pick one randomly
    int diag_count = state_machine[state];
    int rand_index = static_cast<int>(static_cast<double>(random) * diag_count);
    if (rand_index >= diag_count) {
        rand_index = diag_count - 1;
    }
    diagram_index = state_machine[state + rand_index + 1];
}

} // namespace kernels
} // namespace madspace

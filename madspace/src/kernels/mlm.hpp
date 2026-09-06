#pragma once

#include "definitions.hpp"
#include "kinematics.hpp"

namespace madspace {
namespace kernels {

constexpr int N_EXT_MAX = 12;
constexpr double ONE_PLUS_TINY = 1.000001;
// Stand-in for a clustering measure that came out inf or NaN; still far
// above any physical scale, and small enough that SCALE_MAX * 10 is finite.
constexpr double SCALE_MAX = 1e307;
// Words per clustering transition in the compiled state machine:
// (data, next_offset, trace_data). Must match state_machine_item_size in
// madspace/phasespace/mlm_clustering.hpp.
constexpr int STATE_ITEM_SIZE = 3;
// trace_data values, mirroring TraceMode in mlm_clustering.cpp: which daughter
// of a clustering the mother's parton line continues into.
constexpr int TRACE_FIRST = 0;
constexpr int TRACE_SECOND = 1;
constexpr int TRACE_HARDER = 2;
constexpr int TRACE_BOTH = 3;
// JetScaleScheme in mlm_clustering.hpp.
constexpr int SCHEME_EMISSION = 0;
constexpr int SCHEME_PRODUCTION = 1;

// mT^2 = E^2 - pz^2 (hadronic) or E^2 (lepton collider).
// based on djb_clus from Template/NLO/SubProcesses/cluster.f
template <typename T>
KERNELSPEC FVal<T> djb_clus(const FourMom<T>& p, bool hadronic) {
    auto r = hadronic ? (p[0] - p[3]) * (p[0] + p[3]) : p[0] * p[0];
    return max(r, 0.0);
}

// kt/Durham clustering measure for two partons.
// mass1/mass2 are the tracked clustering masses kept in a separate array and updated by
// update_momenta - NOT the Lorentz-invariant masses computed from the 4-momentum.
// jet_radius is the jet-radius parameter (Fortran common /to_dj/D).
// based on dj_clus from Template/NLO/SubProcesses/cluster.f
template <typename T>
KERNELSPEC FVal<T> dj_clus(
    const FourMom<T>& p1,
    const FourMom<T>& p2,
    FVal<T> mass1,
    FVal<T> mass2,
    bool hadronic,
    FVal<T> jet_radius
) {
    if (!hadronic) {
        // Durham e+e- measure: 2*min(E1^2, E2^2)*(1 - cos_theta)
        auto p1a = sqrt(esquare<T>(p1));
        auto p2a = sqrt(esquare<T>(p2));
        if (p1a * p2a == 0.0) {
            return 0.0;
        }
        auto costh = edot<T>(p1, p2) / (p1a * p2a);
        return 2.0 * min(p1[0] * p1[0], p2[0] * p2[0]) * max(1.0 - costh, 0.0);
    }

    // hadronic: massless+massive pair clusters to the lighter parton's mT^2
    bool massive1 = (mass1 > 0.0);
    bool massive2 = (mass2 > 0.0);
    if (!massive1 && massive2) {
        return djb_clus<T>(p1, true) * ONE_PLUS_TINY;
    }
    if (massive1 && !massive2) {
        return djb_clus<T>(p2, true) * ONE_PLUS_TINY;
    }

    // both massless or both massive: generalised kt measure in (eta, phi)
    auto pt1_sq = p1[1] * p1[1] + p1[2] * p1[2];
    auto pt2_sq = p2[1] * p2[1] + p2[2] * p2[2];
    if (pt1_sq == 0.0 || pt2_sq == 0.0) {
        return 0.0;
    }
    auto p1a = sqrt(pt1_sq + p1[3] * p1[3]);
    auto p2a = sqrt(pt2_sq + p2[3] * p2[3]);
    auto eta1 = 0.5 * log((p1a + p1[3]) / (p1a - p1[3]));
    auto eta2 = 0.5 * log((p2a + p2[3]) / (p2a - p2[3]));
    auto m_max_sq = max(mass1 * mass1, mass2 * mass2);
    auto dphi_cos = (p1[1] * p2[1] + p1[2] * p2[2]) / sqrt(pt1_sq * pt2_sq);
    auto result = m_max_sq +
        min(pt1_sq, pt2_sq) * 2.0 * (cosh(eta1 - eta2) - dphi_cos) /
            (jet_radius * jet_radius);
    // An exactly collinear pair makes eta1 or eta2 infinite and their
    // difference NaN. The Fortran zeroes both negative and NaN results here
    // ("prevent numerical inaccuracies"); max() alone would let the NaN
    // through.
    if (!(result > 0.0)) {
        return 0.0;
    }
    return result;
}

// Clustering scale for the pair (momentum1=pi, momentum2=pj).
// mass1/mass2 are the tracked clustering masses for momentum1/momentum2.
//
// Parameters replacing Fortran globals / derived quantities:
//   is_initial   : momentum2 is a beam particle (Fortran: j<=2 in cluster_one_step)
//   hadronic     : hadronic collider (Fortran: lpp[] from run.inc)
//   jet_radius            : jet-radius parameter (Fortran: common /to_dj/D)
//   massive_in/out1/out2 replace the Fortran get_clustering_type cl[0:2] bit-array
//   resonant     : replaces the iBWlist lookup in Fortran cluster_scale
//   mass_in/width_in: carried for future use; not consumed in the scale formula
//
// based on cluster_scale from Template/NLO/SubProcesses/cluster.f
template <typename T>
KERNELSPEC FVal<T> compute_scale(
    const FourMom<T>& momentum1,
    const FourMom<T>& momentum2,
    const FourMom<T>& momentum_sum,
    FVal<T> mass1,
    FVal<T> mass2,
    bool resonant,
    bool is_initial,
    bool massive_in,
    bool massive_out1,
    bool massive_out2,
    bool hadronic,
    FVal<T> jet_radius
) {
    if (is_initial) {
        // scale = mT of the final-state parton
        // small penalty when it goes against the beam
        auto scale = sqrt(djb_clus<T>(momentum2, hadronic));
        if ((momentum1[3] < 0.0) != (momentum2[3] < 0.0)) {
            scale = scale * ONE_PLUS_TINY;
        }
        return scale;
    }
    if (resonant) {
        return sqrt(max(lsquare<T>(momentum_sum), 0.0));
    }
    if (!massive_in && massive_out1 && !massive_out2) {
        return sqrt(fabs(ldot<T>(momentum2, momentum_sum))) / 2.0;
    }
    if (!massive_in && !massive_out1 && massive_out2) {
        return sqrt(fabs(ldot<T>(momentum1, momentum_sum))) / 2.0;
    }
    if (massive_in && !massive_out1 && !massive_out2) {
        return sqrt(max(lsquare<T>(momentum_sum), 0.0));
    }
    return sqrt(dj_clus<T>(momentum1, momentum2, mass1, mass2, hadronic, jet_radius));
}

// Update momenta and the separately-tracked masses after one clustering step.
// Based on update_momenta from Template/NLO/SubProcesses/cluster.f
template <typename T>
KERNELSPEC void update_momenta(
    int n_part,
    FourMom<T>* momenta,
    FVal<T>* masses,
    int& alive,
    int i_remove,
    int i_keep,
    bool resonant
) {
    alive &= ~(1 << i_remove);

    if (i_keep < 2) { // initial-state clustering
        int j_other = 1 - i_keep;
        for (int k = 0; k < 4; ++k) {
            momenta[i_keep][k] -= momenta[i_remove][k];
        }

        masses[i_keep] = (masses[i_keep] > 0.0) != (masses[i_remove] > 0.0)
            ? max(masses[i_keep], masses[i_remove])
            : 0.0;

        FourMom<T> com_boost_vector = {
            momenta[i_keep][0] + momenta[j_other][0],
            -(momenta[i_keep][1] + momenta[j_other][1]),
            -(momenta[i_keep][2] + momenta[j_other][2]),
            -(momenta[i_keep][3] + momenta[j_other][3]),
        };
        if (lsquare<T>(com_boost_vector) > 100.0) {
            // boost j_keep to COM frame to define the rotation axis, then apply to all
            // alive particles
            auto jkeep_cm = boost<T>(momenta[i_keep], com_boost_vector, 1.0);
            for (int j = 0; j < n_part; ++j) {
                if (alive & (1 << j)) {
                    momenta[j] = rotate_inverse<T>(
                        boost<T>(momenta[j], com_boost_vector, 1.0), jkeep_cm
                    );
                }
            }
        }
    } else { // final-state clustering
        for (int k = 0; k < 4; ++k) {
            momenta[i_keep][k] += momenta[i_remove][k];
        }

        if (resonant) {
            masses[i_keep] = sqrt(max(lsquare<T>(momenta[i_keep]), 0.0));
        } else {
            masses[i_keep] = max(masses[i_keep], masses[i_remove]);
        }
    }
}

template <typename T>
KERNELSPEC void mlm_clustering(
    FIn<T, 2> momenta,
    FIn<T, 0> random,
    IIn<T, 1> state_machine,
    FIn<T, 1> external_masses,
    FIn<T, 1> bw_masses,
    FIn<T, 1> bw_widths,
    FIn<T, 0> bw_cutoff,
    FIn<T, 0> jet_radius,
    FIn<T, 0> cm_energy,
    IIn<T, 0> jet_scale_scheme,
    FIn<T, 0> xqcut,
    FOut<T, 0> ren_scale,
    FOut<T, 0> fact_scale1,
    FOut<T, 0> fact_scale2,
    FOut<T, 1> outgoing_scales,
    IOut<T, 0> diagram_index,
    FOut<T, 0> xqcut_weight,
    bool hadronic
) {
    // we do not support SIMD for now, so we can assume simple types
    static_assert(std::is_same_v<IVal<T>, int>);
    static_assert(std::is_same_v<FVal<T>, double>);

    int state = 0, cluster_count = 0;
    int cluster_max = momenta.size() - 3;
    int n_part = momenta.size();
    FourMom<T> momenta_tmp[N_EXT_MAX];
    FVal<T> masses_tmp[N_EXT_MAX];
    int alive = (1 << n_part) - 1;
    int cluster_history[N_EXT_MAX - 3];
    int cluster_trace[N_EXT_MAX - 3];
    FVal<T> cluster_scales[N_EXT_MAX - 3];

    for (int i = 0; i < n_part; ++i) {
        for (int j = 0; j < 4; ++j) {
            momenta_tmp[i][j] = momenta[i][j];
        }
        masses_tmp[i] = external_masses[i];
    }

    int win_next_state = -1, win_data = 0, win_trace = TRACE_FIRST;
    bool win_resonant = false;
    FVal<T> win_scale = SCALE_MAX * 10.;
    while (cluster_count < cluster_max) {
        int data = state_machine[state];
        int next_state = state_machine[state + 1];
        int trace_data = state_machine[state + 2];
        int particle1 = data & 0xFF;
        int particle2 = (data >> 8) & 0xFF;
        int mass_index = (data >> 16) & 0xFF;
        bool massive_in = (data >> 24) & 1;
        bool massive_out1 = (data >> 25) & 1;
        bool massive_out2 = (data >> 26) & 1;
        bool is_last = (data >> 30) & 1;
        bool is_initial = (particle1 < 2);

        FourMom<T> momentum_sum{
            momenta_tmp[particle1][0] + momenta_tmp[particle2][0],
            momenta_tmp[particle1][1] + momenta_tmp[particle2][1],
            momenta_tmp[particle1][2] + momenta_tmp[particle2][2],
            momenta_tmp[particle1][3] + momenta_tmp[particle2][3],
        };

        bool resonant = false;
        if (mass_index != 0) {
            FVal<T> prop_m2 = lsquare<T>(momentum_sum);
            FVal<T> mass = bw_masses[mass_index - 1];
            FVal<T> width = bw_widths[mass_index - 1];
            FVal<T> half_window = FVal<T>(bw_cutoff) * width;
            FVal<T> m_min = max(mass - half_window, 0.0);
            FVal<T> m_max = mass + half_window;
            resonant = (prop_m2 >= m_min * m_min) && (prop_m2 <= m_max * m_max);
        }

        FVal<T> scale = compute_scale<T>(
            momenta_tmp[particle1],
            momenta_tmp[particle2],
            momentum_sum,
            masses_tmp[particle1],
            masses_tmp[particle2],
            resonant,
            is_initial,
            massive_in,
            massive_out1,
            massive_out2,
            hadronic,
            jet_radius
        );

        // An exactly collinear pair - which the boost and rotation applied after
        // an initial-state clustering can produce - makes the measure inf or
        // NaN. Map those onto a large finite value: they then lose to any
        // well-defined clustering, but a clustering is still always chosen, so
        // the walk cannot fall off the end of the state machine.
        if (!(scale < SCALE_MAX)) {
            scale = SCALE_MAX;
        }

        // The MG5 fortran code extracted the resonance structure from the integration
        // channel. This is not always possible in MG7, so prefer resonant configs
        // over non-resonant ones
        if (win_next_state == -1 || (!win_resonant && resonant) ||
            (win_resonant == resonant && scale < win_scale)) {
            win_next_state = next_state;
            win_scale = scale;
            win_data = data;
            win_trace = trace_data & 0x3;
            win_resonant = resonant;
        }
        if (is_last) {
            int p1_win = win_data & 0xFF;
            int p2_win = (win_data >> 8) & 0xFF;
            update_momenta<T>(
                n_part, momenta_tmp, masses_tmp, alive, p2_win, p1_win, win_resonant
            );
            state = win_next_state;
            cluster_history[cluster_count] = win_data;
            cluster_trace[cluster_count] = win_trace;
            cluster_scales[cluster_count] = win_scale;
            ++cluster_count;
            // Reset the whole selection, not just the scale: leaving
            // win_resonant set would stop any non-resonant candidate from ever
            // winning the next step, and leaving win_next_state set would send
            // the walk to a stale state if that happened.
            win_next_state = -1;
            win_data = 0;
            win_trace = TRACE_FIRST;
            win_resonant = false;
            win_scale = SCALE_MAX * 10.;
        } else {
            state += STATE_ITEM_SIZE;
        }
    }

    // Renormalization scale: geometric mean of QCD clustering scales
    // (non-QCD entries replaced by the maximum scale).
    // Factorization scale: smallest QCD clustering scale.
    FVal<T> fac_scale = 1e308, max_scale = 0.0;
    for (int i = 0; i < cluster_max; ++i) {
        FVal<T> scale = cluster_scales[i];
        bool is_qcd = (cluster_history[i] >> 27) & 1;
        if (is_qcd && scale < fac_scale) {
            fac_scale = scale;
        }
        if (scale > max_scale) {
            max_scale = scale;
        }
    }

    for (int i = 0; i < n_part - 2; ++i) {
        outgoing_scales[i] = 0.0;
    }

    // Which external legs the parton line of each surviving slot ends at.
    // madevent calls this ipart; a slot stands for two legs after a
    // g -> q qbar splitting, and for none once its line has been absorbed.
    int rep1[N_EXT_MAX], rep2[N_EXT_MAX];
    for (int i = 0; i < n_part; ++i) {
        rep1[i] = i;
        rep2[i] = -1;
    }

    bool by_production = jet_scale_scheme == SCHEME_PRODUCTION;
    FVal<T> ren_scale_val = 1.0;
    // The generation-level merging cut. A clustering that emitted a jet has to
    // be at or above xqcut, otherwise the matrix element is describing
    // radiation the parton shower is meant to produce and the event is
    // dropped. This mirrors the "Check xqcut for vertices with jet daughters
    // only" block of Template/LO/SubProcesses/reweight.f: the test runs per
    // clustering step, on each daughter that is still a bare final-state jet,
    // and it does not ask whether the vertex was a QCD one.
    //
    // is_last_cluster is exactly the "still a bare external leg" bookkeeping
    // the emission scheme already needs, so the two share it.
    bool passes_xqcut = true;
    int is_last_cluster = 0b11111111'11111111'11111100;
    for (int i = 0; i < cluster_max; ++i) {
        FVal<T> scale = cluster_scales[i];
        int data = cluster_history[i];
        int particle1 = data & 0xFF;
        int particle2 = (data >> 8) & 0xFF;
        bool is_qcd = (data >> 27) & 1;
        bool is_jet1 = (data >> 28) & 1;
        bool is_jet2 = (data >> 29) & 1;
        if (FVal<T>(xqcut) > 0.0 && scale < FVal<T>(xqcut) &&
            ((is_jet1 && (is_last_cluster & (1 << particle1))) ||
             (is_jet2 && (is_last_cluster & (1 << particle2))))) {
            passes_xqcut = false;
        }
        if (is_qcd) {
            if (by_production) {
                // Book this vertex onto every external leg the two daughters'
                // lines end at, keeping the hardest vertex each leg takes part
                // in. That is the scale at which the leg's line was produced.
                for (int k = 0; k < 2; ++k) {
                    if (!(k == 0 ? is_jet1 : is_jet2)) {
                        continue;
                    }
                    int daughter = k == 0 ? particle1 : particle2;
                    for (int m = 0; m < 2; ++m) {
                        int leg = m == 0 ? rep1[daughter] : rep2[daughter];
                        if (leg >= 2 && scale > outgoing_scales[leg - 2]) {
                            outgoing_scales[leg - 2] = scale;
                        }
                    }
                }
            } else {
                // Book the vertex at which the leg itself was emitted, i.e.
                // the first (softest) clustering it takes part in.
                if (is_jet1 && (is_last_cluster & (1 << particle1))) {
                    outgoing_scales[particle1 - 2] = scale;
                }
                if (is_jet2 && (is_last_cluster & (1 << particle2))) {
                    outgoing_scales[particle2 - 2] = scale;
                }
            }
            ren_scale_val *= scale;
        } else {
            ren_scale_val *= max_scale;
        }
        is_last_cluster &= ~((1 << particle1) | (1 << particle2));

        // Carry the parton line into the mother, which occupies slot
        // particle1. Only needed for the production scheme, but keeping it
        // unconditional costs nothing and keeps the two branches comparable.
        int trace = cluster_trace[i];
        int a1 = rep1[particle1], a2 = rep2[particle1];
        int b1 = rep1[particle2], b2 = rep2[particle2];
        if (trace == TRACE_HARDER || trace == TRACE_BOTH) {
            // madevent compares the transverse momenta of the representative
            // legs in the original event, not in the clustered one
            bool second_harder;
            if (a1 < 0) {
                second_harder = true;
            } else if (b1 < 0) {
                second_harder = false;
            } else {
                FVal<T> pt_a = momenta[a1][1] * momenta[a1][1] +
                    momenta[a1][2] * momenta[a1][2];
                FVal<T> pt_b = momenta[b1][1] * momenta[b1][1] +
                    momenta[b1][2] * momenta[b1][2];
                second_harder = pt_b > pt_a;
            }
            if (trace == TRACE_HARDER) {
                if (second_harder) {
                    rep1[particle1] = b1;
                    rep2[particle1] = b2;
                }
            } else {
                // both daughters carry the line on, hardest first
                rep1[particle1] = second_harder ? b1 : a1;
                rep2[particle1] = second_harder ? a1 : b1;
            }
        } else if (trace == TRACE_SECOND) {
            rep1[particle1] = b1;
            rep2[particle1] = b2;
        }
        // TRACE_FIRST: slot particle1 already holds daughter 1's line
    }

    // Any outgoing leg that no QCD clustering booked a scale onto keeps
    // sqrt(s) rather than zero, mirroring the "ptclus = etot" fallback in
    // Template/LO/SubProcesses/reweight.f. This covers the legs that are not
    // jets at all, and jets whose winning clustering history happened to
    // contain no QCD splitting. pt_clust is what an MLM veto compares against
    // qcut, so a leg reported at zero would trip a veto that a leg with no
    // clustering scale must never trip.
    for (int i = 0; i < n_part - 2; ++i) {
        if (outgoing_scales[i] == 0.0) {
            outgoing_scales[i] = cm_energy;
        }
    }

    ren_scale_val = pow(ren_scale_val, 1.0 / cluster_max);
    if (fac_scale > ren_scale_val) {
        fac_scale = ren_scale_val;
    }

    ren_scale = ren_scale_val;
    fact_scale1 = fac_scale;
    fact_scale2 = fac_scale;
    // A weight rather than a flag, so that it can simply multiply the event
    // weight the way every other cut in madspace does.
    xqcut_weight = passes_xqcut ? 1.0 : 0.0;

    int diag_count = state_machine[state];
    int rand_index = static_cast<int>(FVal<T>(random) * diag_count);
    if (rand_index >= diag_count) {
        rand_index = diag_count - 1;
    }
    diagram_index = state_machine[state + rand_index + 1];
}

template <typename T>
KERNELSPEC void kernel_mlm_clustering_hadronic(
    FIn<T, 2> momenta,
    FIn<T, 0> random,
    IIn<T, 1> state_machine,
    FIn<T, 1> external_masses,
    FIn<T, 1> bw_masses,
    FIn<T, 1> bw_widths,
    FIn<T, 0> bw_cutoff,
    FIn<T, 0> jet_radius,
    FIn<T, 0> cm_energy,
    IIn<T, 0> jet_scale_scheme,
    FIn<T, 0> xqcut,
    FOut<T, 0> ren_scale,
    FOut<T, 0> fact_scale1,
    FOut<T, 0> fact_scale2,
    FOut<T, 1> outgoing_scales,
    IOut<T, 0> diagram_index,
    FOut<T, 0> xqcut_weight
) {
    mlm_clustering<T>(
        momenta,
        random,
        state_machine,
        external_masses,
        bw_masses,
        bw_widths,
        bw_cutoff,
        jet_radius,
        cm_energy,
        jet_scale_scheme,
        xqcut,
        ren_scale,
        fact_scale1,
        fact_scale2,
        outgoing_scales,
        diagram_index,
        xqcut_weight,
        true
    );
}

template <typename T>
KERNELSPEC void kernel_mlm_clustering_leptonic(
    FIn<T, 2> momenta,
    FIn<T, 0> random,
    IIn<T, 1> state_machine,
    FIn<T, 1> external_masses,
    FIn<T, 1> bw_masses,
    FIn<T, 1> bw_widths,
    FIn<T, 0> bw_cutoff,
    FIn<T, 0> jet_radius,
    FIn<T, 0> cm_energy,
    IIn<T, 0> jet_scale_scheme,
    FIn<T, 0> xqcut,
    FOut<T, 0> ren_scale,
    FOut<T, 0> fact_scale1,
    FOut<T, 0> fact_scale2,
    FOut<T, 1> outgoing_scales,
    IOut<T, 0> diagram_index,
    FOut<T, 0> xqcut_weight
) {
    mlm_clustering<T>(
        momenta,
        random,
        state_machine,
        external_masses,
        bw_masses,
        bw_widths,
        bw_cutoff,
        jet_radius,
        cm_energy,
        jet_scale_scheme,
        xqcut,
        ren_scale,
        fact_scale1,
        fact_scale2,
        outgoing_scales,
        diagram_index,
        xqcut_weight,
        false
    );
}

} // namespace kernels
} // namespace madspace

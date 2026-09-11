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
// ScaleScheme in mlm_clustering.hpp.
constexpr int SCALES_CLUSTERING_MEAN = 0;
constexpr int SCALES_MADEVENT = 1;
// How the beam parton line is decided to carry on past a vertex: from each
// daughter's own flavour, or from the propagated goodjet flag of reweight.f.
constexpr int LINE_FLAVOR = 0;
constexpr int LINE_GOODJET = 1;
// How alpha_s is evaluated for a merged event.
constexpr int ALPHAS_NONE = 0;          // one coupling at mu_R, as before
constexpr int ALPHAS_PER_VERTEX = 1;    // alphas(pt_i) at each vertex, as madevent
constexpr int ALPHAS_GEOMETRIC = 2;     // alphas at the geometric mean of the pt_i
// trace_data bits above the 2-bit trace mode: the flavour of the mother.
constexpr int TRACE_IS_JET_IN = 1 << 2;
constexpr int TRACE_IS_COLORED_IN = 1 << 3;
// Bits 4..11 hold the mother's pdf flavour class plus one, so that zero means
// the line is not a parton and the pdf reweighting chain stops there. The
// classes themselves are assigned in mlm_clustering.cpp; the kernel only
// carries them, since which density they stand for is not known until the
// flavour has been sampled.
constexpr int TRACE_FLAVOR_SHIFT = 4;
constexpr int TRACE_FLAVOR_MASK = 0xFF;
// More of the mother's flavour, for isjetvx and the iqjets demotion.
constexpr int TRACE_IS_OCTET_IN = 1 << 12;
constexpr int TRACE_MOTHER_IS_DAU1 = 1 << 13;
constexpr int TRACE_MOTHER_IS_DAU2 = 1 << 14;
constexpr int TRACE_ALL_COLORLESS = 1 << 15;
// jet_leg_mask carries is_jet per leg in the low half and is_octet in the high
// half; n_ext_max is 12, so one word holds both.
constexpr int LEG_OCTET_SHIFT = 16;

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
    IIn<T, 0> scale_scheme,
    IIn<T, 0> beam_flags,
    IIn<T, 0> jet_leg_mask,
    IIn<T, 0> parton_line_scheme,
    IIn<T, 0> alphas_scheme,
    IIn<T, 0> pdf_reweighting,
    FOut<T, 0> ren_scale,
    FOut<T, 0> fact_scale1,
    FOut<T, 0> fact_scale2,
    FOut<T, 1> outgoing_scales,
    IOut<T, 0> diagram_index,
    FOut<T, 0> xqcut_weight,
    FOut<T, 1> alphas_scales,
    FOut<T, 0> alphas_weight,
    FOut<T, 0> pdf_scale1,
    FOut<T, 0> pdf_scale2,
    IOut<T, 1> pdf_rw_flavor,
    FOut<T, 1> pdf_rw_x,
    FOut<T, 1> pdf_rw_q_num,
    FOut<T, 1> pdf_rw_q_den,
    FOut<T, 1> pdf_rw_active,
    FOut<T, 1> pdf_rw_beam,
    bool hadronic
) {
    // we do not support SIMD for now, so we can assume simple types
    static_assert(std::is_same_v<IVal<T>, int>);
    static_assert(std::is_same_v<FVal<T>, double>);

    int state = 0, cluster_count = 0;
    int cluster_max = momenta.size() - 3;
    int n_part = momenta.size();
    FourMom<T> momenta_tmp[N_EXT_MAX];
    // The same clustering tracked in the lab frame, without the boosts and
    // rotations update_momenta applies. Only the 2 -> 1 root needs it: cluster.f
    // boosts and rotates back before taking that vertex's scale, so it is a
    // transverse mass in the lab and not in whatever frame the walk ended up in.
    FourMom<T> momenta_lab[N_EXT_MAX];
    FVal<T> masses_tmp[N_EXT_MAX];
    int alive = (1 << n_part) - 1;
    int cluster_history[N_EXT_MAX - 3];
    int cluster_trace[N_EXT_MAX - 3];
    FVal<T> cluster_scales[N_EXT_MAX - 3];
    // mT of the final-state daughter of each initial-state clustering,
    // zero for a final-state one. This is mt2ij in cluster.f, kept
    // unsquared like every other scale here.
    FVal<T> cluster_mt[N_EXT_MAX - 3];
    // The Pythia ISR momentum fraction of each initial-state clustering,
    // zcl in cluster.f, zero for a final-state one. The pdf reweighting walks
    // the beam's momentum fraction down the ladder with it.
    FVal<T> cluster_z[N_EXT_MAX - 3];

    for (int i = 0; i < n_part; ++i) {
        for (int j = 0; j < 4; ++j) {
            momenta_tmp[i][j] = momenta[i][j];
            momenta_lab[i][j] = momenta[i][j];
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
            win_trace = trace_data;
            win_resonant = resonant;
        }
        if (is_last) {
            int p1_win = win_data & 0xFF;
            int p2_win = (win_data >> 8) & 0xFF;
            // Read the daughter before update_momenta boosts it away. Only an
            // initial-state clustering has one, matching the iwin < 3 guard in
            // cluster.f.
            cluster_mt[cluster_count] = p1_win < 2
                ? sqrt(djb_clus<T>(momenta_tmp[p2_win], hadronic))
                : FVal<T>(0.0);
            // zclus() of Template/LO/Source/kin_functions.f: the ratio of the
            // partonic invariants before and after the emission is taken back
            // out of the beam, both measured against the other beam. Taken
            // here, before update_momenta, because that is where cluster.f
            // takes it; the boosts it applies preserve the dot products
            // anyway.
            cluster_z[cluster_count] = 0.0;
            if (p1_win < 2) {
                int other_beam = 1 - p1_win;
                FourMom<T> sum_prev, sum_red;
                for (int k = 0; k < 4; ++k) {
                    sum_prev[k] =
                        momenta_tmp[p1_win][k] + momenta_tmp[other_beam][k];
                    sum_red[k] = momenta_tmp[p1_win][k] -
                        momenta_tmp[p2_win][k] + momenta_tmp[other_beam][k];
                }
                FVal<T> s_prev = lsquare<T>(sum_prev);
                FVal<T> s_red = lsquare<T>(sum_red);
                // The Fortran gives up below 1 GeV^2 rather than dividing;
                // a z of zero then fails the 0 < z < 1 test downstream and the
                // momentum fraction is left alone.
                if (s_red >= 1.0 && s_prev > 0.0) {
                    cluster_z[cluster_count] = s_red / s_prev;
                }
            }
            // The mother keeps slot p1_win. A final-state clustering merges the
            // two daughters; an initial-state one takes the emission back out
            // of the beam, as pcl(imo) = pcl(ida1) - pcl(ida2) does in
            // cluster.f.
            for (int k = 0; k < 4; ++k) {
                momenta_lab[p1_win][k] = p1_win < 2
                    ? momenta_lab[p1_win][k] - momenta_lab[p2_win][k]
                    : momenta_lab[p1_win][k] + momenta_lab[p2_win][k];
            }
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
    bool slot_is_jet[N_EXT_MAX];
    // Colour of the line currently in each slot, tracked the same way. Only
    // the two beam slots are ever read back, at the 2 -> 1 root.
    bool slot_is_colored[N_EXT_MAX];
    // and whether it is an octet, which decides whether an emission counts as
    // a jet unconditionally: a gluon has a soft singularity the shower has to
    // be left free to fill.
    bool slot_is_octet[N_EXT_MAX];
    for (int i = 0; i < n_part; ++i) {
        rep1[i] = i;
        rep2[i] = -1;
        slot_is_jet[i] = (jet_leg_mask >> i) & 1;
        slot_is_colored[i] = i < 2 ? (((beam_flags >> (2 * i)) & 1) != 0) : true;
        slot_is_octet[i] = (jet_leg_mask >> (i + LEG_OCTET_SHIFT)) & 1;
    }
    // What the iqjets bookkeeping below needs from each step but cannot
    // reconstruct afterwards: which slots were still bare external legs, and
    // where the mother's line ended up. Recorded here rather than walked a
    // second time.
    int step_bare[N_EXT_MAX - 3];
    int step_mother_leg1[N_EXT_MAX - 3], step_mother_leg2[N_EXT_MAX - 3];

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
    int is_last_cluster = 0b11111111'11111111'11111100;
    for (int i = 0; i < cluster_max; ++i) {
        FVal<T> scale = cluster_scales[i];
        int data = cluster_history[i];
        int particle1 = data & 0xFF;
        int particle2 = (data >> 8) & 0xFF;
        bool is_qcd = (data >> 27) & 1;
        bool is_jet1 = (data >> 28) & 1;
        bool is_jet2 = (data >> 29) & 1;
        step_bare[i] = is_last_cluster;
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
        int trace = cluster_trace[i] & 0x3;
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

        // the merged line takes the mother's flavour
        slot_is_jet[particle1] = (cluster_trace[i] & TRACE_IS_JET_IN) != 0;
        slot_is_colored[particle1] = (cluster_trace[i] & TRACE_IS_COLORED_IN) != 0;
        slot_is_octet[particle1] = (cluster_trace[i] & TRACE_IS_OCTET_IN) != 0;
        step_mother_leg1[i] = rep1[particle1];
        step_mother_leg2[i] = rep2[particle1];
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
    FVal<T> fac_scale1 = fac_scale, fac_scale2 = fac_scale;

    // One pdf reweighting slot per clustering that can sit on a beam line,
    // plus one per beam for the 2 -> 1 root, which sits on both. A slot that
    // no step claims stays inert: its two scales are equal, so its ratio is
    // one, and its weight is one whatever the density comes out as.
    // Which outgoing legs the merging cut applies to. Filled by madevent's
    // iqjets walk below where that scale scheme is in use; without it the only
    // thing available is the leg's own flavour and the vertex's.
    // The scale each clustering is reweighted at. rewgt reads pt2ijcl *after*
    // setclscales has overwritten it - at the central vertex with the emitted
    // object's transverse mass, and at the last one to keep it above the first
    // - so the coupling does not see the raw clustering measure. Defaults to
    // that measure for the scheme that has no such table.
    FVal<T> alphas_step[N_EXT_MAX - 3];
    for (int i = 0; i < cluster_max; ++i) {
        alphas_step[i] = cluster_scales[i];
    }

    bool merging_jet[N_EXT_MAX];
    bool have_merging_jets = false;
    for (int i = 0; i < n_part; ++i) {
        merging_jet[i] = false;
    }

    int pdf_slot_count = n_part - 1;
    int rw_flavor[N_EXT_MAX - 1];
    FVal<T> rw_x[N_EXT_MAX - 1];
    FVal<T> rw_q_num[N_EXT_MAX - 1];
    FVal<T> rw_q_den[N_EXT_MAX - 1];
    FVal<T> rw_active[N_EXT_MAX - 1];
    // Which beam a slot sits on. Only the two root slots know that statically;
    // for the rest it depends on which clustering won, so it has to travel
    // with the slot.
    FVal<T> rw_beam[N_EXT_MAX - 1];
    for (int i = 0; i < pdf_slot_count; ++i) {
        rw_flavor[i] = 0;
        rw_x[i] = 1.0;
        rw_q_num[i] = 1.0;
        rw_q_den[i] = 1.0;
        rw_active[i] = 0.0;
        rw_beam[i] = 0.0;
    }
    FVal<T> pdf_scale_val[2] = {-1.0, -1.0};

    if (scale_scheme == SCALES_MADEVENT) {
        // madevent clusters one step further than this kernel does: its loop
        // runs all the way to the 2 -> 1 configuration, so it has one more
        // scale than cluster_max. That last step absorbs whatever is left of
        // the final state into a beam, and its scale is the transverse mass of
        // that object - for Drell-Yan, the boson's mT. It is also the step
        // jcentral usually lands on, which is why madevent's factorisation
        // scale comes out as mT of the lepton pair rather than a jet scale.
        //
        // Rather than extending the state machine, take that step here: with
        // the walk finished exactly one final-state slot is still alive, and
        // the clustering is fully determined by which beam it goes to.
        int leftover = -1;
        for (int i = 2; i < n_part; ++i) {
            if (alive & (1 << i)) {
                leftover = i;
                break;
            }
        }
        FVal<T> extra_scale = leftover < 0
            ? FVal<T>(0.0)
            : sqrt(djb_clus<T>(momenta_lab[leftover], hadronic));
        // Follow each beam's parton line through the clustering, exactly as
        // setclscales does in Template/LO/SubProcesses/reweight.f.
        //
        //   jfirst   the first initial-state clustering on this beam
        //   jlast    the last one while the line is still a jet
        //   jcentral the last one while the line is still coloured, which
        //            carries on past jlast through e.g. a W emission
        //
        // Beam j's line always occupies slot j here, because a clustering
        // keeps the lower of the two indices, so the slot index is madevent's
        // ibeam(j) without any extra bookkeeping.
        int jfirst[2] = {-1, -1}, jlast[2] = {-1, -1}, jcentral[2] = {-1, -1};
        bool partonline[2], qcdline[2];
        for (int j = 0; j < 2; ++j) {
            qcdline[j] = (beam_flags >> (2 * j)) & 1;
            partonline[j] = qcdline[j];
        }
        // goodjet(i) of reweight.f: "every clustering this line has been
        // through so far was a jet one". It starts as the flavour of each
        // external - QCD for a beam, jet for an outgoing leg - and can only
        // ever be cleared, so a line that once passed through a non-jet vertex
        // never counts as a parton line again. Reading each daughter's flavour
        // instead, which is what LINE_FLAVOR does, keeps beam lines alive past
        // vertices where madevent would have stopped them.
        bool goodjet[N_EXT_MAX];
        for (int i = 0; i < n_part; ++i) {
            goodjet[i] = i < 2 ? ((beam_flags >> (2 * i)) & 1) != 0
                               : ((jet_leg_mask >> i) & 1) != 0;
        }
        bool use_goodjet = parton_line_scheme == LINE_GOODJET;

        // iqjets of reweight.f: which outgoing legs count as merging jets.
        // This is the gate on the xqcut check, and the reason a jet emitted
        // where no radiation happened is exempt from it.
        //
        //   1      certainly a jet: emitted while the beam line was still a
        //          parton line, or a gluon, which has a soft singularity the
        //          shower must be left free to fill
        //   jcode  possibly a jet, resolved at the end
        //   0      not a merging jet
        //
        // jcode counts how many times the walk has crossed between QCD and
        // non-QCD vertices, so that an emission can be placed relative to those
        // crossings; increasecode makes the first jet vertex after a run of
        // non-jet ones count as another crossing.
        int iqjets[N_EXT_MAX];
        for (int i = 0; i < n_part; ++i) {
            iqjets[i] = 0;
        }
        int jcode = 1;
        bool increasecode = false;
        // isjet of the beam line itself as it stands, which is ipdgcl(ida(i))
        // in the jcode test - the daughter on the beam side, before the vertex
        // turns it into the mother.
        bool line_is_jet[2];
        for (int j = 0; j < 2; ++j) {
            line_is_jet[j] = ((beam_flags >> (2 * j + 1)) & 1) != 0;
        }

        // One step of the beam-line bookkeeping, for beam j at step i.
        // emitted_leg is ipart(1, ida(3-i)), the first external leg of the
        // emitted line, or -1 where there is none; is_jet_in and is_colored_in
        // are the mother's flavour.
        auto walk_line = [&](int j,
                             int i,
                             bool is_emitted_jet,
                             bool emitted_is_octet,
                             int emitted_leg,
                             bool is_jet_in,
                             bool is_colored_in) {
            if (partonline[j]) {
                if (jfirst[j] < 0) {
                    jfirst[j] = i;
                }
                jlast[j] = i;
                partonline[j] = is_emitted_jet && is_jet_in;
            } else {
                if (jfirst[j] < 0) {
                    jfirst[j] = i;
                }
                // the beam line has stopped being a parton line, so whatever
                // it becomes is not a good jet either
                goodjet[j] = false;
            }
            if (!is_emitted_jet || !line_is_jet[j] || !is_jet_in) {
                ++jcode;
                increasecode = true;
            } else if (increasecode) {
                ++jcode;
                increasecode = false;
            }
            if (is_emitted_jet && emitted_leg >= 2) {
                iqjets[emitted_leg] =
                    (partonline[j] || emitted_is_octet) ? 1 : jcode;
            }
            line_is_jet[j] = is_jet_in;
            if (qcdline[j]) {
                jcentral[j] = i;
                qcdline[j] = is_colored_in;
            }
        };

        int steps = cluster_max + (leftover >= 0 ? 1 : 0);
        for (int i = 0; i < cluster_max; ++i) {
            int data = cluster_history[i];
            int particle1 = data & 0xFF;
            int particle2 = (data >> 8) & 0xFF;
            int trace = cluster_trace[i];
            bool is_jet_in = (trace & TRACE_IS_JET_IN) != 0;
            if (particle1 >= 2) {
                // Final-state clustering. isjetvx() of reweight.f: a QCD
                // vertex, and one where a jet actually came out - which it did
                // if a jet daughter sits opposite either a jet mother or a
                // mother repeating the other daughter's flavour, i.e. an
                // emission off a line rather than a splitting into something
                // else.
                bool is_qcd_vertex = (data >> 27) & 1;
                bool is_jet1 = (data >> 28) & 1;
                bool is_jet2 = (data >> 29) & 1;
                bool mother_is_dau1 = (trace & TRACE_MOTHER_IS_DAU1) != 0;
                bool mother_is_dau2 = (trace & TRACE_MOTHER_IS_DAU2) != 0;
                bool is_octet_in = (trace & TRACE_IS_OCTET_IN) != 0;
                bool is_jet_vertex = is_qcd_vertex &&
                    ((is_jet1 && (is_jet_in || mother_is_dau2)) ||
                     (is_jet2 && (is_jet_in || mother_is_dau1)));
                int mother_leg1 = step_mother_leg1[i];
                int mother_leg2 = step_mother_leg2[i];
                if (!is_jet_vertex) {
                    // "Remove non-gluon jets that lead up to non-jet
                    // vertices": a quark that was counted as a jet earlier but
                    // whose line runs into a vertex that made no jet was not a
                    // merging emission after all. A vertex whose three lines
                    // all carry colour, or none of them do, is left alone.
                    if (!is_qcd_vertex && mother_leg1 >= 2) {
                        bool leg1_octet =
                            ((jet_leg_mask >> (mother_leg1 + LEG_OCTET_SHIFT)) & 1) != 0;
                        if (!leg1_octet) {
                            if ((trace & TRACE_ALL_COLORLESS) != 0) {
                                // a W W Z or h h h vertex: nothing to demote
                            } else if (mother_leg2 < 0) {
                                iqjets[mother_leg1] = 0;
                            } else if (iqjets[mother_leg1] > 0 &&
                                       mother_leg2 >= 2 &&
                                       iqjets[mother_leg2] > 0) {
                                // both halves of an octet's line are tagged,
                                // so one of them can go
                                iqjets[mother_leg1] = 0;
                            }
                        } else if (is_octet_in) {
                            iqjets[mother_leg1] = 0;
                        }
                    }
                    if (mother_leg2 >= 2 && !is_octet_in) {
                        bool leg2_octet =
                            ((jet_leg_mask >> (mother_leg2 + LEG_OCTET_SHIFT)) & 1) != 0;
                        if (!leg2_octet) {
                            iqjets[mother_leg2] = 0;
                        }
                    }
                    goodjet[particle1] = false;
                    continue;
                }
                // a jet vertex: every daughter that is still a bare external
                // jet was emitted here
                if (is_jet1 && particle1 >= 2 &&
                    (step_bare[i] & (1 << particle1))) {
                    iqjets[particle1] = 1;
                }
                if (is_jet2 && particle2 >= 2 &&
                    (step_bare[i] & (1 << particle2))) {
                    iqjets[particle2] = 1;
                }
                goodjet[particle1] = is_jet_in && goodjet[particle1] &&
                    goodjet[particle2];
                continue;
            }
            bool emitted = use_goodjet ? goodjet[particle2]
                                       : ((data >> 29) & 1) != 0;
            walk_line(
                particle1,
                i,
                emitted,
                slot_is_octet[particle2],
                rep1[particle2],
                is_jet_in,
                (trace & TRACE_IS_COLORED_IN) != 0
            );
        }
        if (leftover >= 0) {
            // The 2 -> 1 root clusters both beams into what is left of the
            // final state, so it lies on both beam lines and reweight.f walks
            // it twice, once per beam. Seen from beam j the emitted object is
            // that leftover and the mother is the *other* beam's parton, which
            // is what decides whether either line carries on past the root.
            bool leftover_jet =
                use_goodjet ? goodjet[leftover] : slot_is_jet[leftover];
            for (int j = 0; j < 2; ++j) {
                // Seen from beam j the mother is the other beam's line as it
                // stands at the root, not the external parton it started as -
                // reweight.f reads ipdgcl(imo) with imo = idacl(n,3-i).
                walk_line(
                    j,
                    cluster_max,
                    leftover_jet,
                    slot_is_octet[leftover],
                    rep1[leftover],
                    slot_is_jet[1 - j],
                    slot_is_colored[1 - j]
                );
            }
        }
        // "Emissions with code 1 are always jets; now take care of possible
        // jets". Once a beam line has stopped being a parton line, everything
        // tagged only provisionally and before the last crossing was not a
        // merging emission.
        if (!partonline[0] || !partonline[1]) {
            if (partonline[0] || partonline[1]) {
                --jcode;
            }
            for (int leg = 2; leg < n_part; ++leg) {
                if (iqjets[leg] > 1 && iqjets[leg] <= jcode) {
                    iqjets[leg] = 0;
                }
            }
        }
        for (int leg = 0; leg < n_part; ++leg) {
            merging_jet[leg] = leg >= 2 && iqjets[leg] > 0;
        }
        have_merging_jets = true;
        for (int j = 0; j < 2; ++j) {
            if (jfirst[j] < 0) {
                jfirst[j] = jlast[j];
            }
        }

        // The scale of each step, across the recorded history plus the extra
        // 2 -> 1 one, as a mutable table because the central vertices are
        // overwritten in place below just as pt2ijcl is in reweight.f.
        FVal<T> pt_step[N_EXT_MAX - 2];
        for (int i = 0; i < steps; ++i) {
            pt_step[i] = i == cluster_max ? extra_scale : cluster_scales[i];
        }

        // "Set central scale to mT2": at an initial-state clustering the
        // relevant scale is not the clustering measure but the transverse mass
        // of what was emitted there. For Drell-Yan the central vertex is where
        // the lepton pair attaches, which is why madevent's factorisation
        // scale comes out as mT(ll) and not a jet scale.
        for (int j = 0; j < 2; ++j) {
            if (jcentral[j] < 0) {
                continue;
            }
            // At the root cluster.f takes mt2ij from daughter 2, which there
            // is the second beam's parton: mT of a beam parton is zero, so
            // the root vertex never gets overridden.
            FVal<T> mt = jcentral[j] == cluster_max ? FVal<T>(0.0)
                                                    : cluster_mt[jcentral[j]];
            if (mt > 0.0) {
                pt_step[jcentral[j]] = mt;
            }
        }

        // "Ensure that last scales are at least as big as first scales".
        // reweight.f raises the entry in pt2ijcl itself, so when jlast and
        // jcentral are the same vertex - which is the usual case, both beam
        // lines running QCD all the way to the root - the central scale is
        // raised with it and mu_F comes out as that one scale rather than a
        // geometric mean of two. Writing into pt_step keeps that.
        for (int j = 0; j < 2; ++j) {
            if (jlast[j] >= 0 && jfirst[j] >= 0 &&
                pt_step[jfirst[j]] > pt_step[jlast[j]]) {
                pt_step[jlast[j]] = pt_step[jfirst[j]];
            }
        }
        for (int i = 0; i < cluster_max; ++i) {
            alphas_step[i] = pt_step[i];
        }

        FVal<T> s_last[2], s_central[2];
        for (int j = 0; j < 2; ++j) {
            s_last[j] = jlast[j] < 0 ? 0.0 : pt_step[jlast[j]];
            s_central[j] = jcentral[j] < 0 ? 0.0 : pt_step[jcentral[j]];
        }

        // mu_R: the geometric mean of the four scales. madevent writes it as
        // (pt2 pt2 pt2 pt2)^(1/8) over squared scales, which is the same thing.
        FVal<T> last_scale = pt_step[steps - 1];
        // Take the roots before multiplying. A degenerate clustering leaves a
        // scale at SCALE_MAX, and the product of four of those overflows to
        // infinity, which then reaches alpha_s as a NaN.
        if (jlast[0] >= 0 && jlast[1] >= 0) {
            ren_scale_val = pow(s_last[0], 0.25) * pow(s_central[0], 0.25) *
                pow(s_last[1], 0.25) * pow(s_central[1], 0.25);
        } else if (jlast[0] >= 0) {
            ren_scale_val = sqrt(s_last[0]) * sqrt(s_central[0]);
        } else if (jlast[1] >= 0) {
            ren_scale_val = sqrt(s_last[1]) * sqrt(s_central[1]);
        } else if (jcentral[0] >= 0 && jcentral[1] >= 0) {
            ren_scale_val = sqrt(s_central[0]) * sqrt(s_central[1]);
        } else if (jcentral[0] >= 0) {
            ren_scale_val = s_central[0];
        } else if (jcentral[1] >= 0) {
            ren_scale_val = s_central[1];
        } else {
            ren_scale_val = last_scale;
        }

        // mu_F, one per beam. madevent stores q2fact = mu_F^2 as
        // sqrt(pt2[jlast] pt2[jcentral]), i.e. mu_F is the geometric mean of
        // the two scales.
        fac_scale1 = jlast[0] >= 0 ? sqrt(s_last[0]) * sqrt(s_central[0]) : 0.0;
        fac_scale2 = jlast[1] >= 0 ? sqrt(s_last[1]) * sqrt(s_central[1]) : 0.0;
        // "We have a qcd line going through the whole event, use single scale"
        if (jcentral[0] >= 0 && jcentral[0] == jcentral[1]) {
            fac_scale1 = max(fac_scale1, fac_scale2);
            fac_scale2 = fac_scale1;
        }

        // "Take care of case when jcentral are zero".
        //
        // The branch that ends that chain in reweight.f, capping each beam at
        // min(pt2ijcl(jfirst), q2fact) when pdfwgt is on, is deliberately not
        // ported. It exists so madevent can evaluate the PDF low on the
        // clustering ladder and reweight back up; the central scale is kept in
        // q2bck and restored, which is why turning pdfwgt off moves madevent's
        // cross section (1113 -> 1093 pb for Z+0,1,2,3j) but leaves SCALUP
        // unchanged. madspace has no such ladder reweighting, so the central
        // scale is what both the PDF and the LHE should see.
        if (jcentral[0] < 0 && jcentral[1] < 0) {
            if (!(fac_scale1 > 0.0) && !(fac_scale2 > 0.0)) {
                fac_scale1 = pt_step[steps - 1];
                fac_scale2 = fac_scale1;
            }
        } else if (jcentral[0] < 0) {
            if (jfirst[0] >= 0) {
                fac_scale1 = pt_step[jfirst[0]];
            }
        } else if (jcentral[1] < 0) {
            if (jfirst[1] >= 0) {
                fac_scale2 = pt_step[jfirst[1]];
            }
        }
        if (!(fac_scale1 > 0.0)) {
            fac_scale1 = ren_scale_val;
        }
        if (!(fac_scale2 > 0.0)) {
            fac_scale2 = ren_scale_val;
        }
        // and keep everything inside the range the rest of the kernel uses
        if (!(ren_scale_val > 0.0) || !(ren_scale_val < SCALE_MAX)) {
            ren_scale_val = last_scale;
        }
        if (!(fac_scale1 > 0.0) || !(fac_scale1 < SCALE_MAX)) {
            fac_scale1 = ren_scale_val;
        }
        if (!(fac_scale2 > 0.0) || !(fac_scale2 < SCALE_MAX)) {
            fac_scale2 = ren_scale_val;
        }

        // The pdf half of rewgt in Template/LO/SubProcesses/reweight.f, the
        // other thing madevent does to a merged event that a single density at
        // the factorisation scale does not.
        //
        // A merged event's beam density belongs at the scale of the emission
        // that took the parton out of the beam, not at the scale of the hard
        // process. madevent gets there in two moves: it evaluates the density
        // low on the clustering ladder, at min(pt(jfirst), mu_F), and then
        // walks each beam line back up, multiplying by f(x z, Q_i) / f(x z,
        // Q_i-1) at every further clustering the line takes part in, with the
        // momentum fraction rescaled by that clustering's z as it goes. The
        // last step lands on mu_F itself, so the chain ends where the density
        // would have been evaluated in the first place - except when the line
        // has only one clustering on it, where the lowered scale stands with
        // nothing to correct it. That asymmetry is madevent's, and it is a
        // good part of why turning pdfwgt off moves the cross section at all.
        //
        // Only the scales, the momentum fractions and the flavour classes are
        // decided here. Which density a class stands for is not knowable yet:
        // the flavour is sampled from the very densities this is correcting.
        bool reweight_pdf =
            pdf_reweighting != 0 && jcentral[0] >= 0 && jcentral[1] >= 0;
        pdf_scale_val[0] = fac_scale1;
        pdf_scale_val[1] = fac_scale2;
        if (reweight_pdf) {
            FVal<T> q_central[2] = {fac_scale1, fac_scale2};
            for (int j = 0; j < 2; ++j) {
                if (jlast[j] >= 0 && jfirst[j] >= 0 && jfirst[j] <= jlast[j]) {
                    pdf_scale_val[j] = min(pt_step[jfirst[j]], q_central[j]);
                }
            }
            // Where each beam's line has got to: its flavour class, the
            // product of the z it has picked up, and the scale its density
            // was last evaluated at. reweight.f keeps these as ibeam(j),
            // xnow(j) and pt2pdf(ibeam(j)).
            int line_class[2];
            FVal<T> x_frac[2] = {1.0, 1.0};
            FVal<T> pt_pdf[2] = {0.0, 0.0};
            for (int j = 0; j < 2; ++j) {
                line_class[j] =
                    ((beam_flags >> (8 * (j + 1))) & TRACE_FLAVOR_MASK) - 1;
            }
            for (int i = 0; i < steps; ++i) {
                for (int j = 0; j < 2; ++j) {
                    // A recorded clustering sits on the beam whose slot is its
                    // first daughter, and on no beam at all if that daughter is
                    // a final-state one. The root sits on both.
                    bool at_root = i >= cluster_max;
                    if (!at_root && (cluster_history[i] & 0xFF) != j) {
                        continue;
                    }
                    // Once a line stops being a parton reweight.f stops
                    // advancing ibeam(j), so no later clustering can match it
                    // again and the chain is over for good.
                    if (line_class[j] < 0) {
                        continue;
                    }
                    int slot = at_root ? cluster_max + j : i;
                    // zcl is 1 at the root, so the momentum fraction is only
                    // ever rescaled by the recorded clusterings.
                    FVal<T> z = at_root ? FVal<T>(1.0) : cluster_z[i];
                    if (z > 0.0 && z < 1.0) {
                        x_frac[j] *= z;
                    }
                    FVal<T> q_now = i == jlast[j] ? q_central[j]
                                                  : min(pt_step[i], q_central[j]);
                    if (!(pt_pdf[j] > 0.0)) {
                        // the first clustering on the line only records where
                        // the density it already has was evaluated
                        pt_pdf[j] = q_now;
                    } else if (pt_pdf[j] < q_now && i <= jlast[j]) {
                        rw_flavor[slot] = line_class[j];
                        rw_x[slot] = x_frac[j];
                        rw_q_num[slot] = q_now;
                        rw_q_den[slot] = pt_pdf[j];
                        rw_active[slot] = 1.0;
                        rw_beam[slot] = j;
                        pt_pdf[j] = q_now;
                    }
                    // and the line becomes the mother; at the root that is the
                    // s-channel object, which ends the chain.
                    line_class[j] = at_root
                        ? -1
                        : ((cluster_trace[i] >> TRACE_FLAVOR_SHIFT) &
                           TRACE_FLAVOR_MASK) -
                            1;
                }
            }
        }
    }

    // The merging cut. A matrix-element jet below xqcut is radiation the parton
    // shower is meant to produce, so the event is dropped - but only for a leg
    // that is a merging jet in the first place. madevent decides that with
    // iqjets, which is only ever set for a leg emitted at a jet vertex; a leg
    // it leaves at zero is exempt however soft the vertex it sits on. Without
    // that gate a jet is rejected at vertices that produced no radiation at
    // all, most visibly a quark pairing into a W, whose measure is a
    // lepton-side kt with nothing to do with the jet's own transverse
    // momentum.
    bool passes_xqcut = true;
    if (FVal<T>(xqcut) > 0.0) {
        for (int i = 0; i < cluster_max; ++i) {
            if (!(cluster_scales[i] < FVal<T>(xqcut))) {
                continue;
            }
            int data = cluster_history[i];
            int particle1 = data & 0xFF;
            int particle2 = (data >> 8) & 0xFF;
            bool is_qcd = (data >> 27) & 1;
            for (int k = 0; k < 2; ++k) {
                int slot = k == 0 ? particle1 : particle2;
                if (slot < 2 || !(step_bare[i] & (1 << slot))) {
                    continue;
                }
                bool counts = have_merging_jets
                    ? merging_jet[slot]
                    : (is_qcd && ((data >> (28 + k)) & 1) != 0);
                if (counts) {
                    passes_xqcut = false;
                }
            }
        }
    }

    ren_scale = ren_scale_val;
    fact_scale1 = fac_scale1;
    fact_scale2 = fac_scale2;
    // A weight rather than a flag, so that it can simply multiply the event
    // weight the way every other cut in madspace does.
    // Scale of each clustering vertex for the alpha_s reweighting, following
    // the loop in Template/LO/SubProcesses/reweight.f: every clustering except
    // the last one is reweighted by alphas(pt_clust) / alphas(mu_R), and only
    // where a parton is produced. A vertex that is not reweighted is handed
    // mu_R itself, so its ratio is exactly one and the consumer needs no mask.
    //
    // madevent gates this on goodjet / ispartonvx; the QCD flag of the vertex
    // is the same statement for every case that arises here - all three lines
    // coloured - and is what the rest of this kernel already uses.
    //
    // reweight.f drops an event outright when a reweighted vertex sits at or
    // below 2 GeV, where the coupling is not to be trusted; alphas_weight is
    // that veto.
    //
    // ALPHAS_GEOMETRIC hands every reweighted vertex the same scale, the
    // geometric mean of the individual ones, so the product below turns into
    // alphas(<pt>)^n. That is the cheaper approximation of the same idea: one
    // coupling for the whole ladder rather than one per rung.
    bool alphas_ok = true;
    FVal<T> log_sum = 0.0;
    int reweighted = 0;
    for (int i = 0; i < cluster_max; ++i) {
        if ((cluster_history[i] >> 27) & 1) {
            FVal<T> scale = alphas_step[i];
            if (!(scale * scale > 4.0)) {
                alphas_ok = false;
            } else {
                log_sum += log(scale);
                ++reweighted;
            }
        }
    }
    FVal<T> mean_scale =
        reweighted > 0 ? exp(log_sum / reweighted) : ren_scale_val;
    for (int i = 0; i < cluster_max; ++i) {
        bool is_qcd_vertex = (cluster_history[i] >> 27) & 1;
        if (alphas_scheme == ALPHAS_NONE || !is_qcd_vertex) {
            alphas_scales[i] = ren_scale_val;
        } else if (alphas_scheme == ALPHAS_GEOMETRIC) {
            alphas_scales[i] = mean_scale;
        } else {
            alphas_scales[i] = alphas_step[i];
        }
    }

    // Kept apart from xqcut_weight: that one is the merging cut and nothing
    // else, and only the reweighting cares where the coupling stops being
    // usable.
    alphas_weight = (alphas_scheme == ALPHAS_NONE || alphas_ok) ? 1.0 : 0.0;
    xqcut_weight = passes_xqcut ? 1.0 : 0.0;

    // Any scale scheme other than madevent's has no clustering ladder to walk,
    // so the density stays where the factorisation scale put it.
    if (!(pdf_scale_val[0] > 0.0)) {
        pdf_scale_val[0] = fac_scale1;
    }
    if (!(pdf_scale_val[1] > 0.0)) {
        pdf_scale_val[1] = fac_scale2;
    }
    pdf_scale1 = pdf_scale_val[0];
    pdf_scale2 = pdf_scale_val[1];
    for (int i = 0; i < pdf_slot_count; ++i) {
        if (rw_active[i] == 0.0) {
            // Somewhere the density is defined, so that an inert slot cannot
            // trip the low-density veto the consumer applies.
            rw_q_num[i] = pdf_scale_val[0];
            rw_q_den[i] = pdf_scale_val[0];
        }
        pdf_rw_flavor[i] = rw_flavor[i];
        pdf_rw_x[i] = rw_x[i];
        pdf_rw_q_num[i] = rw_q_num[i];
        pdf_rw_q_den[i] = rw_q_den[i];
        pdf_rw_active[i] = rw_active[i];
        pdf_rw_beam[i] = rw_beam[i];
    }

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
    IIn<T, 0> scale_scheme,
    IIn<T, 0> beam_flags,
    IIn<T, 0> jet_leg_mask,
    IIn<T, 0> parton_line_scheme,
    IIn<T, 0> alphas_scheme,
    IIn<T, 0> pdf_reweighting,
    FOut<T, 0> ren_scale,
    FOut<T, 0> fact_scale1,
    FOut<T, 0> fact_scale2,
    FOut<T, 1> outgoing_scales,
    IOut<T, 0> diagram_index,
    FOut<T, 0> xqcut_weight,
    FOut<T, 1> alphas_scales,
    FOut<T, 0> alphas_weight,
    FOut<T, 0> pdf_scale1,
    FOut<T, 0> pdf_scale2,
    IOut<T, 1> pdf_rw_flavor,
    FOut<T, 1> pdf_rw_x,
    FOut<T, 1> pdf_rw_q_num,
    FOut<T, 1> pdf_rw_q_den,
    FOut<T, 1> pdf_rw_active,
    FOut<T, 1> pdf_rw_beam
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
        scale_scheme,
        beam_flags,
        jet_leg_mask,
        parton_line_scheme,
        alphas_scheme,
        pdf_reweighting,
        ren_scale,
        fact_scale1,
        fact_scale2,
        outgoing_scales,
        diagram_index,
        xqcut_weight,
        alphas_scales,
        alphas_weight,
        pdf_scale1,
        pdf_scale2,
        pdf_rw_flavor,
        pdf_rw_x,
        pdf_rw_q_num,
        pdf_rw_q_den,
        pdf_rw_active,
        pdf_rw_beam,
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
    IIn<T, 0> scale_scheme,
    IIn<T, 0> beam_flags,
    IIn<T, 0> jet_leg_mask,
    IIn<T, 0> parton_line_scheme,
    IIn<T, 0> alphas_scheme,
    IIn<T, 0> pdf_reweighting,
    FOut<T, 0> ren_scale,
    FOut<T, 0> fact_scale1,
    FOut<T, 0> fact_scale2,
    FOut<T, 1> outgoing_scales,
    IOut<T, 0> diagram_index,
    FOut<T, 0> xqcut_weight,
    FOut<T, 1> alphas_scales,
    FOut<T, 0> alphas_weight,
    FOut<T, 0> pdf_scale1,
    FOut<T, 0> pdf_scale2,
    IOut<T, 1> pdf_rw_flavor,
    FOut<T, 1> pdf_rw_x,
    FOut<T, 1> pdf_rw_q_num,
    FOut<T, 1> pdf_rw_q_den,
    FOut<T, 1> pdf_rw_active,
    FOut<T, 1> pdf_rw_beam
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
        scale_scheme,
        beam_flags,
        jet_leg_mask,
        parton_line_scheme,
        alphas_scheme,
        pdf_reweighting,
        ren_scale,
        fact_scale1,
        fact_scale2,
        outgoing_scales,
        diagram_index,
        xqcut_weight,
        alphas_scales,
        alphas_weight,
        pdf_scale1,
        pdf_scale2,
        pdf_rw_flavor,
        pdf_rw_x,
        pdf_rw_q_num,
        pdf_rw_q_den,
        pdf_rw_active,
        pdf_rw_beam,
        false
    );
}

} // namespace kernels
} // namespace madspace

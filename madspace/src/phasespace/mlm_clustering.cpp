#include "madspace/phasespace/mlm_clustering.hpp"

#include <bit>
#include <cmath>
#include <format>
#include <map>
#include <set>
#include <unordered_map>

using namespace madspace;

namespace {

// Largest number of external particles the kernel can handle (mlm.hpp
// N_EXT_MAX), and the number of bits available for a clustering mask.
constexpr std::size_t n_ext_max = 12;

// Properties of the line a clustering mask stands for: the external particle
// itself for a single-bit mask, the propagator recombining them otherwise.
struct LineMeta {
    double mass = 0.;
    double width = 0.;
    int pdg_id = 0;
    bool known = false;
};

// Mirrors isjet() in Template/LO/SubProcesses/reweight.f: a final state
// particle a parton shower would turn into a jet, and hence one that gets its
// own clustering scale in the LHE output. 81 is the merged-flavor placeholder
// used by flavor grouping.
bool is_jet_pdg(int pdg_id, int max_jet_flavor) {
    int a = std::abs(pdg_id);
    return (a >= 1 && a <= max_jet_flavor) || a == 21 || a == 81;
}

// Color representation of a line, as the signed color of the model (1 for a
// singlet, +-3 for a (anti)triplet, 8 for an octet, +-6 for a sextet). The
// merged-flavor placeholders that survive in propagator pdg ids stand for the
// particles clean_pids() maps them to in launch.py: 81 is a quark, 82 an
// electron and 83 a neutrino.
int color_rep(int pdg_id, const std::unordered_map<int, int>& pdg_color_types) {
    if (auto search = pdg_color_types.find(pdg_id); search != pdg_color_types.end()) {
        return std::abs(search->second);
    }
    int a = std::abs(pdg_id);
    if ((a >= 1 && a <= 6) || a == 81) {
        return 3;
    }
    if (a == 21) {
        return 8;
    }
    return 1;
}

// Whether a line carries color. Used to decide whether a *vertex* is a QCD
// splitting, which is what selects the clustering scales entering mu_R/mu_F.
bool is_colored_pdg(int pdg_id, const std::unordered_map<int, int>& pdg_color_types) {
    return color_rep(pdg_id, pdg_color_types) != 1;
}

// How the parton line of the mother of a clustering continues into its two
// daughters, so that a clustering scale can be booked onto the external legs
// the line ends at. Mirrors ipartupdate() in
// Template/LO/SubProcesses/reweight.f, which decides the same thing from the
// color representations (its leading pdg-equality tests give the same answer
// as the color ones for every case they cover).
enum TraceMode {
    trace_first = 0,   // the line continues into daughter 1
    trace_second = 1,  // ... into daughter 2
    trace_harder = 2,  // ... into whichever daughter is harder (decided per event)
    trace_both = 3,    // the mother stands for both daughters' lines
};

TraceMode trace_mode_for(int color_in, int color_1, int color_2) {
    if (color_in == 8) {
        // g -> g g, or an octet splitting to two octets: follow the harder one
        if (color_1 == 8 && color_2 == 8) {
            return trace_harder;
        }
        // g -> q qbar: both daughters carry the line on
        if (color_1 == 3 && color_2 == 3) {
            return trace_both;
        }
        // g -> g H and friends: follow the colored daughter
        if (color_1 == 8) {
            return trace_first;
        }
        if (color_2 == 8) {
            return trace_second;
        }
    } else if (color_in == 3) {
        // an epsilon^ijk vertex, 3 -> 3 3, is treated like a photon emission
        // and follows the first daughter
        if (color_1 == 3 && color_2 == 3) {
            return trace_first;
        }
        // q -> q g, q -> q Z/H/W, and the exotic q -> q' S variants
        if (color_1 == 3) {
            return trace_first;
        }
        if (color_2 == 3) {
            return trace_second;
        }
    } else if (color_in == 6) {
        if (color_1 == 3 && color_2 == 3) {
            return trace_both;
        }
    } else if (color_in == 1) {
        return trace_both;
    }
    // Nothing matched. The Fortran stops the run here; keeping the line on the
    // first daughter is wrong but local, and only costs the clustering scale of
    // one leg rather than the whole event.
    return trace_first;
}

struct CompileContext {
    const nested_vector2<int>& valid_diags;
    const std::vector<LineMeta>& mask_meta;
    std::vector<double>& bw_masses;
    std::vector<double>& bw_widths;
    std::map<std::pair<double, double>, int>& bw_indices;
    const std::unordered_map<int, int>& pdg_color_types;
    bool have_pdg_ids;
    int max_jet_flavor;
};

using StateKey = std::pair<std::vector<int>, std::vector<int>>;
struct StateItem {
    int next_state;
    int particle1;
    int particle2;
    int mass_index;
    bool massive_in;
    bool massive_out1;
    bool massive_out2;
    bool is_qcd;
    bool is_jet1;
    bool is_jet2;
    TraceMode trace_mode;
};

// 1-based index into the Breit-Wigner tables handed to the kernel, or 0 for
// "never treat this clustering as resonant".
int breit_wigner_index(CompileContext& ctx, const LineMeta& meta) {
    if (!meta.known || meta.width <= 0. || meta.mass <= 0.) {
        return 0;
    }
    auto key = std::pair{meta.mass, meta.width};
    if (auto search = ctx.bw_indices.find(key); search != ctx.bw_indices.end()) {
        return search->second;
    }
    int index = static_cast<int>(ctx.bw_masses.size()) + 1;
    ctx.bw_masses.push_back(meta.mass);
    ctx.bw_widths.push_back(meta.width);
    ctx.bw_indices[key] = index;
    return index;
}

StateItem make_state_item(
    CompileContext& ctx, int next_state, int mask_in, int mask_1, int mask_2
) {
    int particle1 = std::countr_zero(static_cast<unsigned int>(mask_1));
    int particle2 = std::countr_zero(static_cast<unsigned int>(mask_2));
    auto& meta_in = ctx.mask_meta.at(mask_in);
    auto& meta_1 = ctx.mask_meta.at(mask_1);
    auto& meta_2 = ctx.mask_meta.at(mask_2);
    // The state machine always names the merged pseudo-particle by the lowest
    // bit of the combined mask, so particle1 < particle2 and an initial-state
    // (t-channel) clustering is exactly the case particle1 < 2.
    bool is_initial = particle1 < 2;

    // A t-channel propagator is spacelike and can never go on shell, so the
    // resonance test is only meaningful for a final-state clustering.
    int mass_index = is_initial ? 0 : breit_wigner_index(ctx, meta_in);

    int color_1 = color_rep(meta_1.pdg_id, ctx.pdg_color_types);
    int color_2 = color_rep(meta_2.pdg_id, ctx.pdg_color_types);
    int color_in = color_rep(meta_in.pdg_id, ctx.pdg_color_types);
    if (meta_in.pdg_id == 0) {
        // The propagator's flavor was not supplied (the topology fixtures used
        // by the tests carry masses only), so color_rep would call it a
        // singlet. Infer it from the daughters instead, which is unambiguous
        // for a QCD vertex.
        if (color_1 == 3 && color_2 == 3) {
            color_in = 8;
        } else if (color_1 == 8 && color_2 == 8) {
            color_in = 8;
        } else if (color_1 == 3 || color_2 == 3) {
            color_in = 3;
        }
    }
    // An initial-state clustering carries the beam line on, which is daughter 1
    // by construction, so the color-based table does not apply to it.
    TraceMode trace_mode =
        is_initial ? trace_first : trace_mode_for(color_in, color_1, color_2);

    bool is_qcd, is_jet1, is_jet2;
    if (!ctx.have_pdg_ids) {
        // No flavor information supplied: fall back to assuming every
        // clustering is a massless QCD splitting between jets.
        is_qcd = is_jet1 = is_jet2 = true;
    } else {
        // A vertex is a QCD splitting when all three of its lines carry color.
        // Checking the parent matters: q qbar -> Z has two colored children and
        // a colorless parent, and is not a QCD vertex.
        //
        // A pdg id of 0 means the propagator's flavor was not supplied (the
        // topology fixtures used by the tests carry masses only). Fall back to
        // the two children then: a splitting of two colored lines is QCD unless
        // the parent says otherwise, which is also the only test available for
        // t-channel lines whose flavor changes along the chain.
        is_qcd = color_1 != 1 && color_2 != 1 &&
            (meta_in.pdg_id == 0 || color_in != 1);
        // Only final state particles get a clustering scale in the LHE output,
        // so a beam leg is never a jet here.
        is_jet1 = !is_initial && is_jet_pdg(meta_1.pdg_id, ctx.max_jet_flavor);
        is_jet2 = is_jet_pdg(meta_2.pdg_id, ctx.max_jet_flavor);
    }

    return {
        .next_state = next_state,
        .particle1 = particle1,
        .particle2 = particle2,
        .mass_index = mass_index,
        .massive_in = meta_in.mass > 0.,
        .massive_out1 = meta_1.mass > 0.,
        .massive_out2 = meta_2.mass > 0.,
        .is_qcd = is_qcd,
        .is_jet1 = is_jet1,
        .is_jet2 = is_jet2,
        .trace_mode = trace_mode,
    };
}

void find_clusterings(
    CompileContext& ctx,
    const std::vector<int>& particle_masks,
    const std::vector<int>& diagrams,
    nested_vector2<StateItem>& states,
    std::map<StateKey, int>& state_map,
    std::set<int>& dead_states,
    int prev_index
) {
    int n_masks = particle_masks.size();
    for (int i = 0; i < n_masks - 1; ++i) {
        for (int j = i + 1; j < n_masks; ++j) {
            int mask_i = particle_masks.at(i), mask_j = particle_masks.at(j);
            int mask = mask_i | mask_j;
            auto& valid = ctx.valid_diags.at(mask);
            if (valid.size() == 0) {
                continue;
            }
            std::vector<int> new_masks, new_diags;
            new_masks.insert(
                new_masks.end(), particle_masks.begin(), particle_masks.begin() + i
            );
            new_masks.push_back(mask);
            new_masks.insert(
                new_masks.end(),
                particle_masks.begin() + i + 1,
                particle_masks.begin() + j
            );
            new_masks.insert(
                new_masks.end(), particle_masks.begin() + j + 1, particle_masks.end()
            );
            for (int index : diagrams) {
                if (std::find(valid.begin(), valid.end(), index) != valid.end()) {
                    new_diags.push_back(index);
                }
            }
            // valid_diags says some diagram allows this clustering, not that
            // any of the diagrams still in play does. With none left there is
            // no clustering history to continue, and following it anyway
            // builds a state the walk can never leave.
            if (new_diags.size() == 0) {
                continue;
            }

            bool is_terminal = new_masks.size() == 3;
            StateKey key;
            if (is_terminal) {
                key = {{}, new_diags};
            } else {
                key = {new_masks, new_diags};
            }
            int index;
            bool is_new_state;
            if (auto search = state_map.find(key); search != state_map.end()) {
                index = search->second;
                is_new_state = false;
                // A state already known to be a dead end stays one.
                if (dead_states.contains(index)) {
                    continue;
                }
            } else {
                index = states.size();
                state_map[key] = index;
                states.push_back({});
                is_new_state = true;
            }

            // Only expand a state the first time it is reached. Expanding it
            // again would append a second copy of the same transitions, once
            // per path leading to it.
            if (is_new_state) {
                if (is_terminal) {
                    for (int diag_index : new_diags) {
                        states.at(index).push_back({
                            .next_state = diag_index,
                            .particle1 = 0,
                            .particle2 = 0,
                            .mass_index = 0,
                            .massive_in = false,
                            .massive_out1 = false,
                            .massive_out2 = false,
                            .is_qcd = false,
                            .is_jet1 = false,
                            .is_jet2 = false,
                            .trace_mode = trace_first,
                        });
                    }
                } else {
                    find_clusterings(
                        ctx, new_masks, new_diags, states, state_map,
                        dead_states, index
                    );
                }
                // The expansion may have found nothing: every clustering left
                // is unsupported by the remaining diagrams. Drop the
                // transition rather than pointing it at a state the walk
                // cannot leave.
                if (states.at(index).size() == 0) {
                    dead_states.insert(index);
                    continue;
                }
            }

            states.at(prev_index)
                .push_back(make_state_item(ctx, index, mask, mask_i, mask_j));
        }
    }
}

// Record the properties of the line a mask stands for. Different diagrams can
// put different propagators on the same partition of external legs (a photon
// and a Z, say); keep the one with a width, since that is the one the
// resonance test is about.
void set_mask_meta(std::vector<LineMeta>& mask_meta, int mask, LineMeta meta) {
    auto& current = mask_meta.at(mask);
    if (!current.known || (current.width <= 0. && meta.width > 0.)) {
        meta.known = true;
        current = meta;
    }
}

} // namespace

MLMClustering::MLMClustering(
    std::vector<Topology> topologies,
    nested_vector3<std::size_t> permutations,
    nested_vector2<std::size_t> diagram_indices,
    double cm_energy,
    JetScaleScheme jet_scale_scheme,
    std::unordered_map<int, int> pdg_color_types,
    double bw_cutoff,
    double jet_radius,
    bool hadronic,
    std::vector<int> external_pdg_ids,
    int max_jet_flavor
) :
    FunctionGenerator(
        "MLMClustering",
        {{"momenta",
          batch_four_vec_array(topologies.at(0).outgoing_masses().size() + 2)}},
        {{"ren_scale", batch_float},
         {"fact_scale1", batch_float},
         {"fact_scale2", batch_float},
         {"outgoing_scales",
          batch_float_array(topologies.at(0).outgoing_masses().size())},
         {"diagram_index", batch_int}}
    ),
    _cm_energy(cm_energy),
    _jet_scale_scheme(jet_scale_scheme),
    _bw_cutoff(bw_cutoff),
    _jet_radius(jet_radius),
    _hadronic(hadronic) {
    std::size_t n_ext = topologies.at(0).outgoing_masses().size() + 2;
    if (n_ext > n_ext_max) {
        throw std::invalid_argument(std::format(
            "MLM clustering supports at most {} external particles, got {}",
            n_ext_max,
            n_ext
        ));
    }
    bool have_pdg_ids = external_pdg_ids.size() > 0;
    if (have_pdg_ids && external_pdg_ids.size() != n_ext) {
        throw std::invalid_argument(std::format(
            "expected {} external pdg ids, got {}", n_ext, external_pdg_ids.size()
        ));
    }

    nested_vector2<int> valid_diags(1 << n_ext);
    std::vector<LineMeta> mask_meta(1 << n_ext);
    std::vector<int> particle_masks;
    std::vector<int> all_diags;

    // The kernel indexes momenta and external_masses by external leg, while a
    // topology's masses are indexed by topology slot, so the permutation has to
    // be applied. Start from a sentinel so the first diagram fills the table and
    // every later one is checked against it.
    _external_masses.assign(n_ext, -1.);

    // create a list of all diagram indices that are possible for a given clustering,
    // where a binary encoding of the clustering is used
    for (auto [topo, permutations, diag_indices] :
         zip(topologies, permutations, diagram_indices)) {
        auto& incoming_masses = topo.incoming_masses();
        auto& outgoing_masses = topo.outgoing_masses();
        for (auto [permutation, diag_index] : zip(permutations, diag_indices)) {
            all_diags.push_back(diag_index);
            particle_masks.assign(topo.decays().size(), 0);
            for (std::size_t i = 2; i < permutation.size(); ++i) {
                particle_masks.at(topo.outgoing_indices().at(permutation.at(i) - 2)) = 1
                    << i;
            }

            // permutation maps external leg -> topology slot, so the mass of leg
            // i is the mass of the slot it is mapped to.
            for (std::size_t leg = 0; leg < n_ext; ++leg) {
                double mass = leg < 2
                    ? incoming_masses.at(leg)
                    : outgoing_masses.at(permutation.at(leg) - 2);
                double& stored = _external_masses.at(leg);
                if (stored < 0.) {
                    stored = mass;
                } else if (stored != mass) {
                    // The kernel keeps a single external mass per leg, so a
                    // permutation that moves a mass between legs would silently
                    // give the wrong clustering measure.
                    throw std::invalid_argument(std::format(
                        "MLM clustering needs the mass of external leg {} to be the "
                        "same for every diagram and permutation, got {} and {}",
                        leg,
                        stored,
                        mass
                    ));
                }
                set_mask_meta(
                    mask_meta,
                    1 << leg,
                    {.mass = mass,
                     .width = 0.,
                     .pdg_id = have_pdg_ids ? external_pdg_ids.at(leg) : 0}
                );
            }

            bool has_t_channel = topo.t_integration_order().size() > 0;
            for (auto& decay : std::views::reverse(topo.decays())) {
                if (decay.child_indices.size() == 0) {
                    continue;
                }
                if (decay.index == 0 && has_t_channel) {
                    continue;
                }
                if (decay.child_indices.size() > 2) {
                    throw std::logic_error("does not support 1->n decays with n > 2");
                }
                int mask =
                    (particle_masks.at(decay.child_indices.at(0)) |
                     particle_masks.at(decay.child_indices.at(1)));
                particle_masks.at(decay.index) = mask;
                valid_diags.at(mask).push_back(diag_index);
                set_mask_meta(
                    mask_meta,
                    mask,
                    {.mass = decay.mass, .width = decay.width, .pdg_id = decay.pdg_id}
                );
            }

            if (!has_t_channel) {
                continue;
            }

            // for the t-channel part, one of the initial state particles has to be
            // involved in the clustering. The k-th mask accumulated from either
            // beam is the k-th t-channel propagator along the chain, which is
            // walked starting from beam 2.
            auto& t_masses = topo.t_propagator_masses();
            auto& t_pdg_ids = topo.t_propagator_pdg_ids();
            std::size_t t_count = t_masses.size();
            auto& t_children = topo.decays().at(0).child_indices;
            std::size_t k = 0;
            for (int mask = 1; std::size_t index : t_children) {
                mask |= particle_masks.at(index);
                valid_diags.at(mask).push_back(diag_index);
                if (k < t_count) {
                    // width stays 0: a spacelike propagator is never resonant
                    set_mask_meta(
                        mask_meta,
                        mask,
                        {.mass = t_masses.at(t_count - 1 - k),
                         .width = 0.,
                         .pdg_id = t_pdg_ids.at(t_count - 1 - k)}
                    );
                }
                ++k;
            }
            k = 0;
            for (int mask = 2; std::size_t index : std::views::reverse(t_children)) {
                mask |= particle_masks.at(index);
                valid_diags.at(mask).push_back(diag_index);
                if (k < t_count) {
                    set_mask_meta(
                        mask_meta,
                        mask,
                        {.mass = t_masses.at(k),
                         .width = 0.,
                         .pdg_id = t_pdg_ids.at(k)}
                    );
                }
                ++k;
            }
        }
    }

    std::vector<int> masks;
    masks.reserve(n_ext);
    for (int i = 0; i < n_ext; ++i) {
        masks.push_back(1 << i);
    }
    nested_vector2<StateItem> states{{}};
    std::map<StateKey, int> state_map;
    std::map<std::pair<double, double>, int> bw_indices;
    state_map[{{masks}, {all_diags}}] = 0;
    CompileContext ctx{
        .valid_diags = valid_diags,
        .mask_meta = mask_meta,
        .bw_masses = _bw_masses,
        .bw_widths = _bw_widths,
        .bw_indices = bw_indices,
        .pdg_color_types = pdg_color_types,
        .have_pdg_ids = have_pdg_ids,
        .max_jet_flavor = max_jet_flavor,
    };
    std::set<int> dead_states;
    find_clusterings(ctx, masks, all_diags, states, state_map, dead_states, 0);

    if (states.at(0).size() == 0) {
        throw std::logic_error(
            "MLM clustering found no valid clustering for this process"
        );
    }
    for (std::size_t i = 0; auto& state : states) {
        // Dead ends are unreachable: no transition points at them any more, but
        // they still occupy a slot in `states` so that the offsets stay stable.
        if (state.size() == 0 && !dead_states.contains(static_cast<int>(i))) {
            throw std::logic_error(
                "MLM clustering reached a state with no valid clustering left"
            );
        }
        ++i;
    }

    // Layout: a clustering state is a run of (data, next_offset, trace_data)
    // triples ending at the triple whose data has the is_last bit set; a
    // terminal state is a count followed by that many diagram indices.
    std::vector<int> first_indices;
    first_indices.reserve(states.size());
    for (int offset = 0; auto& state : states) {
        first_indices.push_back(offset);
        if (state.size() == 0) {
            // a dropped dead end: nothing points at it, so it takes no space
        } else if (state.at(0).particle1 == 0 && state.at(0).particle2 == 0) {
            offset += 1 + state.size();
        } else {
            offset += state_machine_item_size * state.size();
        }
    }
    for (auto& state : states) {
        if (state.size() == 0) {
            continue;
        }
        if (state.at(0).particle1 == 0 && state.at(0).particle2 == 0) {
            _cluster_state_machine.push_back(state.size());
            for (auto& item : state) {
                _cluster_state_machine.push_back(item.next_state);
            }
        } else {
            for (auto& item : state) {
                _cluster_state_machine.push_back(
                    (item.particle1 << 0) + (item.particle2 << 8) +
                    (item.mass_index << 16) + (item.massive_in << 24) +
                    (item.massive_out1 << 25) + (item.massive_out2 << 26) +
                    (item.is_qcd << 27) + (item.is_jet1 << 28) + (item.is_jet2 << 29) +
                    ((&item == &state.back()) << 30)
                );
                _cluster_state_machine.push_back(first_indices.at(item.next_state));
                _cluster_state_machine.push_back(static_cast<int>(item.trace_mode));
            }
        }
    }
}

NamedVector<Value> MLMClustering::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    std::array<Value, 5> mlm_out;
    Value random = fb.squeeze(fb.random(fb.batch_size(args.values()), 1));
    if (_hadronic) {
        mlm_out = fb.mlm_clustering_hadronic(
            args.at(0),
            random,
            _cluster_state_machine,
            _external_masses,
            _bw_masses,
            _bw_widths,
            _bw_cutoff,
            _jet_radius,
            _cm_energy,
            static_cast<me_int_t>(_jet_scale_scheme)
        );
    } else {
        mlm_out = fb.mlm_clustering_leptonic(
            args.at(0),
            random,
            _cluster_state_machine,
            _external_masses,
            _bw_masses,
            _bw_widths,
            _bw_cutoff,
            _jet_radius,
            _cm_energy,
            static_cast<me_int_t>(_jet_scale_scheme)
        );
    }
    return {return_types().keys(), {mlm_out.begin(), mlm_out.end()}};
}

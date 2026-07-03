#include "madspace/phasespace/mlm_clustering.hpp"

#include <bit>
#include <map>

using namespace madspace;

namespace {

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
};

void find_clusterings(
    const std::vector<int>& particle_masks,
    const std::vector<int>& diagrams,
    const nested_vector2<int>& valid_diags,
    nested_vector2<StateItem>& states,
    std::map<StateKey, int>& state_map,
    int prev_index
) {
    int n_masks = particle_masks.size();
    for (int i = 0; i < n_masks - 1; ++i) {
        for (int j = i + 1; j < n_masks; ++j) {
            int mask_i = particle_masks.at(i), mask_j = particle_masks.at(j);
            int mask = mask_i | mask_j;
            auto& valid = valid_diags.at(mask);
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

            StateKey key;
            if (new_masks.size() == 3) {
                key = {{}, new_diags};
            } else {
                key = {new_masks, new_diags};
            }
            int index;
            if (auto search = state_map.find(key); search != state_map.end()) {
                index = search->second;
            } else {
                index = states.size();
                state_map[key] = index;
                states.push_back({});
            }
            states.at(prev_index)
                .push_back({
                    .next_state = index,
                    .particle1 = std::countr_zero(static_cast<unsigned int>(mask_i)),
                    .particle2 = std::countr_zero(static_cast<unsigned int>(mask_j)),
                    .mass_index = 0,
                    .massive_in = false,
                    .massive_out1 = false,
                    .massive_out2 = false,
                    .is_qcd = true,
                    .is_jet1 = true,
                    .is_jet2 = true,
                });

            if (new_masks.size() == 3) {
                auto& current_states = states.at(index);
                for (int diag_index : new_diags) {
                    current_states.push_back({
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
                    });
                }
            } else {
                find_clusterings(
                    new_masks, new_diags, valid_diags, states, state_map, index
                );
            }
        }
    }
}

} // namespace

MLMClustering::MLMClustering(
    std::vector<Topology> topologies,
    nested_vector3<std::size_t> permutations,
    nested_vector2<std::size_t> diagram_indices,
    double bw_cutoff,
    double jet_radius,
    bool hadronic
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
    _bw_cutoff(bw_cutoff),
    _jet_radius(jet_radius),
    _hadronic(hadronic) {
    auto& incoming_masses = topologies.at(0).incoming_masses();
    auto& outgoing_masses = topologies.at(0).outgoing_masses();
    std::size_t n_ext = outgoing_masses.size() + 2;
    nested_vector2<int> valid_diags(1 << n_ext);
    std::vector<int> particle_masks;
    std::vector<int> all_diags;

    _external_masses.insert(
        _external_masses.end(), incoming_masses.begin(), incoming_masses.end()
    );
    _external_masses.insert(
        _external_masses.end(), outgoing_masses.begin(), outgoing_masses.end()
    );

    // create a list of all diagram indices that are possible for a given clustering,
    // where a binary encoding of the clustering is used
    for (auto [topo, permutations, diag_indices] :
         zip(topologies, permutations, diagram_indices)) {
        for (auto [permutation, diag_index] : zip(permutations, diag_indices)) {
            all_diags.push_back(diag_index);
            particle_masks.assign(topo.decays().size(), 0);
            for (std::size_t i = 2; i < permutation.size(); ++i) {
                particle_masks.at(topo.outgoing_indices().at(permutation.at(i) - 2)) = 1
                    << i;
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
            }

            if (!has_t_channel) {
                continue;
            }

            // for the t-channel part, one of the initial state particles has to be
            // involved in the clustering
            for (int mask = 1; std::size_t index : topo.decays().at(0).child_indices) {
                mask |= particle_masks.at(index);
                valid_diags.at(mask).push_back(diag_index);
            }
            for (int mask = 2;
                 std::size_t index :
                 std::views::reverse(topo.decays().at(0).child_indices)) {
                mask |= particle_masks.at(index);
                valid_diags.at(mask).push_back(diag_index);
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
    state_map[{{masks}, {all_diags}}] = 0;
    find_clusterings(masks, all_diags, valid_diags, states, state_map, 0);

    std::vector<int> first_indices;
    first_indices.reserve(states.size());
    for (int offset = 0; auto& state : states) {
        first_indices.push_back(offset);
        if (state.at(0).particle1 == 0 && state.at(0).particle2 == 0) {
            offset += 1 + state.size();
        } else {
            offset += 2 * state.size();
        }
    }
    for (auto& state : states) {
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
            _jet_radius
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
            _jet_radius
        );
    }
    return {return_types().keys(), {mlm_out.begin(), mlm_out.end()}};
}

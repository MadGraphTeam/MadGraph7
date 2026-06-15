#include "madspace/phasespace/mlm_clustering.hpp"

using namespace madspace;

MLMClustering::MLMClustering(
    std::vector<Topology> topologies,
    nested_vector3<std::size_t> permutations,
    nested_vector2<std::size_t> diagram_indices
) :
    FunctionGenerator(
        "MLMClustering",
        {{"momenta",
          batch_four_vec_array(topologies.at(0).outgoing_masses().size() + 2)}},
        {{"ren_scale", batch_float},
         {"fact_scale1", batch_float},
         {"fact_scale2", batch_float},
         {"cluster_history",
          batch_int_array(topologies.at(0).outgoing_masses().size() - 1)},
         {"cluster_scales",
          batch_float_array(topologies.at(0).outgoing_masses().size() - 1)}}
    ) {
    std::size_t n_ext = topologies.at(0).outgoing_masses().size() + 2;
    nested_vector2<std::size_t> valid_diags(1 << n_ext);
    std::vector<std::size_t> particle_masks;
    std::vector<std::size_t> all_diags;

    // create a list of all diagram indices that are possible for a given clustering,
    // where a binary encoding of the clustering is used
    for (auto [topo, permutations, diag_indices] :
         zip(topologies, permutations, diagram_indices)) {
        for (auto [permutation, diag_index] : zip(permutations, diag_indices)) {
            all_diags.push_back(diag_index);
            particle_masks.assign(topo.decays().size(), 0);
            for (std::size_t i = 2; i < permutation.size(); ++i) {
                particle_masks.at(topo.outgoing_indices().at(permutation.at(i))) = 1
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
                std::size_t mask =
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
            for (std::size_t mask = 1;
                 std::size_t index : topo.decays().at(0).child_indices) {
                mask |= particle_masks.at(index);
                valid_diags.at(mask).push_back(diag_index);
            }
            for (std::size_t mask = 2;
                 std::size_t index :
                 std::views::reverse(topo.decays().at(0).child_indices)) {
                mask |= particle_masks.at(index);
                valid_diags.at(mask).push_back(diag_index);
            }
        }
    }
}

NamedVector<Value> MLMClustering::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    auto mlm_out = fb.mlm_clustering_hadronic(args.at(0), _cluster_state_machine);
    return {return_types().keys(), {mlm_out.begin(), mlm_out.end()}};
}

#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/topology.hpp"

namespace madspace {

class MLMClustering : public FunctionGenerator {
public:
    MLMClustering(
        std::vector<Topology> topologies,
        nested_vector3<std::size_t> permutations,
        nested_vector2<std::size_t> diagram_indices,
        double bw_cutoff = 15,
        double jet_radius = 0.4,
        bool hadronic = true,
        // pdg ids of the external particles, in leg order. When empty, every
        // clustering is assumed to be a QCD splitting between jets.
        std::vector<int> external_pdg_ids = {},
        int max_jet_flavor = 4
    );

    // The compiled clustering state machine, in the flat encoding the kernel
    // walks. Exposed so that its structure can be checked directly.
    const std::vector<me_int_t>& cluster_state_machine() const {
        return _cluster_state_machine;
    }
    const std::vector<double>& external_masses() const { return _external_masses; }
    const std::vector<double>& bw_masses() const { return _bw_masses; }
    const std::vector<double>& bw_widths() const { return _bw_widths; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<me_int_t> _cluster_state_machine;
    std::vector<double> _external_masses;
    std::vector<double> _bw_masses;
    std::vector<double> _bw_widths;
    double _bw_cutoff;
    double _jet_radius;
    bool _hadronic;
};

} // namespace madspace

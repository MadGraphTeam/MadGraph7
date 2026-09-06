#pragma once

#include <unordered_map>

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/topology.hpp"

namespace madspace {

// Number of words each clustering transition occupies in the compiled state
// machine: (data, next_offset, trace_data).
inline constexpr int state_machine_item_size = 3;

// How a clustering scale is assigned to an outgoing jet.
enum class JetScaleScheme {
    // The scale of the vertex at which the leg itself was emitted, i.e. the
    // softest clustering it takes part in.
    emission = 0,
    // The hardest vertex on the parton line the leg belongs to, which is the
    // scale at which that line was produced. This is what madevent writes, and
    // what Pythia uses as the shower starting scale for the leg when MLM
    // merging turns on Beams:setProductionScalesFromLHEF.
    production = 1,
};

class MLMClustering : public FunctionGenerator {
public:
    MLMClustering(
        std::vector<Topology> topologies,
        nested_vector3<std::size_t> permutations,
        nested_vector2<std::size_t> diagram_indices,
        // Collider energy. An outgoing leg that no QCD clustering assigned a
        // scale to is reported at this value rather than at zero, so that an
        // MLM veto can never trip on it.
        double cm_energy,
        JetScaleScheme jet_scale_scheme = JetScaleScheme::production,
        // Signed color representation per pdg id, as exported in the
        // subprocess metadata. Used to follow a parton line through the
        // clustering; falls back to the Standard Model assignment for a pdg id
        // that is not listed.
        std::unordered_map<int, int> pdg_color_types = {},
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
    double _cm_energy;
    JetScaleScheme _jet_scale_scheme;
    double _bw_cutoff;
    double _jet_radius;
    bool _hadronic;
};

} // namespace madspace

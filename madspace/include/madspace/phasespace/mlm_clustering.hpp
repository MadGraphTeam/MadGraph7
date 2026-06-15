#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/topology.hpp"

namespace madspace {

class MLMClustering : public FunctionGenerator {
public:
    MLMClustering(
        std::vector<Topology> topologies,
        nested_vector3<std::size_t> permutations,
        nested_vector2<std::size_t> diagram_indices
    );

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<me_int_t> _cluster_state_machine;
};

} // namespace madspace

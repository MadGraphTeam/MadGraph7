#include "madspace/nlo/cuts.hpp"

#include <cmath>
#include <stdexcept>

#include "madspace/nlo/jets.hpp"

using namespace madspace;

namespace {

// the number of particles of the layouts, checked to be one and the same
std::size_t
layout_particle_count(const nested_vector2<int>& layouts, std::size_t incoming) {
    if (layouts.empty()) {
        throw std::invalid_argument("NLOCuts: no flavor layouts");
    }
    if (incoming != 1 && incoming != 2) {
        throw std::invalid_argument(
            "NLOCuts: the kinematics needs two incoming particles or one (a decay)"
        );
    }
    std::size_t count = layouts.at(0).size();
    for (const auto& layout : layouts) {
        if (layout.size() != count) {
            throw std::invalid_argument(
                "NLOCuts: the flavor layouts have different numbers of particles"
            );
        }
    }
    if (count <= incoming) {
        throw std::invalid_argument(
            "NLOCuts: the kinematics needs an outgoing particle"
        );
    }
    return count;
}

} // namespace

NLOCuts::NLOCuts(
    const nested_vector2<int>& layouts,
    std::size_t incoming,
    std::size_t unresolved,
    double jetalgo,
    double jetradius,
    double ptj,
    double etaj,
    int maxjetflavor,
    double ptl,
    double etal,
    double drll,
    double drll_sf,
    double mll,
    double mll_sf
) :
    FunctionGenerator(
        "NLOCuts",
        NamedVector<Type>{
            {"momenta",
             batch_four_vec_array(
                 static_cast<int>(layout_particle_count(layouts, incoming))
             )},
            {"layout_index", batch_int},
        },
        NamedVector<Type>{{"mask", batch_float}}
    ),
    _particle_count(layout_particle_count(layouts, incoming)),
    _incoming(incoming),
    _unresolved(unresolved),
    _options{
        jetalgo,
        jetradius,
        ptj,
        etaj,
        static_cast<double>(maxjetflavor),
        ptl,
        etal,
        drll,
        drll_sf,
        mll,
        mll_sf,
        static_cast<double>(unresolved)
    } {
    for (const auto& layout : layouts) {
        if (NLOJets::parton_count(layout, incoming, maxjetflavor) > jet_parton_max) {
            throw std::invalid_argument(
                "NLOCuts: more QCD partons than the jet clustering takes"
            );
        }
        _layouts.emplace_back(layout.begin(), layout.end());
    }
}

NamedVector<Value> NLOCuts::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    return {
        {"mask",
         fb.nlo_cuts(
             args.at("momenta"),
             args.at("layout_index"),
             _layouts,
             _options,
             static_cast<me_int_t>(_incoming)
         )}
    };
}

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
    double mll_sf,
    double ptgmin,
    double etagamma,
    double r0gamma,
    double xn,
    double epsgamma,
    bool isoem,
    bool gamma_is_j,
    double rphreco,
    double etaphreco,
    bool lepphreco,
    bool quarkphreco,
    const std::vector<int>& isolated_photons
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
        static_cast<double>(unresolved),
        ptgmin,
        etagamma,
        r0gamma,
        xn,
        epsgamma,
        isoem ? 1. : 0.,
        gamma_is_j ? 1. : 0.,
        rphreco,
        etaphreco,
        lepphreco ? 1. : 0.,
        quarkphreco ? 1. : 0.
    } {
    if (!isolated_photons.empty() && isolated_photons.size() != layouts.size()) {
        throw std::invalid_argument(
            "NLOCuts: the isolated photons need one entry per flavor layout"
        );
    }
    if (_particle_count > particle_max) {
        throw std::invalid_argument(
            "NLOCuts: more particles than the photon cuts take"
        );
    }
    for (std::size_t c = 0; c < layouts.size(); ++c) {
        const auto& layout = layouts.at(c);
        // the photons take part in the jets where they are not isolated
        std::size_t photons = 0;
        for (std::size_t i = incoming; i < layout.size(); ++i) {
            photons += layout.at(i) == 22 ? 1 : 0;
        }
        if (NLOJets::parton_count(layout, incoming, maxjetflavor) + photons >
            jet_parton_max) {
            throw std::invalid_argument(
                "NLOCuts: more QCD partons and photons than the jet clustering takes"
            );
        }
        _layouts.emplace_back(layout.begin(), layout.end());
        _isolated_photons.push_back(
            isolated_photons.empty() ? -1 : isolated_photons.at(c)
        );
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
             _isolated_photons,
             _options,
             static_cast<me_int_t>(_incoming)
         )}
    };
}

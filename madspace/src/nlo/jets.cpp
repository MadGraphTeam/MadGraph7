#include "madspace/nlo/jets.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

using namespace madspace;

namespace {

// whether a particle of a layout is a QCD parton of its jets, a final-state
// quark up to maxjetflavor or a gluon, as cuts.f takes them
bool is_parton(int pdg, std::size_t index, std::size_t incoming, int maxjetflavor) {
    int flavor = std::abs(pdg);
    return index >= incoming &&
        ((flavor >= 1 && flavor <= maxjetflavor) || flavor == 21);
}

} // namespace

std::size_t NLOJets::parton_count(
    const std::vector<int>& layout, std::size_t incoming, int maxjetflavor
) {
    std::size_t partons = 0;
    for (std::size_t i = 0; i < layout.size(); ++i) {
        partons += is_parton(layout.at(i), i, incoming, maxjetflavor);
    }
    return partons;
}

NLOJets::Layout NLOJets::layout(
    const nested_vector2<int>& layouts,
    std::size_t jet_slots,
    std::size_t incoming,
    int maxjetflavor
) {
    if (layouts.empty()) {
        throw std::invalid_argument("NLOJets: no flavor layouts");
    }
    if (incoming != 1 && incoming != 2) {
        throw std::invalid_argument(
            "NLOJets: the kinematics needs two incoming particles or one (a decay)"
        );
    }
    std::size_t count = layouts.at(0).size();
    if (count <= incoming) {
        throw std::invalid_argument(
            "NLOJets: the kinematics needs an outgoing particle"
        );
    }
    Layout result{count, jet_slots, {}, {}};
    // the particles outside the partons, the same final-state ones for every
    // layout
    std::vector<int> kept;
    std::size_t most = 0;
    for (std::size_t c = 0; c < layouts.size(); ++c) {
        const auto& pdgs = layouts.at(c);
        if (pdgs.size() != count) {
            throw std::invalid_argument(
                "NLOJets: the flavor layouts have different numbers of particles"
            );
        }
        most = std::max(most, parton_count(pdgs, incoming, maxjetflavor));
        std::vector<me_int_t> parton;
        std::vector<int> others;
        for (std::size_t i = 0; i < pdgs.size(); ++i) {
            bool p = is_parton(pdgs.at(i), i, incoming, maxjetflavor);
            parton.push_back(p ? 1 : 0);
            if (!p) {
                others.push_back(pdgs.at(i));
            }
        }
        if (c == 0) {
            kept = others;
        } else if (others.size() != kept.size() ||
                   !std::equal(
                       others.begin() + incoming, others.end(), kept.begin() + incoming
                   )) {
            throw std::invalid_argument(
                "NLOJets: the flavor layouts differ outside their QCD partons"
            );
        }
        result.parton.push_back(parton);
    }
    if (jet_slots == 0) {
        result.slot_count = most;
    } else if (jet_slots < most) {
        throw std::invalid_argument("NLOJets: fewer jet slots than QCD partons");
    }
    if (result.slot_count > parton_max) {
        throw std::invalid_argument(
            "NLOJets: more QCD partons than the jet clustering takes"
        );
    }
    // the first layout outside its partons, then the jets
    result.pids = kept;
    result.pids.insert(result.pids.end(), result.slot_count, 21);
    return result;
}

NLOJets::NLOJets(
    const nested_vector2<int>& layouts,
    std::size_t jet_slots,
    std::size_t incoming,
    double jetalgo,
    double jetradius,
    double ptj,
    double etaj,
    int maxjetflavor
) :
    NLOJets(
        layout(layouts, jet_slots, incoming, maxjetflavor),
        incoming,
        {jetalgo, jetradius, ptj, etaj}
    ) {}

NLOJets::NLOJets(
    Layout layout, std::size_t incoming, const std::vector<double>& options
) :
    FunctionGenerator(
        "NLOJets",
        {
            {"momenta", batch_four_vec_array(static_cast<int>(layout.particle_count))},
            {"layout_index", batch_int},
        },
        {
            {"jet_momenta", batch_four_vec_array(static_cast<int>(layout.pids.size()))},
            {"jet_count", batch_int},
        }
    ),
    _particle_count(layout.particle_count),
    _slot_count(layout.slot_count),
    _incoming(incoming),
    _pids(layout.pids),
    _parton(layout.parton),
    _options(options) {
    if (!(options.at(1) > 0.)) {
        throw std::invalid_argument("NLOJets: the jet radius must be positive");
    }
}

NamedVector<Value> NLOJets::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    auto [jet_momenta, jet_count] = fb.nlo_jets(
        args.at("momenta"),
        args.at("layout_index"),
        _parton,
        _options,
        static_cast<me_int_t>(_pids.size())
    );
    return {{"jet_momenta", jet_momenta}, {"jet_count", jet_count}};
}

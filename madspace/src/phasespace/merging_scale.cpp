#include "madspace/phasespace/merging_scale.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

using namespace madspace;

namespace {

// madevent's is_a_j or is_a_b: a gluon, or a quark up to the b (or the top if
// max_jet_flavor is 6)
bool is_massless_parton(int pdg_id, int max_jet_flavor) {
    int id = std::abs(pdg_id);
    return id == 21 || (id >= 1 && id <= std::max(5, std::min(max_jet_flavor, 6)));
}

} // namespace

MergingScale::MergingScale(
    Measure measure,
    const std::vector<int>& pdg_ids,
    const std::vector<bool>& merging_legs,
    double d_parameter,
    bool hadronic1,
    bool hadronic2,
    int max_jet_flavor
) :
    FunctionGenerator(
        "MergingScale",
        {{"momenta", batch_four_vec_array(pdg_ids.size())}},
        {{"scale", batch_float}}
    ),
    _measure(measure),
    _pdg_ids(pdg_ids.begin(), pdg_ids.end()),
    _merging_legs(pdg_ids.size(), 0),
    _d_parameter(d_parameter),
    _beam_flags((hadronic1 ? 1 : 0) | (hadronic2 ? 2 : 0)),
    _mode(0) {
    if (pdg_ids.size() < 3 || merging_legs.size() != pdg_ids.size()) {
        throw std::invalid_argument(
            "MergingScale needs one pdg id and one merging flag per external leg"
        );
    }
    if (measure == kt_durham && !(d_parameter > 0.)) {
        throw std::invalid_argument("MergingScale: the D parameter must be positive");
    }
    std::size_t merging_count = 0, massless_count = 0;
    for (std::size_t i = 2; i < pdg_ids.size(); ++i) {
        if (!merging_legs.at(i)) {
            continue;
        }
        _merging_legs.at(i) = 1;
        ++merging_count;
        if (is_massless_parton(pdg_ids.at(i), max_jet_flavor)) {
            ++massless_count;
        }
    }
    if (merging_count == 0 || massless_count == 0) {
        _mode = 0;
    } else if (measure == pt_lund && massless_count == 2 && merging_count == 2 &&
               pdg_ids.size() == 4) {
        _mode = (hadronic1 || hadronic2) ? 1 : 0;
    } else {
        _mode = 2;
    }
}

std::vector<std::size_t> MergingScale::merging_leg_indices() const {
    std::vector<std::size_t> indices;
    for (std::size_t i = 0; i < _merging_legs.size(); ++i) {
        if (_merging_legs.at(i)) {
            indices.push_back(i);
        }
    }
    return indices;
}

bool MergingScale::bounds_pt() const {
    return _measure == kt_durham && _mode == 2 && _beam_flags != 0;
}

NamedVector<Value> MergingScale::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    Value momenta = args.at(0);
    if (_measure == kt_durham) {
        return {
            {"scale",
             fb.merging_scale_kt_durham(
                 momenta, _merging_legs, _d_parameter, _beam_flags, _mode
             )}
        };
    } else {
        return {
            {"scale",
             fb.merging_scale_pt_lund(
                 momenta, _merging_legs, _pdg_ids, _beam_flags, _mode
             )}
        };
    }
}

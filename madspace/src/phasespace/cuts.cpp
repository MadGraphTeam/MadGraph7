#include "madspace/phasespace/cuts.hpp"

#include "madspace/compgraphs/type.hpp"
#include "madspace/util.hpp"

#include <algorithm>

using namespace madspace;

Cuts::Cuts(const std::vector<CutItem>& cut_data) :
    FunctionGenerator(
        "Cuts", cut_data.at(0).observable.arg_types(), {{"mask", batch_float}}
    ),
    _cut_data(cut_data) {}

Cuts::Cuts(std::size_t particle_count) :
    FunctionGenerator(
        "Cuts",
        {{"momenta", batch_four_vec_array(particle_count)}},
        {{"mask", batch_float}}
    ) {}

NamedVector<Value>
Cuts::build_function_impl(FunctionBuilder& fb, const NamedVector<Value>& args) const {
    ValueVec weights;
    for (auto& item : _cut_data) {
        if (item.observable.not_found()) {
            continue;
        }
        Value obs = item.observable.build_function(fb, args).at(0);
        if (obs.type.shape.size() == 0) {
            weights.push_back(fb.cut_one(obs, item.min, item.max));
        } else if (item.mode == CutMode::all) {
            weights.push_back(fb.cut_all(obs, item.min, item.max));
        } else {
            weights.push_back(fb.cut_any(obs, item.min, item.max));
        }
    }
    return {{"mask", fb.product(weights)}};
}

std::vector<std::string> Cuts::non_mirror_invariant_cuts() const {
    std::vector<std::string> names;
    for (auto& item : _cut_data) {
        if (item.observable.mirror_invariant()) {
            continue;
        }
        auto name = item.observable.name();
        names.push_back(name.empty() ? "(unnamed cut)" : name);
    }
    return names;
}

namespace {

// The accessors below hand bounds to the phase-space mappings, which apply each
// one as a hard floor on a fixed particle or pair. That is only right for a
// bound every selected object has to satisfy: an ordered selection ("the
// leading jet") names a rank rather than a particle, and with CutMode::any a
// single object passing is enough, so neither says anything about a given
// particle unless the selection holds exactly one. Such cuts stay filters.
bool binds_each_object(const Cuts::CutItem& item, std::size_t object_count) {
    return !item.observable.ordered() &&
        (item.mode == Cuts::CutMode::all || object_count == 1);
}

} // namespace

double Cuts::sqrt_s_min() const {
    double sqrt_s_min = 0.;
    for (auto& item : _cut_data) {
        if (item.observable.observable() == Observable::obs_sqrt_s &&
            sqrt_s_min < item.min) {
            sqrt_s_min = item.min;
        }
    }
    return sqrt_s_min;
}

std::vector<double> Cuts::eta_max() const {
    std::vector<double> eta_max(
        arg_types().at(0).shape.at(0) - 2, std::numeric_limits<double>::infinity()
    );
    for (auto& item : _cut_data) {
        double item_max = std::numeric_limits<double>::infinity();
        if (item.observable.observable() == Observable::obs_eta_abs) {
            item_max = item.max;
        } else if (item.observable.observable() == Observable::obs_eta) {
            item_max = std::max(-item.min, item.max);
        } else {
            continue;
        }
        auto indices = item.observable.simple_observable_indices();
        if (!binds_each_object(item, indices.size())) {
            continue;
        }
        for (std::size_t index : indices) {
            if (index < 2) {
                continue;
            }
            double& limit = eta_max.at(index - 2);
            if (limit > item_max) {
                limit = item_max;
            }
        }
    }
    return eta_max;
}

std::vector<double> Cuts::pt_min() const {
    std::vector<double> pt_min(arg_types().at(0).shape.at(0) - 2, 0.);
    for (auto& item : _cut_data) {
        if (item.observable.observable() != Observable::obs_pt) {
            continue;
        }
        auto indices = item.observable.simple_observable_indices();
        if (!binds_each_object(item, indices.size())) {
            continue;
        }
        for (std::size_t index : indices) {
            if (index < 2) {
                continue;
            }
            double& limit = pt_min.at(index - 2);
            if (limit < item.min) {
                limit = item.min;
            }
        }
    }
    return pt_min;
}

namespace {

using PairList = std::vector<std::pair<std::size_t, std::size_t>>;

// The pairs, in the observable's own (full, beams included) indexing, whose
// invariant mass an observable measures: the genuine pair observables, and
// "mass" of summed momenta when that sum is a pair. With one group
// ("lepton-sum-mass") the sum is a pair only when the group holds exactly two
// particles; with two groups ("jet-jet-sum-mass") it is one pair per
// combination the groups can form. Empty for anything else.
PairList mass_pairs(const Observable& o) {
    PairList pairs;
    const auto& idx = o.indices();
    switch (o.observable()) {
    case Observable::obs_mass:
        if (!o.sum_momenta()) {
            break;
        }
        if (idx.size() == 1 && idx.at(0).size() == 2) {
            pairs.emplace_back(idx.at(0).at(0), idx.at(0).at(1));
            break;
        }
        [[fallthrough]];
    case Observable::obs_pair_mass:
    case Observable::obs_sfos_pair_mass:
        if (idx.size() == 2) {
            for (std::size_t k = 0; k < idx.at(0).size(); ++k) {
                pairs.emplace_back(idx.at(0).at(k), idx.at(1).at(k));
            }
        }
        break;
    default:
        break;
    }
    return pairs;
}

PairList delta_r_pairs(const Observable& o) {
    PairList pairs;
    const auto& idx = o.indices();
    if (o.observable() == Observable::obs_delta_r && idx.size() == 2) {
        for (std::size_t k = 0; k < idx.at(0).size(); ++k) {
            pairs.emplace_back(idx.at(0).at(k), idx.at(1).at(k));
        }
    }
    return pairs;
}

} // namespace

std::vector<std::vector<double>> Cuts::pairwise_min(
    const std::function<PairList(const Observable&)>& pairs
) const {
    std::size_t n = arg_types().at(0).shape.at(0) - 2;
    std::vector<std::vector<double>> out(n, std::vector<double>(n, 0.));
    for (auto& item : _cut_data) {
        if (item.observable.not_found()) {
            continue;
        }
        auto item_pairs = pairs(item.observable);
        if (item_pairs.empty() || !binds_each_object(item, item_pairs.size())) {
            continue;
        }
        for (auto [i, j] : item_pairs) {
            if (i < 2 || j < 2) {
                continue;
            }
            i -= 2;
            j -= 2;
            if (i < n && j < n && item.min > out.at(i).at(j)) {
                out.at(i).at(j) = item.min;
                out.at(j).at(i) = item.min;
            }
        }
    }
    return out;
}

std::vector<std::vector<double>> Cuts::m_inv_min() const {
    return pairwise_min(mass_pairs);
}

std::vector<Cuts::PairMassAny> Cuts::pair_mass_any_min() const {
    // The CutMode::any pair mass cuts that m_inv_min leaves out: one pair
    // passing is enough, so the cut bounds no pair in particular, but it does
    // bound any set of particles holding all of its pairs.
    std::size_t n = arg_types().at(0).shape.at(0) - 2;
    std::vector<PairMassAny> out;
    for (auto& item : _cut_data) {
        if (item.observable.not_found() || item.mode != CutMode::any ||
            item.observable.ordered() || !(item.min > 0.)) {
            continue;
        }
        auto item_pairs = mass_pairs(item.observable);
        if (item_pairs.size() < 2) {
            // nothing, or one pair, which m_inv_min already reports
            continue;
        }
        PairMassAny bound{{}, item.min};
        for (auto [i, j] : item_pairs) {
            if (i < 2 || j < 2 || i - 2 >= n || j - 2 >= n) {
                // a pair the outgoing particles cannot account for could be
                // the one that passes, so the cut bounds nothing
                bound.pairs.clear();
                break;
            }
            bound.pairs.emplace_back(i - 2, j - 2);
        }
        if (!bound.pairs.empty()) {
            out.push_back(std::move(bound));
        }
    }
    return out;
}

std::vector<std::vector<double>> Cuts::dr_min() const {
    return pairwise_min(delta_r_pairs);
}

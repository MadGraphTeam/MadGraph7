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

std::vector<std::vector<double>> Cuts::pairwise_min(
    Observable::ObservableOption obs,
    const std::function<
        std::vector<std::pair<std::size_t, std::size_t>>(const Observable&)>& pairs
) const {
    std::size_t n = arg_types().at(0).shape.at(0) - 2;
    std::vector<std::vector<double>> out(n, std::vector<double>(n, 0.));
    for (auto& item : _cut_data) {
        if (item.observable.observable() != obs) {
            continue;
        }
        auto item_pairs = pairs(item.observable);
        if (!binds_each_object(item, item_pairs.size())) {
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
    // Three ways of asking for the same thing. "mass" with summed momenta is
    // the mass of the summed selection: with one group ("lepton-sum-mass")
    // that is a pair only when the group holds exactly two particles, with two
    // groups ("jet-jet-sum-mass") it is one pair per combination the groups
    // can form. obs_pair_mass is the genuine pairwise cut and covers every
    // pair a group can form.
    auto summed = pairwise_min(Observable::obs_mass, [](const Observable& o) {
        std::vector<std::pair<std::size_t, std::size_t>> pairs;
        const auto& idx = o.indices();
        if (!o.sum_momenta()) {
            return pairs;
        }
        if (idx.size() == 1 && idx.at(0).size() == 2) {
            pairs.emplace_back(idx.at(0).at(0), idx.at(0).at(1));
        } else if (idx.size() == 2) {
            for (std::size_t k = 0; k < idx.at(0).size(); ++k) {
                pairs.emplace_back(idx.at(0).at(k), idx.at(1).at(k));
            }
        }
        return pairs;
    });
    auto pairwise = pairwise_min(Observable::obs_pair_mass, [](const Observable& o) {
        std::vector<std::pair<std::size_t, std::size_t>> pairs;
        const auto& idx = o.indices();
        if (idx.size() == 2) {
            for (std::size_t k = 0; k < idx.at(0).size(); ++k) {
                pairs.emplace_back(idx.at(0).at(k), idx.at(1).at(k));
            }
        }
        return pairs;
    });
    for (auto [row_summed, row_pairwise] : zip(summed, pairwise)) {
        for (auto [a, b] : zip(row_summed, row_pairwise)) {
            a = std::max(a, b);
        }
    }
    return summed;
}

std::vector<std::vector<double>> Cuts::dr_min() const {
    return pairwise_min(Observable::obs_delta_r, [](const Observable& o) {
        std::vector<std::pair<std::size_t, std::size_t>> pairs;
        const auto& idx = o.indices();
        if (idx.size() == 2) {
            for (std::size_t k = 0; k < idx.at(0).size(); ++k) {
                pairs.emplace_back(idx.at(0).at(k), idx.at(1).at(k));
            }
        }
        return pairs;
    });
}

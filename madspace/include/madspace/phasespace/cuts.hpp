#pragma once

#include "madspace/compgraphs.hpp"
#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/observable.hpp"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace madspace {

class Cuts : public FunctionGenerator {
public:
    enum CutMode { any, all };
    struct CutItem {
        Observable observable;
        double min = -std::numeric_limits<double>::infinity();
        double max = std::numeric_limits<double>::infinity();
        CutMode mode = CutMode::all;
    };

    Cuts(const std::vector<CutItem>& cut_data);
    Cuts(std::size_t particle_count);
    // Names of the configured cuts that are not invariant under the
    // initial-state mirror (py, pz -> -py, -pz); empty if every cut is.
    // Mirroring an accepted event after the cuts only reproduces the mirrored
    // half of the initial state if the cuts cannot tell the two orientations
    // apart, since the event that gets written is the mirrored one.
    std::vector<std::string> non_mirror_invariant_cuts() const;
    bool mirror_invariant() const { return non_mirror_invariant_cuts().empty(); }
    double sqrt_s_min() const;
    std::vector<double> eta_max() const;
    std::vector<double> pt_min() const;
    std::vector<std::vector<double>> m_inv_min() const;
    std::vector<std::vector<double>> dr_min() const;

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<std::vector<double>> pairwise_min(
        Observable::ObservableOption obs,
        const std::function<
            std::vector<std::pair<std::size_t, std::size_t>>(const Observable&)>& pairs
    ) const;

    std::vector<CutItem> _cut_data;
};

} // namespace madspace

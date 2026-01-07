#pragma once

#include "madevent/madcode.h"
#include "madevent/phasespace/base.h"
#include "madevent/phasespace/observable.h"

#include <vector>

namespace madevent {

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
    double sqrt_s_min() const;
    std::vector<double> eta_max() const;
    std::vector<double> pt_min() const;

private:
    ValueVec
    build_function_impl(FunctionBuilder& fb, const ValueVec& args) const override;

    std::vector<CutItem> _cut_data;
};

} // namespace madevent

#pragma once

#include "madspace/phasespace/base.h"
#include "madspace/phasespace/observable.h"

namespace madspace {

class ObservableHistograms : public FunctionGenerator {
public:
    struct HistItem {
        Observable observable;
        double min;
        double max;
        std::size_t bin_count;
    };
    ObservableHistograms(const std::vector<HistItem>& observables);
    const std::vector<HistItem>& observables() const { return _observables; }

private:
    ValueVec
    build_function_impl(FunctionBuilder& fb, const ValueVec& args) const override;

    std::vector<HistItem> _observables;
};

} // namespace madspace

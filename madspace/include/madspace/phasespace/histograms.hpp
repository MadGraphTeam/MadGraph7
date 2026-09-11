#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/observable.hpp"

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
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<HistItem> _observables;
};

// Evaluates a list of observables (no binning): one batch_float output per
// observable, used to histogram the final event sample at combine time.
class ObservableValues : public FunctionGenerator {
public:
    ObservableValues(const std::vector<Observable>& observables);
    const std::vector<Observable>& observables() const { return _observables; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<Observable> _observables;
};

} // namespace madspace

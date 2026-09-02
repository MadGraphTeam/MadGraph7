#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/observable.hpp"

namespace madspace {

/**
 * Weighted histograms filled during event generation.
 *
 * Evaluates one @ref Observable per @ref HistItem and accumulates the event
 * weight and its square into a fixed binning [1]. For a fixed number of
 * integrand evaluations these carry smaller statistical uncertainties than
 * histograms filled from unweighted events. On GPUs only the binned data is
 * copied to the host (Sec. 3.2.5 of [1]).
 *
 * `batch` is the leading batch dimension. `i` indexes the histograms. Each
 * histogram has `bin_count + 2` entries: one underflow and one overflow bin.
 *
 * **Arguments**
 * - `weight` – `float`, shape `(batch,)` – the per-event weight.
 * - `momenta` – `float`, shape `(batch, n, 4)` – the event momenta.
 *
 * **Returns**
 * - `values_i` – `float`, shape `(bin_count + 2,)` – summed weight per bin.
 * - `square_values_i` – `float`, shape `(bin_count + 2,)` – summed squared
 *   weight per bin.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 3.2.5)
 */
class ObservableHistograms : public FunctionGenerator {
public:
    /// One histogram: an @ref Observable and its binning.
    struct HistItem {
        /// The observable to bin.
        Observable observable;
        /// Lower edge of the first bin.
        double min;
        /// Upper edge of the last bin.
        double max;
        /// Number of bins between @ref min and @ref max (excluding under/overflow).
        std::size_t bin_count;
    };
    /// @param observables  The histograms to fill.
    ObservableHistograms(const std::vector<HistItem>& observables);
    /// The histogram definitions passed to the constructor.
    const std::vector<HistItem>& observables() const { return _observables; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<HistItem> _observables;
};

} // namespace madspace

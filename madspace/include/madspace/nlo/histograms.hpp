#pragma once

#include "madspace/phasespace/base.hpp"

namespace madspace {

/**
 * The histograms of an NLO integrand at fixed order, with the kinematics of
 * each point (an event and its counter-events) in the same histograms, as the
 * analyses of MadFKS fill them [2] (HwU_fill and HwU_add_points of HwU.f): the
 * weights of the kinematics of one point that fall into one bin add up, and
 * the square of that sum enters the error of the bin. The counterterms then
 * cancel the real emission bin by bin, also in the error, for an observable
 * that is infrared safe. The jets of each kinematics come from @ref NLOJets.
 * The outputs are those of @ref ObservableHistograms, sums
 * over the batch with an underflow and an overflow bin, so that the weighted
 * histograms of the event generator (the `histograms` of `info.json` that
 * MadBoard draws) take them as they are: with @f$N@f$ points the cross
 * section in a bin is the sum over @f$N@f$ and its error
 * @f$\sqrt{(\sum w^2 - (\sum w)^2 / N)} / N@f$. A kinematics with fewer jets
 * than its histogram needs (`jet_count` of @ref HistItem) or an observable
 * that is not finite does not count.
 *
 * `batch` is the leading batch dimension, `i` indexes the histograms and `k`
 * the kinematics.
 *
 * **Arguments**
 * - `weight{k}` – `float`, shape `(batch,)` – the weight of kinematics `k`.
 * - `jet_count{k}` – `int`, shape `(batch,)` – its number of jets.
 * - `value{i}_{k}` – `float`, shape `(batch,)` – the observable of histogram
 *   `i` at kinematics `k`.
 *
 * **Returns**
 * - `values{i}` – `float`, shape `(bin_count + 2,)` – the summed weight per
 *   bin.
 * - `square_values{i}` – `float`, shape `(bin_count + 2,)` – the sum per bin
 *   of the squares of the sums of each point.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 3.2.5)
 * - [2] R. Frederix, S. Frixione, F. Maltoni, T. Stelzer, "Automation of
 *   next-to-leading order computations in QCD: the FKS subtraction",
 *   https://arxiv.org/abs/0908.4272
 */
class NLOHistograms : public FunctionGenerator {
public:
    /// One histogram: its binning and the jets its observable needs.
    struct HistItem {
        /// Lower edge of the first bin.
        double min;
        /// Upper edge of the last bin.
        double max;
        /// Number of bins between @ref min and @ref max (excluding
        /// under/overflow).
        std::size_t bin_count;
        /// Number of jets a kinematics needs for the observable.
        std::size_t jet_count = 0;
    };
    /**
     * @param histograms        The histograms to fill.
     * @param kinematics_count  Number of kinematics of each point.
     * @throws std::invalid_argument without histograms, for a histogram
     *         without bins or with max <= min, or for no or more than 64
     *         kinematics.
     */
    NLOHistograms(
        const std::vector<HistItem>& histograms, std::size_t kinematics_count
    );
    /// The histograms passed to the constructor.
    const std::vector<HistItem>& histograms() const { return _histograms; }
    /// Number of kinematics of each point.
    std::size_t kinematics_count() const { return _kinematics_count; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<HistItem> _histograms;
    std::size_t _kinematics_count;
};

} // namespace madspace

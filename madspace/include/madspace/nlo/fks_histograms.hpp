#pragma once

#include "madspace/nlo/fks_configuration.hpp"
#include "madspace/nlo/histograms.hpp"
#include "madspace/nlo/jets.hpp"
#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/observable.hpp"

namespace madspace {

/**
 * The histograms of the FKS integrand at fixed order of observables of its
 * kinematics: the event, the soft, collinear and soft-collinear
 * counter-events and the Born of the n-body part, each with its weight of
 * @ref FKSCrossSection, as the analyses of MadFKS fill them (fill_plots of
 * fks_singular.f, at the soft counter-event for the n-body part, which has
 * the momenta of the Born there). The observables are those of an
 * @ref Observable on the momenta with jets of @ref NLOJets (built on
 * NLOJets::pids), and the histograms those of @ref NLOHistograms, so that the
 * outputs are those of @ref ObservableHistograms.
 *
 * `batch` is the leading batch dimension, `n` the number of particles of the
 * real processes and `i` indexes the histograms.
 *
 * **Arguments**
 * - `momenta`, `soft_momenta`, `collinear_momenta`, `soft_collinear_momenta`
 *   – `float`, shape `(batch, n, 4)` – the event and its counter-events of
 *   @ref FKSKinematics.
 * - `event_weight`, `soft_weight`, `collinear_weight`, `soft_collinear_weight`
 *   – `float`, shape `(batch,)` – their weights of @ref FKSCrossSection.
 * - `born_momenta` – `float`, shape `(batch, n - 1, 4)` – the Born.
 * - `nbody_weight` – `float`, shape `(batch,)` – the n-body part.
 * - `fks_index` – `int`, shape `(batch,)` – the FKS configuration of the event.
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
class FKSObservableHistograms : public FunctionGenerator {
public:
    /// One histogram: an @ref Observable on the momenta with jets and its
    /// binning.
    struct HistItem {
        /// The observable, one value per event (of shape `(batch,)` or
        /// `(batch, 1)`), on NLOJets::pids.
        Observable observable;
        /// Lower edge of the first bin.
        double min;
        /// Upper edge of the last bin.
        double max;
        /// Number of bins between @ref min and @ref max (excluding
        /// under/overflow).
        std::size_t bin_count;
        /// Number of jets a kinematics needs for the observable (two for the
        /// second jet).
        std::size_t jet_count = 0;
    };
    /// The kinematics of each point, in the order of the arguments.
    static const std::vector<std::string> kinematics;
    /**
     * @param configurations  The FKS configurations of one Born.
     * @param histograms      The histograms to fill.
     * @param jetalgo         Power of the jet algorithm (see @ref NLOJets).
     * @param jetradius       Radius of the jets.
     * @param ptj             Least @f$p_T@f$ of a jet.
     * @param etaj            Largest @f$|\eta|@f$ of a jet, none for
     *                        @f$\leq 0@f$.
     * @param maxjetflavor    Heaviest quark flavor in the jets.
     * @throws std::invalid_argument as @ref NLOJets and @ref NLOHistograms,
     *         and for an observable that is not one value per event on the
     *         momenta with jets.
     */
    FKSObservableHistograms(
        const std::vector<FKSConfiguration>& configurations,
        const std::vector<HistItem>& histograms,
        double jetalgo = -1.,
        double jetradius = 0.4,
        double ptj = 0.,
        double etaj = -1.,
        int maxjetflavor = 4
    );
    /// The histograms passed to the constructor.
    const std::vector<HistItem>& histograms() const { return _histograms; }
    /// The jets of the event and its counter-events.
    const NLOJets& jets() const { return _jets; }
    /// The jets of the Born.
    const NLOJets& born_jets() const { return _born_jets; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<HistItem> _histograms;
    NLOJets _jets, _born_jets;
    NLOHistograms _binning;
};

} // namespace madspace

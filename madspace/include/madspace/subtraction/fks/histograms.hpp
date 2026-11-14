#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/observable.hpp"
#include "madspace/subtraction/fks/configuration.hpp"

namespace madspace {

/**
 * The QCD partons of one kinematics of the FKS integrand clustered into jets,
 * for the observables of its histograms. The momenta it returns hold the
 * incoming particles and the other final-state particles in their order, then
 * the jets by decreasing @f$p_T@f$, and zero momenta in the slots without a
 * jet, with a slot for each final-state parton of the real processes. The
 * layout is that of every configuration and of the Born (with one empty slot
 * more), so that one @ref Observable built on @ref pids serves the event, its
 * counter-events and the n-body part. The QCD partons are the quarks up to
 * `maxjetflavor` and the gluons, clustered with the pp generalized
 * @f$k_T@f$ algorithm of fastjet (power `jetalgo`, radius `jetradius`, E
 * scheme), as @ref FKSCuts clusters them, into jets with
 * @f$p_T \geq@f$ `ptj` and @f$|\eta| \leq@f$ `etaj` (`etaj` @f$\leq 0@f$
 * for none). Partons without transverse momentum (the soft parton of a soft
 * counter-event) make no jet.
 *
 * `batch` is the leading batch dimension, `n` the number of particles of the
 * kinematics and `m` that of the momenta with jets.
 *
 * **Arguments**
 * - `momenta` – `float`, shape `(batch, n, 4)` – the kinematics in the lab
 *   frame, in the order of the real process of its configuration (of the Born
 *   with @p born).
 * - `fks_index` – `int`, shape `(batch,)` – the FKS configuration of the event.
 *
 * **Returns**
 * - `jet_momenta` – `float`, shape `(batch, m, 4)` – the momenta with jets.
 * - `jet_count` – `int`, shape `(batch,)` – the number of jets.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 * - [2] M. Cacciari, G. P. Salam, G. Soyez, "FastJet user manual",
 *   https://arxiv.org/abs/1111.6097
 */
class FKSJets : public FunctionGenerator {
public:
    /**
     * @param configurations  The FKS configurations of one Born.
     * @param born            Whether the kinematics is the Born, for the
     *                        n-body part.
     * @param jetalgo         Power of the generalized @f$k_T@f$ algorithm: 1
     *                        for @f$k_T@f$, 0 for Cambridge/Aachen, -1 for
     *                        anti-@f$k_T@f$.
     * @param jetradius       Radius of the jets.
     * @param ptj             Least @f$p_T@f$ of a jet.
     * @param etaj            Largest @f$|\eta|@f$ of a jet, none for
     *                        @f$\leq 0@f$.
     * @param maxjetflavor    Heaviest quark flavor in the jets.
     * @throws std::invalid_argument for configurations that differ outside
     *         their QCD partons, for more partons than the clustering takes
     *         (FKS_JET_PARTONS_MAX) or a radius that is not positive.
     */
    FKSJets(
        const std::vector<FKSConfiguration>& configurations,
        bool born = false,
        double jetalgo = -1.,
        double jetradius = 0.4,
        double ptj = 0.,
        double etaj = -1.,
        int maxjetflavor = 4
    );
    /// Number of particles of the kinematics.
    std::size_t particle_count() const { return _particle_count; }
    /// Number of the momenta with jets, the other particles and the slots.
    std::size_t jet_particle_count() const { return _pids.size(); }
    /// Number of slots for jets, the final-state partons of the real
    /// processes.
    std::size_t slot_count() const { return _slot_count; }
    /// The PDG codes of the momenta with jets: those of the incoming and
    /// other particles of the Born of the first configuration, then 21 for
    /// each jet slot.
    const std::vector<int>& pids() const { return _pids; }
    /// Whether the kinematics is the Born.
    bool born() const { return _born; }
    /// jetalgo, jetradius, ptj and etaj.
    const std::vector<double>& options() const { return _options; }
    /// The most final-state partons of one kinematics, those of the
    /// clustering of @ref FKSCuts (FKSCuts::jet_parton_max).
    static constexpr std::size_t parton_max = 16;

private:
    // the particles of the kinematics, the jet slots, the PDG codes of the
    // momenta with jets and the partons of each configuration
    struct Layout {
        std::size_t particle_count;
        std::size_t slot_count;
        std::vector<int> pids;
        nested_vector2<me_int_t> parton;
    };
    static Layout layout(
        const std::vector<FKSConfiguration>& configurations, bool born, int maxjetflavor
    );
    FKSJets(Layout layout, bool born, const std::vector<double>& options);
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    bool _born;
    std::size_t _particle_count;
    std::size_t _slot_count;
    std::vector<int> _pids;
    nested_vector2<me_int_t> _parton;
    std::vector<double> _options;
};

/**
 * The histograms of the FKS integrand at fixed order, with the event and the
 * counter-events of each point in the same histograms, as the analyses of
 * MadFKS fill them [2] (HwU_fill and HwU_add_points of HwU.f): the weights of
 * the kinematics of one point that fall into one bin add up, and the square
 * of that sum enters the error of the bin. The counterterms then cancel the
 * real emission bin by bin, also in the error, for an observable that is
 * infrared safe. The outputs are those of @ref ObservableHistograms, sums
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
class FKSHistograms : public FunctionGenerator {
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
    FKSHistograms(
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

/**
 * The histograms of the FKS integrand at fixed order of observables of its
 * kinematics: the event, the soft, collinear and soft-collinear
 * counter-events and the Born of the n-body part, each with its weight of
 * @ref FKSCrossSection, as the analyses of MadFKS fill them (fill_plots of
 * fks_singular.f, at the soft counter-event for the n-body part, which has
 * the momenta of the Born there). The observables are those of an
 * @ref Observable on the momenta with jets of @ref FKSJets (built on
 * FKSJets::pids), and the histograms those of @ref FKSHistograms, so that the
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
        /// `(batch, 1)`), on FKSJets::pids.
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
     * @param jetalgo         Power of the jet algorithm (see @ref FKSJets).
     * @param jetradius       Radius of the jets.
     * @param ptj             Least @f$p_T@f$ of a jet.
     * @param etaj            Largest @f$|\eta|@f$ of a jet, none for
     *                        @f$\leq 0@f$.
     * @param maxjetflavor    Heaviest quark flavor in the jets.
     * @throws std::invalid_argument as @ref FKSJets and @ref FKSHistograms,
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
    const FKSJets& jets() const { return _jets; }
    /// The jets of the Born.
    const FKSJets& born_jets() const { return _born_jets; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<HistItem> _histograms;
    FKSJets _jets, _born_jets;
    FKSHistograms _binning;
};

} // namespace madspace

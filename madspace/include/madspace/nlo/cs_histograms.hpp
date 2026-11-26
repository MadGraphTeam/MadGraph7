#pragma once

#include "madspace/nlo/histograms.hpp"
#include "madspace/nlo/jets.hpp"
#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/observable.hpp"

namespace madspace {

/**
 * The histograms of a stream of the Catani-Seymour integrand at fixed order
 * of observables of its kinematics, those of the driver in Python
 * (CSHistograms of madgraph/mg7/nlo/cs.py): the n-body stream fills the Born
 * of each point with its weight, the real stream the real emission with its
 * weight and the Born of each dipole with the weight of the dipole, all
 * kinematics of a point together, so that the cancellations of the real
 * emission and its dipoles stay within a bin. Each kinematics has its jets,
 * an @ref NLOJets of whose layouts it takes the first (the Borns of the
 * dipoles of a real process can be different processes), and per histogram
 * its @ref Observable on the momenta with these jets. The histograms are those
 * of @ref NLOHistograms, so that the outputs are those of
 * @ref ObservableHistograms.
 *
 * `batch` is the leading batch dimension, `n` the number of particles of the
 * Born, `d` the number of dipoles and `i` indexes the histograms.
 *
 * **Arguments** (n-body stream)
 * - `momenta` – `float`, shape `(batch, n, 4)` – the Born.
 * - `weight` – `float`, shape `(batch,)` – its weight.
 *
 * **Arguments** (real stream)
 * - `momenta` – `float`, shape `(batch, n + 1, 4)` – the real emission.
 * - `real_weight` – `float`, shape `(batch,)` – its weight.
 * - `born_momenta` – `float`, shape `(batch, d, n, 4)` – the Borns of the
 *   dipoles.
 * - `dipole_weights` – `float`, shape `(batch, d)` – their weights.
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
 * - [2] S. Catani, M. H. Seymour, "A general algorithm for calculating jet
 *   cross sections in NLO QCD", https://arxiv.org/abs/hep-ph/9605323
 */
class CSObservableHistograms : public FunctionGenerator {
public:
    /// One histogram: an @ref Observable per kinematics on its momenta with
    /// jets, and its binning.
    struct HistItem {
        /// The observable of each kinematics, one value per event (of shape
        /// `(batch,)` or `(batch, 1)`), on the NLOJets::pids of its jets.
        std::vector<Observable> observables;
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
    /**
     * @param jets        The jets of each kinematics, in the flavor layout
     *                    of its process: the Born for the n-body stream, the
     *                    real process and then the Born of each dipole for
     *                    the real stream.
     * @param histograms  The histograms to fill.
     * @param real        Whether the kinematics are those of the real stream.
     * @throws std::invalid_argument for other than one kinematics of the
     *         n-body stream or for one of the real stream, for Borns of
     *         different numbers of particles, for observables of another
     *         number of kinematics, for an observable that is not one value
     *         per event on the momenta with the jets of its kinematics, and as
     *         @ref NLOHistograms.
     */
    CSObservableHistograms(
        const std::vector<NLOJets>& jets,
        const std::vector<HistItem>& histograms,
        bool real
    );
    /// The histograms passed to the constructor.
    const std::vector<HistItem>& histograms() const { return _histograms; }
    /// The jets of each kinematics.
    const std::vector<NLOJets>& jets() const { return _jets; }
    /// Whether the kinematics are those of the real stream.
    bool real() const { return _real; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    std::vector<HistItem> _histograms;
    std::vector<NLOJets> _jets;
    bool _real;
    NLOHistograms _binning;
};

} // namespace madspace

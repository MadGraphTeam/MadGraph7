#pragma once

#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "madspace/driver/backend.hpp"
#include "madspace/driver/context.hpp"
#include "madspace/driver/io.hpp"
#include "madspace/driver/systematics.hpp"
#include "madspace/phasespace/histograms.hpp"

namespace madspace {

/// One @ref EventHistograms histogram: an observable name and its binning.
struct EventHistogramSpec {
    /// Observable name; must match a key of every subprocess's @ref
    /// SubprocessObservables::values.
    std::string name;
    /// Lower edge of the first bin.
    double min;
    /// Upper edge of the last bin.
    double max;
    /// Number of bins between @ref min and @ref max (excluding under/overflow).
    std::size_t bin_count;
};

/// The observables of one subprocess, evaluated on its external momenta.
struct SubprocessObservables {
    /// The @ref ObservableValues function generator for this subprocess.
    ObservableValues values;
    /// Number of external particles this subprocess's momenta carry.
    std::size_t particle_count;
};

/**
 * Histograms of the final (unweighted) event sample.
 *
 * Filled at combine time with the nominal weight and every @ref
 * SystematicsCalculator variation weight, so that each observable carries its
 * scale envelope and PDF uncertainty band alongside its central value.
 */
class EventHistograms {
public:
    /**
     * @param context      Context the per-subprocess observable runtimes are
     *                     built on.
     * @param specs        The histograms to fill.
     * @param observables  `observables[subprocess]` evaluates the
     *                     histogrammed observables, in the order of `specs`,
     *                     for that (unmerged) subprocess's events; a
     *                     `nullopt` entry skips the subprocess.
     */
    EventHistograms(
        ContextPtr context,
        const std::vector<EventHistogramSpec>& specs,
        const std::vector<std::optional<SubprocessObservables>>& observables
    );

    /// The histogram definitions passed to the constructor.
    const std::vector<EventHistogramSpec>& specs() const { return _specs; }
    /// Number of weight columns each histogram carries (nominal plus every
    /// systematic variation).
    std::size_t weight_count() const { return _weight_count; }

    /**
     * Fill from a batch of combined events.
     *
     * `syst_weights` holds `weight_count` variation weights per event
     * (row-major), as computed by @ref SystematicsCalculator. Column 0 of the
     * histograms is the nominal weight. Thread-safe.
     */
    void fill(
        EventBuffer& buffer,
        const std::vector<double>& syst_weights,
        std::size_t weight_count
    );

    /// Bin contents normalized to cross sections (summing over bins gives the
    /// cross section of the corresponding weight); with `systematics`, the
    /// scale envelope and the PDF uncertainty bands are added per bin.
    nlohmann::json to_json(const SystematicsCalculator* systematics = nullptr) const;

private:
    struct RuntimeData {
        RuntimePtr runtime;
        std::size_t particle_count;
    };

    std::vector<EventHistogramSpec> _specs;
    std::vector<std::optional<RuntimeData>> _runtimes;
    std::size_t _weight_count = 0;
    // sums[observable][weight column][bin], bins: underflow, bin_count, overflow
    std::vector<std::vector<std::vector<double>>> _sums;
    std::vector<std::vector<std::vector<double>>> _square_sums;
    std::size_t _event_count = 0;
    std::mutex _mutex;
};

} // namespace madspace

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

struct EventHistogramSpec {
    std::string name;
    double min;
    double max;
    std::size_t bin_count;
};

// The observables of one subprocess, evaluated on its external momenta
struct SubprocessObservables {
    ObservableValues values;
    std::size_t particle_count;
};

// Histograms of the final (unweighted) event sample, filled at combine time
// with the nominal weight and every systematic variation weight, so that each
// observable carries its scale envelope and PDF uncertainty band.
class EventHistograms {
public:
    // `observables[subprocess]` evaluates the histogrammed observables (in the
    // order of `specs`) for the events of that (unmerged) subprocess; a
    // nullopt entry skips the subprocess.
    EventHistograms(
        ContextPtr context,
        const std::vector<EventHistogramSpec>& specs,
        const std::vector<std::optional<SubprocessObservables>>& observables
    );

    const std::vector<EventHistogramSpec>& specs() const { return _specs; }
    std::size_t weight_count() const { return _weight_count; }

    // Fill from a batch of combined events; `syst_weights` holds `weight_count`
    // variation weights per event (row-major), as computed by the
    // SystematicsCalculator. Column 0 of the histograms is the nominal weight.
    // Thread-safe.
    void fill(EventBuffer& buffer, const std::vector<double>& syst_weights, std::size_t weight_count);

    // Bin contents normalised to cross sections (sum over bins = cross section
    // of the corresponding weight); with `systematics`, the scale envelope and
    // the PDF uncertainty bands are added per bin.
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

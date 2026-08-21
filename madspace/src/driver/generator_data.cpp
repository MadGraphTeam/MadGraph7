#include "madspace/driver/generator_data.hpp"

#include <algorithm>
#include <cmath>

using namespace madspace;

std::size_t madspace::compute_generation_batch_event_count(
    std::size_t count_target,
    double count_unweighted,
    std::size_t count_opt,
    std::size_t cross_section_count,
    double cross_section_rel_error,
    const GeneratorConfig& config
) {
    double efficiency = count_opt > 0
        ? std::max(count_unweighted / static_cast<double>(count_opt), 1. / count_opt)
        : 1.;

    double true_remaining =
        std::max(static_cast<double>(count_target) - count_unweighted, 0.);

    if (true_remaining <= config.finish_remaining_fraction * count_target) {
        return static_cast<std::size_t>(
            std::max(1., std::ceil(true_remaining / efficiency))
        );
    }

    double rel_error = cross_section_count > 1 && std::isfinite(cross_section_rel_error)
        ? cross_section_rel_error
        : 1.;

    double target_uncertainty = static_cast<double>(count_target) * rel_error;
    double safe_target = static_cast<double>(count_target) -
        config.batch_overshoot_sigma * target_uncertainty;
    double safe_remaining = std::max(safe_target - count_unweighted, 0.);
    double capped = std::min(
        config.max_batch_fraction * static_cast<double>(count_target), safe_remaining
    );

    return static_cast<std::size_t>(std::max(1., std::ceil(capped / efficiency)));
}

void madspace::to_json(nlohmann::json& j, const GeneratorStatus& status) {
    j = nlohmann::json{
        {"subprocess", status.subprocess},
        {"name", status.name},
        {"mean", status.mean},
        {"error", status.error},
        {"rel_std_dev", status.rel_std_dev},
        {"count", status.count},
        {"count_opt", status.count_opt},
        {"count_after_cuts", status.count_after_cuts},
        {"count_after_cuts_opt", status.count_after_cuts_opt},
        {"count_unweighted", status.count_unweighted},
        {"count_target", status.count_target},
        {"iterations", status.iterations},
        {"optimized", status.optimized},
        {"done", status.done},
    };
}

void madspace::to_json(nlohmann::json& j, const Histogram& hist) {
    j = nlohmann::json{
        {"name", hist.name},
        {"min", hist.min},
        {"max", hist.max},
        {"bin_values", hist.bin_values},
        {"bin_errors", hist.bin_errors},
    };
}

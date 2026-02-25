#pragma once

#include <random>
#include <vector>

#include "madspace/runtime/generator_data.h"

#include "madspace/madcode.h"
#include "madspace/phasespace.h"
#include "madspace/runtime/discrete_optimizer.h"
#include "madspace/runtime/generator_data.h"
#include "madspace/runtime/io.h"
#include "madspace/runtime/runtime_base.h"
#include "madspace/runtime/vegas_optimizer.h"

namespace madspace {

class ChannelEventGenerator {
public:
    static inline const int integrand_flags = Integrand::sample |
        Integrand::return_momenta | Integrand::return_indices |
        Integrand::return_random | Integrand::return_discrete;

    ChannelEventGenerator(
        const std::vector<ContextPtr>& contexts,
        const Integrand& integrand,
        const std::string& event_file,
        const std::string& weight_file,
        const GeneratorConfig& config,
        std::size_t index,
        std::size_t subprocess_index,
        const std::string& name,
        const std::optional<ObservableHistograms>& histograms,
        double integral_estimate = 0.
    );

    const GeneratorStatus& status() const { return _status; }

private:
    struct ContextRuntimes {
        RuntimePtr integrand;
        RuntimePtr unweighter;
        RuntimePtr vegas_histogram;
        RuntimePtr discrete_histogram;
        RuntimePtr observable_histograms;
    };
    void unweight_file(std::mt19937 rand_gen);
    void integrate_and_optimize(const GeneratorBatchJob& job, bool run_optim);
    double channel_weight_sum(std::size_t event_count);
    void start_job(GeneratorBatchJob& job);
    void build_vegas_jobs(std::vector<GeneratorBatchJob>& ready_jobs, bool unweight);
    void clear_events();
    void update_max_weight(Tensor weights);
    void unweight_and_write(const TensorVec& unweighted_events);

    GeneratorStatus _status;
    GeneratorConfig _config;
    std::vector<ContextPtr> _contexts;
    std::vector<ContextRuntimes> _runtimes;
    EventFile _event_file;
    EventFile _weight_file;
    std::optional<VegasGridOptimizer> _vegas_optimizer;
    std::optional<DiscreteOptimizer> _discrete_optimizer;
    std::size_t _batch_size;
    RunningIntegral _cross_section;
    bool _needs_optimization = true;
    double _max_weight = 0.;
    std::size_t _unweighted_count = 0;
    double _integral_fraction = 1.;
    std::size_t _iters_without_improvement = 0;
    double _best_rsd = std::numeric_limits<double>::max();
    std::vector<double> _large_weights;
    std::vector<Histogram> _histograms;

    friend class EventGenerator;
};

} // namespace madspace

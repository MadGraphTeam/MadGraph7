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
        std::size_t subprocess_index,
        const std::string& name,
        const std::optional<ObservableHistograms>& histograms
    );

    const GeneratorStatus& status() const { return _status; }
    const RunningIntegral& cross_section() const { return _cross_section; }
    const std::vector<Histogram>& histograms() const { return _histograms; }
    EventFile& event_file() { return _event_file; }
    EventFile& weight_file() { return _weight_file; }
    std::size_t max_weight() const { return _max_weight; }
    bool needs_optimization() const {
        return (_vegas_optimizer || _discrete_optimizer) && !_status.optimized;
    }
    void set_target_count(double target_count) { _status.count_target = target_count; }

    void unweight_file(std::mt19937& rand_gen);
    void integrate_and_optimize(const GeneratorBatchJob& job, bool run_optim);
    double channel_weight_sum(std::size_t event_count);
    void start_job(GeneratorBatchJob& job, ResultQueue& result_queue);
    std::size_t next_vegas_batch_size();
    void clear_events();
    void update_max_weight(Tensor weights);
    void unweight_and_write(const TensorVec& unweighted_events);

private:
    struct ContextRuntimes {
        RuntimePtr integrand;
        RuntimePtr unweighter;
        RuntimePtr vegas_histogram;
        RuntimePtr discrete_histogram;
        RuntimePtr observable_histograms;
    };

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
    double _max_weight = 0.;
    std::size_t _unweighted_count = 0;
    std::size_t _iters_without_improvement = 0;
    double _best_rsd = std::numeric_limits<double>::max();
    std::vector<double> _large_weights;
    std::vector<Histogram> _histograms;
};

} // namespace madspace

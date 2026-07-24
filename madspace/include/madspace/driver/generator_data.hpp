#pragma once

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "madspace/driver/logger.hpp"
#include "madspace/driver/tensor.hpp"

namespace madspace {

class RunningIntegral {
public:
    RunningIntegral() : _mean(0), _var_sum(0), _count(0) {}
    double mean() const { return _mean; }
    double variance() const { return _count > 1 ? _var_sum / (_count - 1) : 0; }
    double error() const { return std::sqrt(variance() / _count); }
    double rel_error() const { return error() / mean(); }
    double rel_std_dev() const { return std::sqrt(variance()) / _mean; }
    std::size_t count() const { return _count; }
    void reset() {
        _mean = 0;
        _var_sum = 0;
        _count = 0;
    }
    void push(double value) {
        ++_count;
        if (_count == 1) {
            _mean = value;
            _var_sum = 0;
        } else {
            double mean_diff = value - _mean;
            _mean += mean_diff / _count;
            _var_sum += mean_diff * (value - _mean);
        }
    }

private:
    double _mean;
    double _var_sum;
    std::size_t _count;
};

struct GeneratorConfig {
    std::size_t target_count = 10000; // TODO: don't include here
    double vegas_damping = 0.2;
    double max_overweight_truncation = 0.01;
    std::size_t freeze_max_weight_after = 10000;
    std::size_t start_batch_size = 1000;
    std::size_t max_batch_size = 64000;
    std::size_t survey_min_iters = 3;
    std::size_t survey_max_iters = 4;
    double survey_target_precision = 0.1;
    std::size_t optimization_patience = 3;
    double optimization_threshold = 0.99;
    std::size_t cpu_batch_size = 1000;
    std::size_t gpu_batch_size = 64000;
    Verbosity verbosity = Verbosity::silent;
    bool write_live_data = false;
    int combine_thread_count = -1;
    double cut_efficiency_threshold = 0.7;
    std::size_t max_cut_repetitions = 100;
    // Steady-state event generation dispatches a channel's jobs in barrier-gated
    // batches (a new batch is only sized/created once the previous one has fully
    // committed), so a batch's size is a deterministic function of committed data,
    // never of scheduling timing. generation_batch_fraction caps a batch to this
    // fraction of the channel's currently remaining need, so a single batch never
    // bets everything on an efficiency estimate that may still be noisy (e.g. for
    // costly/high-variance channels early on); min_batch_jobs floors the batch size
    // so a lone active channel still gets enough parallel jobs to keep all worker
    // threads busy. See EventGenerator::next_batch_job_count().
    double generation_batch_fraction = 0.5;
    std::size_t min_batch_jobs = 1;
};

struct GeneratorStatus {
    std::size_t subprocess;
    std::string name;
    double mean;
    double error;
    double rel_std_dev;
    std::size_t count;
    std::size_t count_opt;
    std::size_t count_after_cuts;
    std::size_t count_after_cuts_opt;
    double count_unweighted;
    double count_target;
    std::size_t iterations;
    bool optimized;
    bool done;
};

struct Histogram {
    std::string name;
    double min;
    double max;
    std::vector<double> bin_values;
    std::vector<double> bin_errors;
};

struct GeneratorBatchJob {
    std::size_t channel_index;
    bool unweight;
    // Nonzero to make start_jobs() split this ready_job into ceil(vegas_batch_size /
    // context_batch_size) sub-jobs and dispatch all of them atomically (bypassing
    // the usual per-context dispatch cap), so channel_job_count reflects the whole
    // batch from the moment it's dispatched rather than only whichever prefix fit
    // under the cap -- see is_vegas_batch for what the batch is *for*.
    std::size_t vegas_batch_size;
    std::size_t split_job_count;
    Tensor weights;
    TensorVec events;
    TensorVec unweighted_events;
    TensorVec hists;
    TensorVec vegas_hist;
    TensorVec discrete_hist;
    std::size_t context_index;
    std::size_t job_id;
    double max_weight;
    // Base seed for this job's deterministic random stream, derived at scheduling
    // time from (context seed, channel, per-channel job sequence). 0 means "use the
    // non-deterministic pool generator" (context seed 0). See ChannelEventGenerator.
    std::uint64_t rng_seed = 0;
    // True for a VEGAS-grid-optimization batch (triggers clear_events()/
    // optimize_vegas() at the start/end of the batch); false for a steady-state
    // generation batch (see EventGenerator::_channel_batch_pending), even though
    // both use vegas_batch_size > 0 to get the same atomic whole-batch dispatch.
    bool is_vegas_batch = false;
};

void to_json(nlohmann::json& j, const GeneratorStatus& status);
void to_json(nlohmann::json& j, const Histogram& hist);

} // namespace madspace

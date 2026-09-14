#pragma once

#include <cmath>
#include <cstddef>
#include <optional>
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
    double finish_remaining_fraction = 0.05;
    double max_batch_fraction = 0.6;
    double batch_overshoot_sigma = 1.0;
};

// Determine number of events to be scheduled in the next step. If close to finishing,
// return remaining count / efficiency. Otherwise select a count that is safely below
// the target using the integration uncertainty, and impose an upper limit which
// fration of the target events can be scheduled in one go.
std::size_t compute_generation_batch_event_count(
    std::size_t count_target,
    double count_unweighted,
    std::size_t count_opt,
    std::size_t abs_cross_section_count,
    double abs_cross_section_rel_error,
    const GeneratorConfig& config
);

struct GeneratorStatus {
    std::size_t subprocess;
    std::string name;
    double mean;
    double error;
    double mean_abs;  // E[|w|]
    double error_abs; // error on E[|w|]
    double rel_std_dev;
    std::size_t count;
    std::size_t count_opt;
    std::size_t count_after_cuts;
    std::size_t count_after_cuts_opt;
    double count_unweighted;
    std::size_t count_target;
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

// Lightweight pending-work entry for EventGenerator::_ready_jobs, before start_jobs()
// has decided which context/how many sub-jobs to create from it. Kept separate from
// GeneratorBatchJob so a queue of pending batches doesn't carry the weight of every
// dispatched job's tensors and RNG bookkeeping.
struct ReadyJob {
    std::size_t channel_index;
    bool unweight;
    // VEGAS batch: fixed size, start_jobs() dispatches it atomically in one go.
    // Generation batch: events not yet dispatched -- start_jobs() decrements this in
    // place as it creates sub-jobs, one device batch at a time, round-robining with
    // other channels' ReadyJobs over however many calls it takes to reach zero.
    std::size_t batch_event_count;
    bool is_vegas_batch = false;
};

struct GeneratorBatchJob {
    std::size_t channel_index;
    bool unweight;
    // Copied from the originating ReadyJob at dispatch time. For a VEGAS batch, this
    // is the batch's fixed total, read by start_job()'s shrink-to-fit and by the
    // done_event_count accounting in survey()/survey_deterministic(). For a
    // generation batch it isn't read after dispatch -- generation sub-jobs always
    // request a full device batch (see start_job()).
    std::size_t batch_event_count;
    // Total sub-jobs the batch was split into; only meaningful for VEGAS batches
    // (see commit_generate_job()'s clear_events trigger). Generation batches are
    // dispatched one sub-job per start_jobs() visit, so this isn't a full-batch count
    // for them and isn't read.
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
    // Top-level seed plus job identity, used to derive this job's DerivedSeed(s)
    // independently for generate vs unweight (see start_job()/submit_unweight_job()).
    std::optional<std::uint64_t> rng_seed;
    // Per-channel dispatch sequence, assigned once at start_job() time.
    std::size_t rng_job_index = 0;
    bool rng_is_survey = false;
    std::size_t rng_survey_pass = 0;
    // True for a VEGAS-grid-optimization batch, dispatched atomically by start_jobs()
    // and shrunk to fit by start_job(). False for a steady-state generation batch,
    // dispatched incrementally as device-sized sub-jobs (see ReadyJob).
    bool is_vegas_batch = false;
};

void to_json(nlohmann::json& j, const GeneratorStatus& status);
void to_json(nlohmann::json& j, const Histogram& hist);

} // namespace madspace

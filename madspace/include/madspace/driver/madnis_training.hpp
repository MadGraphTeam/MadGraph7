#pragma once

#include <chrono>
#include <memory>

#include "madspace/compgraphs.hpp"
#include "madspace/driver/adam_optimizer.hpp"
#include "madspace/driver/format.hpp"
#include "madspace/driver/logger.hpp"
#include "madspace/driver/status_file.hpp"
#include "madspace/phasespace.hpp"

namespace madspace {

class MadnisTraining {
public:
    static void set_abort_check_function(std::function<void(void)> func) {
        _abort_check_function = func;
    }

    struct Config {
        double learning_rate = 1e-3;
        std::size_t batches = 1000;
        std::size_t log_interval = 100;
        std::size_t integration_history_length = 1000;
        std::size_t channel_dropping_interval = 100;
        double channel_dropping_threshold = 0.01;
        std::size_t cpu_generator_batch_size = 1000;
        std::size_t gpu_generator_batch_size = 64000;
        std::size_t gpu_generator_batch_granularity = 1000;
        std::size_t generator_target_size_factor = 32;
        std::size_t batch_size_offset = 512;
        std::size_t batch_size_per_channel = 128;
        double uniform_channel_ratio = 0.1;
        AdamOptimizer::LRSchedule lr_schedule = AdamOptimizer::none;
        double adam_beta1 = 0.9;
        double adam_beta2 = 0.999;
        double adam_eps = 1e-8;
        double grad_clip_threshold = 0.0;
        std::size_t buffer_capacity = 0;
        std::size_t minimum_buffer_size = 10000;
        std::size_t buffered_steps = 0;
        double buffer_unweighting_quantile = 0.99;
        double fixed_cwnet_fraction = 0.33;
        double softclip_threshold = 0.0;
        std::size_t compressed_channel_weight_count = 50;
    };
    MadnisTraining(
        ContextPtr generator_context,
        ContextPtr optimizer_context,
        const Config& config,
        const std::vector<std::shared_ptr<Integrand>>& integrands,
        const std::optional<ChannelWeightNetwork>& cwnet
    );
    const Config& config() const { return _config; }
    void train_step(std::size_t batch_index);
    std::vector<std::size_t> active_channels() const;
    std::size_t active_channel_count() const { return _channels.size(); }
    double average_loss() const;
    double average_learning_rate() const;
    double buffered_fraction() const;
    std::size_t generated_event_count() const { return _generated_event_count; }
    std::size_t buffer_event_count() const;

    // Status histories, one entry appended every config.log_interval batches,
    // for reporting training progress (e.g. to a StatusFile).
    const std::vector<std::size_t>& status_batches() const { return _status_batches; }
    const std::vector<double>& status_losses() const { return _status_losses; }
    const std::vector<std::size_t>& status_channel_counts() const {
        return _status_channel_counts;
    }
    const std::vector<double>& status_learning_rates() const {
        return _status_learning_rates;
    }
    const std::vector<double>& status_buffered_fractions() const {
        return _status_buffered_fractions;
    }
    const std::vector<std::size_t>& status_generated_events() const {
        return _status_generated_events;
    }
    const std::vector<std::size_t>& status_buffer_sizes() const {
        return _status_buffer_sizes;
    }

private:
    struct SampleBatch {
        std::vector<std::size_t> channel_sizes;
        TensorVec tensors;
        std::size_t consumed_count = 0;
        std::size_t size = 0;
        std::size_t channel_index = 0;
    };
    struct SampleJob {
        SampleBatch samples;
        SampleBatch unweighted_samples;
    };
    struct ChannelData {
        std::size_t index;
        std::vector<SampleBatch> sample_batches;
        std::vector<std::tuple<std::size_t, double, double>> integration_history;
        std::size_t history_index = 0;
        std::size_t sample_count = 0;
        std::shared_ptr<Integrand> integrand;
        std::shared_ptr<IntegrandProbability> integrand_prob;
        RuntimePtr generator_runtime = nullptr;
        RuntimePtr unweighter_runtime = nullptr;
        SampleBatch buffer;
    };

    inline static std::function<void(void)> _abort_check_function = [] {};

    void build_runtimes_and_optimizer();
    std::vector<std::size_t> compute_channel_sizes();
    void start_generator_jobs(const std::vector<std::size_t>& channel_fractions);
    TensorVec permute_tensors(const TensorVec& tensors) const;
    void start_single_job(std::size_t channel_index, std::size_t batch_size);
    void start_multi_job(const std::vector<std::size_t> batch_sizes);
    bool check_online_training_batch(const std::vector<std::size_t>& channel_sizes);
    bool check_buffered_training_batch(const std::vector<std::size_t>& channel_sizes);
    TensorVec build_online_training_batch(const std::vector<size_t>& counts);
    TensorVec build_buffered_training_batch(const std::vector<size_t>& counts);
    void process_job_results(const std::vector<std::size_t>& job_ids);
    void buffer_store(ChannelData& channel, SampleBatch& samples);
    void update_history(
        const TensorVec& results,
        const std::vector<std::size_t>& counts,
        double learning_rate,
        bool buffered
    );
    void drop_channels();
    void freeze_cwnet();

    ContextPtr _generator_context;
    ContextPtr _optimizer_context;
    std::optional<ChannelWeightNetwork> _cwnet;
    Config _config;
    RuntimePtr _multi_channel_generator = nullptr;
    RuntimePtr _multi_channel_unweighter = nullptr;
    RuntimePtr _multi_channel_sampler = nullptr;
    std::optional<AdamOptimizer> _optimizer;
    std::vector<ChannelData> _channels;
    std::unordered_map<std::size_t, SampleJob> _running_jobs;
    std::vector<double> _loss_history;
    std::vector<double> _lr_history;
    std::vector<bool> _buffered_history;
    std::size_t _loss_history_index = 0;
    std::vector<std::size_t> _status_batches;
    std::vector<double> _status_losses;
    std::vector<std::size_t> _status_channel_counts;
    std::vector<double> _status_learning_rates;
    std::vector<double> _status_buffered_fractions;
    std::vector<std::size_t> _status_generated_events;
    std::vector<std::size_t> _status_buffer_sizes;
    std::size_t _generated_event_count = 0;
    std::size_t _job_id = 0;
    Tensor _generator_params;
    std::vector<std::size_t> _arg_permutation;
    bool _buffer_ready = false;
    std::vector<std::size_t> _active_flavors_count;
};

class MultiMadnisTraining {
public:
    struct TrainingArgs {
        MadnisTraining::Config config;
        std::vector<std::shared_ptr<Integrand>> integrands;
        std::optional<ChannelWeightNetwork> cwnet;
    };

    MultiMadnisTraining(
        ContextPtr generator_context,
        ContextPtr optimizer_context,
        const std::vector<TrainingArgs>& training_args,
        Verbosity verbosity = Verbosity::log,
        std::shared_ptr<StatusFile> status_file = nullptr
    );
    void train();
    nested_vector2<std::size_t> active_channels() const;

private:
    void print_progress_init();
    void print_progress_update(
        std::size_t subproc_index,
        std::size_t batch_index,
        double loss,
        std::size_t chan_count
    );
    void write_status(std::size_t subproc_index, std::size_t batch_index, bool done);

    Verbosity _verbosity;
    std::vector<MadnisTraining> _subprocesses;
    std::chrono::time_point<std::chrono::steady_clock> _start_time;
    std::size_t _start_cpu_microsec;
    std::chrono::time_point<std::chrono::steady_clock> _last_print_time;
    PrettyBox _pretty_box_upper;
    PrettyBox _pretty_box_lower;
    std::shared_ptr<StatusFile> _status_file;
};

} // namespace madspace

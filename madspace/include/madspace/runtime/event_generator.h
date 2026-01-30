#pragma once

#include <chrono>
#include <optional>
#include <random>
#include <vector>

#include <nlohmann/json.hpp>

#include "madspace/madcode.h"
#include "madspace/phasespace.h"
#include "madspace/runtime/discrete_optimizer.h"
#include "madspace/runtime/format.h"
#include "madspace/runtime/io.h"
#include "madspace/runtime/runtime_base.h"
#include "madspace/runtime/vegas_optimizer.h"

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

class EventGenerator {
public:
    static inline const int integrand_flags = Integrand::sample |
        Integrand::return_momenta | Integrand::return_indices |
        Integrand::return_random | Integrand::return_discrete;
    enum Verbosity { silent, log, pretty };
    struct Config {
        std::size_t target_count = 10000;
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
        std::size_t batch_size = 1000;
        Verbosity verbosity = silent;
        bool write_live_data = false;
    };
    static const Config default_config;
    struct Status {
        std::size_t index;
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
    static void set_abort_check_function(std::function<void(void)> func) {
        _abort_check_function = func;
    }

    EventGenerator(
        ContextPtr context,
        const std::vector<Integrand>& channels,
        const std::string& temp_file_prefix,
        const std::string& status_file = "",
        const Config& config = default_config,
        const std::vector<std::size_t>& channel_subprocesses = {},
        const std::vector<std::string>& channel_names = {},
        const std::vector<ObservableHistograms>& channel_histograms = {}
    );
    void survey();
    void generate();
    void combine_to_compact_npy(const std::string& file_name);
    void combine_to_lhe_npy(const std::string& file_name, LHECompleter& lhe_completer);
    void combine_to_lhe(const std::string& file_name, LHECompleter& lhe_completer);
    Status status() const { return _status_all; }
    std::vector<Status> channel_status() const;
    std::vector<Histogram> histograms() const;

private:
    struct ChannelState {
        std::size_t index;
        RuntimePtr runtime;
        EventFile event_file;
        EventFile weight_file;
        std::optional<VegasGridOptimizer> vegas_optimizer;
        RuntimePtr vegas_histogram;
        std::optional<DiscreteOptimizer> discrete_optimizer;
        RuntimePtr discrete_histogram;
        RuntimePtr observable_histograms;
        std::size_t batch_size;
        std::string name;
        std::size_t subprocess_index;
        RunningIntegral cross_section;
        bool needs_optimization = true;
        double max_weight = 0.;
        double eff_count = 0.;
        double integral_fraction = 1.;
        std::size_t total_sample_count = 0;
        std::size_t total_sample_count_opt = 0;
        std::size_t total_sample_count_after_cuts = 0;
        std::size_t total_sample_count_after_cuts_opt = 0;
        std::size_t iterations = 0;
        std::size_t iters_without_improvement = 0;
        double best_rsd = std::numeric_limits<double>::max();
        std::vector<double> large_weights;
        std::size_t job_count = 0;
        nested_vector2<std::pair<double, double>> histograms;
    };
    struct RunningJob {
        std::size_t channel_index;
        TensorVec events;
        std::size_t vegas_job_count;
    };
    struct CombineChannelData {
        std::size_t cum_count;
        EventBuffer event_buffer;
        EventBuffer weight_buffer;
        std::size_t buffer_index;
    };
    struct TimingData {
        double wall_time_sec;
        double cpu_time_sec;
    };
    inline static std::function<void(void)> _abort_check_function = [] {};

    ContextPtr _context;
    Config _config;
    std::vector<ChannelState> _channels;
    RuntimePtr _unweighter;
    Status _status_all;
    std::unordered_map<std::size_t, RunningJob> _running_jobs;
    std::size_t _job_id;
    std::chrono::time_point<std::chrono::steady_clock> _start_time;
    std::size_t _start_cpu_microsec;
    std::chrono::time_point<std::chrono::steady_clock> _last_print_time;
    std::chrono::time_point<std::chrono::steady_clock> _last_status_time;
    PrettyBox _pretty_box_upper;
    PrettyBox _pretty_box_lower;
    std::string _status_file;
    std::unordered_map<std::string, TimingData> _timing_data;
    std::vector<Histogram> _empty_histograms;

    void reset_start_time();
    void add_timing_data(const std::string& key);
    std::string format_run_time(const std::string& key) const;
    void unweight_all();
    void unweight_channel(ChannelState& channel, std::mt19937 rand_gen);
    std::tuple<Tensor, std::vector<Tensor>> integrate_and_optimize(
        ChannelState& channel, TensorVec& events, bool always_optimize
    );
    double channel_weight_sum(ChannelState& channel, std::size_t event_count);
    void start_job(
        ChannelState& channel, std::size_t batch_size, std::size_t vegas_job_count = 0
    );
    void start_vegas_jobs(ChannelState& channel);
    void clear_channel(ChannelState& channel);
    void update_max_weight(ChannelState& channel, Tensor weights);
    void unweight_and_write(ChannelState& channel, const std::vector<Tensor>& momenta);
    std::size_t max_particle_count();
    std::tuple<std::vector<CombineChannelData>, std::size_t, double> init_combine();
    void read_and_combine(
        std::vector<CombineChannelData>& channel_data,
        EventBuffer& buffer,
        double norm_factor
    );
    void fill_lhe_event(
        LHECompleter& lhe_completer,
        LHEEvent& lhe_event,
        EventBuffer& buffer,
        std::size_t event_index
    );

    void init_status(const std::string& status);
    void write_status(const std::string& status, bool force_write);

    void print_survey_init();
    void print_survey_update(
        bool done,
        std::size_t done_job_count,
        std::size_t total_job_count,
        std::size_t iter
    );
    void print_survey_update_pretty(
        bool done,
        std::size_t done_job_count,
        std::size_t total_job_count,
        std::size_t iter
    );
    void print_survey_update_log(
        bool done,
        std::size_t done_job_count,
        std::size_t total_job_count,
        std::size_t iter
    );

    void print_gen_init();
    void print_gen_update(bool done);
    void print_gen_update_pretty(bool done);
    void print_gen_update_log(bool done);

    void print_combine_init();
    void print_combine_update(std::size_t count);
    void print_combine_update_pretty(std::size_t count);
    void print_combine_update_log(std::size_t count);

    friend void
    to_json(nlohmann::json& j, const EventGenerator::TimingData& timing_data);
};

void to_json(nlohmann::json& j, const EventGenerator::TimingData& timing_data);
void to_json(nlohmann::json& j, const EventGenerator::Status& status);
void to_json(nlohmann::json& j, const EventGenerator::Histogram& hist);

} // namespace madspace

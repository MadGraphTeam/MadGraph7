#pragma once

#include <chrono>
#include <optional>
#include <set>
#include <vector>

#include <nlohmann/json.hpp>

#include "madspace/compgraphs.hpp"
#include "madspace/driver/backend.hpp"
#include "madspace/driver/channel_generator.hpp"
#include "madspace/driver/discrete_optimizer.hpp"
#include "madspace/driver/format.hpp"
#include "madspace/driver/generator_data.hpp"
#include "madspace/driver/io.hpp"
#include "madspace/driver/lhe_output.hpp"
#include "madspace/driver/vegas_optimizer.hpp"
#include "madspace/phasespace.hpp"

namespace madspace {

class EventGenerator {
public:
    static const GeneratorConfig default_config;
    static void set_abort_check_function(std::function<void(void)> func) {
        _abort_check_function = func;
    }

    EventGenerator(
        const std::vector<ContextPtr>& contexts,
        const std::vector<std::shared_ptr<ChannelEventGenerator>>& channels,
        const std::string& status_file = "",
        const GeneratorConfig& config = default_config,
        std::optional<std::uint64_t> seed = std::nullopt
    );
    // `survey_pass` salts job seeds so repeated survey() calls on the same
    // channel (e.g. re-survey after simplification) don't share a seed stream.
    void survey(std::size_t survey_pass = 0);
    void generate();
    void combine_to_compact_npy(const std::string& file_name);
    void combine_to_lhe_npy(const std::string& file_name, LHECompleter& lhe_completer);
    void combine_to_lhe(
        const std::string& file_name,
        LHECompleter& lhe_completer,
        const LHEMeta& meta = {}
    );
    GeneratorStatus status() const { return _status; }
    std::vector<GeneratorStatus> channel_status() const;
    std::vector<Histogram> histograms() const;
    std::unordered_set<std::string> used_globals() const;
    const std::vector<std::shared_ptr<ChannelEventGenerator>>& channels() const {
        return _channels;
    };

private:
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

    GeneratorConfig _config;
    std::vector<std::shared_ptr<ChannelEventGenerator>> _channels;
    GeneratorStatus _status;
    std::vector<ContextPtr> _contexts;
    std::unordered_map<std::size_t, GeneratorBatchJob> _running_jobs;
    std::vector<GeneratorBatchJob> _ready_jobs;
    std::size_t _job_id;
    std::vector<std::size_t> _channel_job_counts;
    std::vector<bool> _channel_optimizing;
    std::vector<double> _channel_integral_fractions;
    std::vector<std::size_t> _context_job_counts;
    // True while a channel has a steady-state batch dispatched but not yet fully
    // committed; keeps next_batch_event_count() from double-counting in-flight work.
    std::vector<bool> _channel_batch_pending;
    // generate_deterministic() only: per-channel commit cursor/buffer, analogous
    // to _ready_gen/_commit_cursor but ordered per channel instead of globally.
    std::vector<std::set<std::size_t>> _channel_ready_gen;
    std::vector<std::size_t> _channel_commit_cursor;
    std::vector<bool> _channel_cursor_set;
    // generate_deterministic() only: same as above, for a job's unweight-stage
    // completion (tracked separately since it's a distinct completion event).
    std::vector<std::set<std::size_t>> _channel_unweight_ready;
    std::vector<std::size_t> _channel_unweight_cursor;
    // generate_deterministic() only: per-context queue of job ids awaiting
    // unweight-stage dispatch, drained with priority by start_jobs().
    std::vector<std::vector<std::size_t>> _context_unweight_queue;
    ResultQueue _result_queue;

    // Base seed for reproducible event generation; nullopt means non-deterministic.
    std::optional<std::uint64_t> _seed;

    // unweight_all() may run more than once per generate() (a channel's target can
    // grow after it looked done, un-finishing it and triggering another round).
    // Salted by this counter so repeated calls don't replay the same stream.
    std::size_t _unweight_call_index = 0;

    // Scheduling context for the running survey()/generate() call, read by
    // start_jobs() to derive job seeds.
    bool _survey_job = false;
    std::size_t _survey_pass = 0;

    // Deterministic-path state, set from _seed. Generate completions are
    // committed in ascending job id, with _commit_cursor as the next id due.
    bool _deterministic = false;
    std::set<std::size_t> _ready_gen;
    std::size_t _commit_cursor = 0;

    std::chrono::time_point<std::chrono::steady_clock> _start_time;
    std::size_t _start_cpu_microsec;
    std::chrono::time_point<std::chrono::steady_clock> _last_print_time;
    std::chrono::time_point<std::chrono::steady_clock> _last_status_time;
    PrettyBox _pretty_box_upper;
    PrettyBox _pretty_box_lower;
    std::string _status_file;
    std::unordered_map<std::string, TimingData> _timing_data;

    void survey_deterministic();
    void generate_deterministic();
    void commit_generate_job(GeneratorBatchJob& job);
    void commit_unweight_job(GeneratorBatchJob& job);
    void finish_channel_job(const GeneratorBatchJob& job);
    void register_dispatched_ids(std::size_t first_id, std::size_t end_id);
    std::size_t next_batch_event_count(std::size_t channel_index) const;
    std::size_t start_jobs();
    void update_integral();
    void update_integral_status();
    void update_integral_fractions();
    void update_counts();
    void reset_start_time();
    void add_timing_data(const std::string& key);
    void unweight_all();
    std::tuple<std::vector<CombineChannelData>, std::size_t, double> init_combine();
    void read_and_combine(
        std::vector<CombineChannelData>& channel_data,
        EventBuffer& buffer,
        double norm_factor,
        MixMaxRandom& rand_gen
    );
    void fill_lhe_event(
        LHECompleter& lhe_completer,
        LHEEvent& lhe_event,
        EventBuffer& buffer,
        std::size_t event_index,
        MixMaxRandom& rand_gen
    );

    void init_status(const std::string& status);
    void write_status(const std::string& status, bool force_write);

    void print_survey_init();
    void print_survey_update(
        bool done,
        std::size_t done_event_count,
        std::size_t total_event_count,
        std::size_t iter
    );
    void print_survey_update_pretty(
        bool done,
        std::size_t done_event_count,
        std::size_t total_event_count,
        std::size_t iter
    );
    void print_survey_update_log(
        bool done,
        std::size_t done_event_count,
        std::size_t total_event_count,
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

} // namespace madspace

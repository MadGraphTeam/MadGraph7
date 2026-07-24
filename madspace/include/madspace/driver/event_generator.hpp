#pragma once

#include <chrono>
#include <optional>
#include <random>
#include <set>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "madspace/compgraphs.hpp"
#include "madspace/driver/backend.hpp"
#include "madspace/driver/channel_generator.hpp"
#include "madspace/driver/discrete_optimizer.hpp"
#include "madspace/driver/format.hpp"
#include "madspace/driver/generator_data.hpp"
#include "madspace/driver/io.hpp"
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
        const GeneratorConfig& config = default_config
    );
    void survey();
    void generate();
    void combine_to_compact_npy(const std::string& file_name);
    void combine_to_lhe_npy(const std::string& file_name, LHECompleter& lhe_completer);
    void combine_to_lhe(const std::string& file_name, LHECompleter& lhe_completer);
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
    ResultQueue _result_queue;

    // Deterministic-path state. _deterministic is set from the context seed.
    //
    // survey_deterministic commits in a single global stream: completions buffered in
    // _ready_gen, committed in ascending job id via _commit_cursor.
    //
    // generate_deterministic commits one stream per channel so a slow channel cannot
    // stall the others: each channel's jobs commit in ascending channel_seq. Per
    // channel, _channel_next_seq is the next dispatch sequence, _channel_commit_cursor
    // the next to commit, and _channel_pending holds completed-but-not-yet-committed
    // jobs (channel_seq -> job_id). Integral fractions (hence per-channel targets) are
    // frozen while _freeze_fractions is set, so the dispatch/stop decisions depend only
    // on each channel's own committed state, not on cross-channel commit interleaving.
    bool _deterministic = false;
    std::set<std::size_t> _ready_gen;
    std::size_t _commit_cursor = 0;
    std::vector<std::size_t> _channel_commit_cursor;
    std::vector<std::size_t> _channel_next_seq;
    std::vector<std::unordered_map<std::size_t, std::size_t>> _channel_pending;
    bool _freeze_fractions = false;

    std::chrono::time_point<std::chrono::steady_clock> _start_time;
    std::size_t _start_cpu_microsec;
    std::chrono::time_point<std::chrono::steady_clock> _last_print_time;
    std::chrono::time_point<std::chrono::steady_clock> _last_status_time;
    PrettyBox _pretty_box_upper;
    PrettyBox _pretty_box_lower;
    std::string _status_file;
    std::unordered_map<std::string, TimingData> _timing_data;

    // Deterministic generation path (enabled when the context seed is non-zero):
    // expensive matrix-element jobs still run on the pool, but their results are
    // committed strictly in job-id order with unweighting folded in synchronously,
    // so the whole computation is a deterministic function of the seed at a fixed
    // thread count.
    void survey_deterministic();
    void generate_deterministic();
    void commit_generate_deterministic(GeneratorBatchJob& job);

    bool start_jobs();
    void update_integral();
    void update_counts();
    void reset_start_time();
    void add_timing_data(const std::string& key);
    void unweight_all();
    std::tuple<std::vector<CombineChannelData>, std::size_t, double> init_combine();
    void read_and_combine(
        std::vector<CombineChannelData>& channel_data,
        EventBuffer& buffer,
        double norm_factor,
        std::mt19937& rand_gen
    );
    void fill_lhe_event(
        LHECompleter& lhe_completer,
        LHEEvent& lhe_event,
        EventBuffer& buffer,
        std::size_t event_index,
        std::mt19937& rand_gen
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

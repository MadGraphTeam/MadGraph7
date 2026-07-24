#pragma once

#include <chrono>
#include <optional>
#include <random>
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
        std::uint64_t seed = 0
    );
    // `survey_pass` distinguishes independent survey() calls that end up
    // scheduling jobs on the same ChannelEventGenerator (e.g. an initial
    // multichannel survey followed by a post-simplification re-survey of a channel
    // that got carried over unchanged): each pass gets its own seed salt, so a
    // pass's job seeds depend only on its own logical identity, never on how many
    // jobs a previous, unrelated survey pass happened to schedule first.
    void survey(std::size_t survey_pass = 0);
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
    // True while a channel has a steady-state (non-VEGAS) generation batch
    // dispatched but not yet fully committed. Mirrors _channel_optimizing's role
    // for VEGAS batches: while true, no new batch is created for this channel, so
    // the next batch's size (see next_batch_job_count()) is always computed from
    // fully-committed data. Without this, jobs already dispatched but not yet
    // committed are invisible to the "does this channel need more work" decision,
    // so a burst of them can all commit after the channel's target was already met
    // -- overshooting it by an amount that depends on real scheduling timing rather
    // than on the seed.
    std::vector<bool> _channel_batch_pending;
    // generate_deterministic() only: per-channel analogue of _ready_gen/
    // _commit_cursor. Each channel's jobs still commit strictly in that channel's
    // own ascending job-id order (out-of-order arrivals buffered in
    // _channel_ready_gen until the cursor catches up), streamed as soon as they're
    // next in line rather than deferred -- but channels no longer block each
    // other's commits, so a channel with much costlier jobs (e.g. higher final-
    // state multiplicity in an inclusive run) can't head-of-line-block the rest.
    // _channel_cursor_set tracks whether _channel_commit_cursor has been
    // initialized for the channel's current round yet; see
    // register_dispatched_ids().
    std::vector<std::set<std::size_t>> _channel_ready_gen;
    std::vector<std::size_t> _channel_commit_cursor;
    std::vector<bool> _channel_cursor_set;
    ResultQueue _result_queue;

    // Base seed for reproducible event generation. 0 means "seed
    // non-deterministically" (the historical default); see seeded_rng().
    std::uint64_t _seed;

    // Job-scheduling context for the currently running survey()/generate() call,
    // read by start_jobs() when it hands a job's seed derivation off to its
    // ChannelEventGenerator. Set once at the top of survey()/generate(), before
    // the scheduling loop starts.
    bool _survey_job = false;
    std::size_t _survey_pass = 0;

    // Deterministic-path state. _deterministic is set from _seed. Generate
    // completions are buffered in _ready_gen and committed in ascending job id, with
    // _commit_cursor pointing at the next job id to commit.
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

    // Deterministic generation path (enabled when the context seed is non-zero):
    // expensive matrix-element jobs still run on the pool, but their results are
    // committed strictly in job-id order with unweighting folded in synchronously,
    // so the whole computation is a deterministic function of the seed at a fixed
    // thread count.
    void survey_deterministic();
    void generate_deterministic();
    void commit_generate_deterministic(GeneratorBatchJob& job);
    // generate_deterministic() only: scans the job ids newly added to
    // _running_jobs by the last start_jobs() call ([first_id, end_id)) and, for
    // any channel seen for the first time since its current round started (i.e.
    // _channel_cursor_set is still false), initializes _channel_commit_cursor to
    // that job's id. Safe because a channel's whole round is always dispatched
    // atomically as one contiguous id range (see GeneratorBatchJob::vegas_batch_size),
    // so the first id encountered for a channel in ascending order is always that
    // round's minimum.
    void register_dispatched_ids(std::size_t first_id, std::size_t end_id);

    // Size a new steady-state generation batch for a channel from fully-committed
    // data only: the channel's own observed efficiency so far (count_unweighted /
    // count_opt, or the pessimistic 1.0 -- i.e. a whole raw batch -- before any data
    // exists yet) and its remaining need (target - count_unweighted). Capped to
    // generation_batch_fraction of the remaining need so a single batch never bets
    // everything on an estimate that may still be noisy, and floored at
    // min_batch_jobs so a lone active channel still gets enough parallel jobs to
    // keep all worker threads busy.
    std::size_t next_batch_job_count(std::size_t channel_index, double target) const;

    bool start_jobs();
    // update_integral() = update_integral_status() + update_integral_fractions().
    // Split so generate_deterministic() can keep the status half (used for
    // logging/progress only) fully live, per commit, while deferring the fractions
    // half (which feeds back into per-channel scheduling decisions, see
    // next_batch_job_count()) to once per round -- see commit_generate_deterministic().
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

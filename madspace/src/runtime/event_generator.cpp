#include "madspace/runtime/event_generator.h"
#include "madspace/runtime/logger.h"

#include <cmath>
#include <filesystem>
#include <format>
#include <ranges>
#include <sys/resource.h>

#include "madspace/util.h"

using namespace madspace;

namespace {

std::size_t cpu_time_microsec() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return 1000000 * (usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) +
        usage.ru_utime.tv_usec + usage.ru_stime.tv_usec;
}

} // namespace

const GeneratorConfig EventGenerator::default_config = {};

EventGenerator::EventGenerator(
    const std::vector<ContextPtr>& contexts,
    const std::vector<ChannelEventGenerator>& channels,
    const std::string& status_file,
    const GeneratorConfig& config
) :
    _config(config),
    _status{
        .index = 0,
        .subprocess = 0,
        .name = "",
        .mean = 0.,
        .error = 0.,
        .rel_std_dev = 0.,
        .count = 0,
        .count_opt = 0,
        .count_after_cuts = 0,
        .count_after_cuts_opt = 0,
        .count_unweighted = 0.,
        .count_target = static_cast<double>(config.target_count),
        .optimized = false,
        .done = false
    },
    _contexts(contexts),
    _job_id(0),
    _channel_job_counts(channels.size()),
    _context_job_counts(contexts.size()),
    _status_file(status_file) {}

void EventGenerator::survey() {
    reset_start_time();
    bool done = false;
    std::size_t min_iters = _config.survey_min_iters;
    std::size_t max_iters = std::max(min_iters, _config.survey_max_iters);
    double target_precision = _config.survey_target_precision;

    std::vector<ThreadPool*> thread_pools;
    for (auto& context : _contexts) {
        thread_pools.push_back(&context->thread_pool());
    }
    MultiThreadPool multi_thread_pool(thread_pools);

    std::size_t total_job_count = 0;
    std::size_t done_job_count = 0;
    for (auto& channel : _channels) {
        std::size_t chan_batch_size = channel._batch_size;
        for (std::size_t iter = channel._status.iterations; iter < min_iters; ++iter) {
            total_job_count +=
                (chan_batch_size + _config.batch_size - 1) / _config.batch_size;
            chan_batch_size = std::min(chan_batch_size * 2, _config.max_batch_size);
        }
    }
    print_survey_init();

    std::size_t iter = 0;
    for (; !done && iter < max_iters; ++iter) {
        std::size_t job_count_before = _running_jobs.size();
        for (auto& channel : _channels) {
            if (channel.status().iterations > iter) {
                continue;
            }
            if (iter >= min_iters &&
                channel._cross_section.rel_error() < target_precision) {
                continue;
            }
            channel.build_vegas_jobs(_ready_jobs, iter >= min_iters - 1);
        }
        start_jobs();
        if (iter >= min_iters) {
            total_job_count += _running_jobs.size() - job_count_before;
        }
        done = true;
        while (auto job_id = multi_thread_pool.wait()) {
            _abort_check_function();
            auto& job = _running_jobs.at(*job_id);
            auto& channel = _channels.at(job.channel_index);
            auto& channel_job_count = _channel_job_counts.at(job.channel_index);
            if (channel_job_count == job.vegas_job_count) {
                channel.clear_events();
            }
            --channel_job_count;
            ++done_job_count;

            channel.integrate_and_optimize(job, channel_job_count == 0);
            update_integral();

            if (iter >= min_iters - 1) {
                channel.update_max_weight(job.weights);
                channel.unweight_and_write(job.unweighted_events);
                update_counts();
                if (channel_job_count == 0 &&
                    channel._cross_section.rel_error() < target_precision) {
                    done = false;
                }
            } else {
                done = false;
            }
            _running_jobs.erase(*job_id);
            print_survey_update(false, done_job_count, total_job_count, iter);
        }
    }
    print_survey_update(true, done_job_count, total_job_count, iter - 1);
}

void EventGenerator::generate() {
    reset_start_time();
    print_gen_init();

    std::vector<ThreadPool*> thread_pools;
    for (auto& context : _contexts) {
        thread_pools.push_back(&context->thread_pool());
    }
    MultiThreadPool multi_thread_pool(thread_pools);
    std::size_t channel_index = 0;
    while (true) {
        _abort_check_function();

        std::size_t job_count_before;
        do {
            job_count_before = _running_jobs.size();
            for (std::size_t i = 0;
                 i < _channels.size() && _running_jobs.size() < target_job_count;
                 ++i, channel_index = (channel_index + 1) % _channels.size()) {
                auto& channel = _channels.at(channel_index);
                if (channel._status.count_unweighted >=
                    channel._integral_fraction * _config.target_count) {
                    continue;
                }
                if ((channel._vegas_optimizer || channel._discrete_optimizer) &&
                    channel._needs_optimization) {
                    if (_channel_job_counts.at(i) == 0) {
                        channel.build_vegas_jobs(_ready_jobs, true);
                    }
                } else {
                    _ready_jobs.push_back({
                        .channel_index = i,
                        .unweight = true,
                        .batch_size = _config.batch_size,
                        .vegas_job_count = 0,
                    });
                }
            }
        } while (_running_jobs.size() - job_count_before > 0);

        if (auto job_id = multi_thread_pool.wait()) {
            auto& job = _running_jobs.at(*job_id);
            auto& channel = _channels.at(job.channel_index);
            auto& channel_job_count = _channel_job_counts.at(job.channel_index);
            bool run_optim =
                (channel._vegas_optimizer || channel._discrete_optimizer) &&
                channel._needs_optimization;
            if (run_optim && channel_job_count == job.vegas_job_count) {
                channel.clear_events();
            }
            --channel_job_count;

            channel.integrate_and_optimize(job, run_optim);
            update_integral();
            channel.update_max_weight(job.weights);
            channel.unweight_and_write(job.unweighted_events);
            update_counts();
            print_gen_update(false);
            _running_jobs.erase(*job_id);
        } else {
            if (_status.done) {
                unweight_all();
            }
            if (_status.done) {
                break;
            }
        }
    }
    print_gen_update(true);
}

bool EventGenerator::start_jobs() {
    std::size_t ready_index = 0;
    std::size_t context_index = 0;
    for (auto [context, job_count] : zip(_contexts, _context_job_counts)) {
        // fill the queue to twice the thread count to keep the worker threads busy
        std::size_t target_count = 2 * context->thread_pool().thread_count();
        for (; job_count < target_count; ++job_count) {
            if (ready_index == _ready_jobs.size()) {
                break;
            }
            auto& job =
                std::get<0>(_running_jobs.emplace(_job_id, _ready_jobs.at(ready_index)))
                    ->second;
            job.job_id = _job_id;
            job.context_index = context_index;
            _channels.at(job.channel_index).start_job(job);
            ++_job_id;
            ++_channel_job_counts.at(job.channel_index);
            ++ready_index;
        }
        if (ready_index == _ready_jobs.size()) {
            break;
        }
        ++context_index;
    }
    _ready_jobs.erase(_ready_jobs.begin(), _ready_jobs.begin() + ready_index);
    return ready_index > 0;
}

void EventGenerator::update_integral() {
    double total_mean = 0., total_var = 0.;
    std::size_t total_count = 0, total_count_opt = 0;
    std::size_t total_count_after_cuts = 0, total_count_after_cuts_opt = 0;
    std::size_t total_integ_count = 0;
    std::size_t iterations = 0;
    bool optimized = true;
    for (auto& channel : _channels) {
        total_mean += channel._cross_section.mean();
        total_var += channel._cross_section.variance() / channel._cross_section.count();
        total_count += channel.status().count;
        total_count_opt += channel.status().count_opt;
        total_count_after_cuts += channel.status().count_after_cuts;
        total_count_after_cuts_opt += channel.status().count_after_cuts_opt;
        total_integ_count += channel._cross_section.count();
        iterations = std::max(channel._status.iterations, iterations);
        if (channel._needs_optimization) {
            optimized = false;
        }
    }
    _status.mean = total_mean;
    _status.error = std::sqrt(total_var);
    _status.rel_std_dev = std::sqrt(total_var * total_integ_count) / total_mean;
    _status.count = total_count;
    _status.count_opt = total_count_opt;
    _status.count_after_cuts = total_count_after_cuts;
    _status.count_after_cuts_opt = total_count_after_cuts_opt;
    _status.iterations = iterations;
    _status.optimized = optimized;
    for (auto& channel : _channels) {
        channel._integral_fraction = channel._cross_section.mean() / total_mean;
    }
}

void EventGenerator::update_counts() {
    double total_eff_count = 0.;
    bool done = true;
    for (auto& channel : _channels) {
        double chan_target = channel._integral_fraction * _config.target_count;
        if (channel._status.count_unweighted < chan_target) {
            total_eff_count += _status.count_unweighted;
            done = false;
        } else {
            total_eff_count += chan_target;
        }
    }
    _status.count_unweighted = total_eff_count;
    _status.done = done;
}

void EventGenerator::combine_to_compact_npy(const std::string& file_name) {
    reset_start_time();
    auto [channel_data, particle_count, norm_factor] = init_combine();
    EventBuffer buffer(
        0, particle_count, DataLayout::of<EventFullRecord, ParticleRecord>()
    );
    EventFile event_file(
        file_name,
        DataLayout::of<EventFullRecord, ParticleRecord>(),
        particle_count,
        EventFile::create
    );
    std::size_t event_count = 0;
    std::size_t last_update_count = 0;
    print_combine_init();
    while (true) {
        _abort_check_function();
        read_and_combine(channel_data, buffer, norm_factor);
        if (buffer.event_count() == 0) {
            break;
        }
        event_count += buffer.event_count();
        if (event_count - last_update_count > 10000) {
            print_combine_update(event_count);
            last_update_count = event_count;
        }
        event_file.write(buffer);
    }
    print_combine_update(_config.target_count);
}

void EventGenerator::combine_to_lhe_npy(
    const std::string& file_name, LHECompleter& lhe_completer
) {
    reset_start_time();
    std::random_device rand_device;
    std::mt19937 rand_gen(rand_device());
    auto [channel_data, particle_count, norm_factor] = init_combine();
    EventBuffer buffer(
        0, particle_count, DataLayout::of<EventFullRecord, ParticleRecord>()
    );
    EventBuffer buffer_out(
        0,
        lhe_completer.max_particle_count(),
        DataLayout::of<PackedLHEEvent, PackedLHEParticle>()
    );
    EventFile event_file(
        file_name,
        DataLayout::of<PackedLHEEvent, PackedLHEParticle>(),
        lhe_completer.max_particle_count(),
        EventFile::create
    );
    std::size_t event_count = 0;
    std::size_t last_update_count = 0;
    LHEEvent lhe_event;
    print_combine_init();
    while (true) {
        _abort_check_function();
        read_and_combine(channel_data, buffer, norm_factor);
        if (buffer.event_count() == 0) {
            break;
        }
        event_count += buffer.event_count();
        buffer_out.resize(buffer.event_count());
        for (std::size_t i = 0; i < buffer.event_count(); ++i) {
            fill_lhe_event(lhe_completer, lhe_event, buffer, i, rand_gen);
            buffer_out.event<PackedLHEEvent>(i).from_lhe_event(lhe_event);
            std::size_t j = 0;
            for (; j < lhe_event.particles.size(); ++j) {
                buffer_out.particle<PackedLHEParticle>(i, j).from_lhe_particle(
                    lhe_event.particles[j]
                );
            }
            for (; j < lhe_completer.max_particle_count(); ++j) {
                buffer_out.particle<PackedLHEParticle>(i, j).from_lhe_particle(
                    LHEParticle{}
                );
            }
        }
        event_file.write(buffer_out);
        if (event_count - last_update_count > 10000) {
            print_combine_update(event_count);
            last_update_count = event_count;
        }
    }
    print_combine_update(_config.target_count);
}

void EventGenerator::combine_to_lhe(
    const std::string& file_name, LHECompleter& lhe_completer
) {
    reset_start_time();
    ThreadPool pool(_config.combine_thread_count);
    ThreadResource<std::mt19937> rand_gens(pool, []() {
        std::random_device rand_device;
        return std::mt19937(rand_device());
    });
    auto [channel_data, particle_count, norm_factor] = init_combine();
    std::vector<std::pair<EventBuffer, std::string>> buffers;
    std::vector<std::size_t> idle_buffers;
    for (std::size_t i = 0; i < 2 * pool.thread_count(); ++i) {
        buffers.push_back(
            {{0, particle_count, DataLayout::of<EventFullRecord, ParticleRecord>()}, {}}
        );
        idle_buffers.push_back(i);
    }
    LHEFileWriter event_file(file_name, LHEMeta{});
    std::size_t event_count = 0;
    std::size_t last_update_count = 0;
    bool done = false;
    print_combine_init();
    while (true) {
        _abort_check_function();
        while (idle_buffers.size() > 0 && !done) {
            std::size_t job_id = idle_buffers.back();
            auto& [in_buffer, out_buffer] = buffers.at(job_id);
            read_and_combine(channel_data, in_buffer, norm_factor);
            if (in_buffer.event_count() == 0) {
                done = true;
                break;
            }
            idle_buffers.pop_back();
            pool.submit(
                [job_id, this, &in_buffer, &out_buffer, &lhe_completer, &rand_gens] {
                    LHEEvent lhe_event;
                    out_buffer.clear();
                    for (std::size_t i = 0; i < in_buffer.event_count(); ++i) {
                        fill_lhe_event(
                            lhe_completer,
                            lhe_event,
                            in_buffer,
                            i,
                            rand_gens.get(ThreadPool::thread_index())
                        );
                        lhe_event.format_to(out_buffer);
                    }
                    return job_id;
                }
            );
        }

        auto done_jobs = pool.wait_multiple();
        for (std::size_t job_id : done_jobs) {
            auto& [in_buffer, out_buffer] = buffers.at(job_id);
            idle_buffers.push_back(job_id);
            event_file.write_string(out_buffer);
            event_count += in_buffer.event_count();
            if (event_count - last_update_count > 10000) {
                print_combine_update(event_count);
                last_update_count = event_count;
            }
        }
        if (done_jobs.size() == 0 && done) {
            break;
        }
    }
    print_combine_update(_config.target_count);
}

void EventGenerator::reset_start_time() {
    _start_time = std::chrono::steady_clock::now();
    _start_cpu_microsec = cpu_time_microsec();
}

void EventGenerator::add_timing_data(const std::string& key) {
    using namespace std::chrono_literals;
    std::size_t diff = cpu_time_microsec() - _start_cpu_microsec;
    _timing_data[key] = EventGenerator::TimingData{
        .wall_time_sec = static_cast<double>(
            (std::chrono::steady_clock::now() - _start_time) / 1.0s
        ),
        .cpu_time_sec = diff / 1e6,
    };
}

std::string EventGenerator::format_run_time(const std::string& key) const {
    using namespace std::chrono_literals;
    auto [wall_time_sec, cpu_time_sec] = _timing_data.at(key);
    std::chrono::duration<double> cpu_duration(cpu_time_sec),
        wall_duration(wall_time_sec);
    // we don't use the ratio feature of duration here because it seems to lead
    // to errors in old gcc versions
    double cpu_centisec = std::fmod(cpu_time_sec / 0.01, 100.);
    double wall_centisec = std::fmod(wall_time_sec / 0.01, 100.);
    return std::format(
        "{:%H:%M:%S}.{:02.0f} wall, {:%H:%M:%S}.{:02.0f} cpu",
        wall_duration,
        wall_centisec,
        cpu_duration,
        cpu_centisec
    );
}

void EventGenerator::unweight_all() {
    std::random_device rand_device;
    std::mt19937 rand_gen(rand_device());
    bool done = true;
    double total_eff_count = 0.;
    for (auto& channel : _channels) {
        channel.unweight_file(rand_gen);

        double chan_target = channel._integral_fraction * _config.target_count;
        if (channel._status.count_unweighted < chan_target) {
            total_eff_count += channel._status.count_unweighted;
            done = false;
        } else {
            total_eff_count += chan_target;
        }
    }
    _status.count_unweighted = total_eff_count;
    _status.done = done;
}

std::vector<GeneratorStatus> EventGenerator::channel_status() const {
    std::vector<GeneratorStatus> status;
    for (auto& channel : _channels) {
        // double target_count = channel.integral_fraction * _config.target_count;
        status.push_back(channel.status());
    }
    return status;
}

std::vector<Histogram> EventGenerator::histograms() const {
    std::vector<Histogram> hists = _channels.at(0)._histograms;
    for (auto& channel : _channels) {
        for (auto [chan_hist, out_hist] : zip(channel._histograms, hists)) {
            for (auto [chan_w, chan_w2, val, err] :
                 zip(chan_hist.bin_values,
                     chan_hist.bin_errors,
                     out_hist.bin_values,
                     out_hist.bin_errors)) {
                auto n = channel._status.count_opt;
                val += chan_w / n;
                err += (chan_w2 - chan_w * chan_w / n) / (n * n);
            }
        }
    }
    for (auto& hist : hists) {
        for (double& err : hist.bin_errors) {
            err = std::sqrt(err);
        }
    }
    return hists;
}

std::tuple<std::vector<EventGenerator::CombineChannelData>, std::size_t, double>
EventGenerator::init_combine() {
    std::vector<EventGenerator::CombineChannelData> channel_data;
    std::size_t count_sum = 0;
    std::size_t particle_count = 0;
    double weight_sum = 0.;
    for (auto& channel : _channels) {
        particle_count = std::max(particle_count, channel._event_file.particle_count());
        std::size_t count =
            std::round(channel._integral_fraction * _config.target_count);
        count_sum += count;
        channel._event_file.seek(0);
        weight_sum += channel.channel_weight_sum(count);
        channel._weight_file.seek(0);
        channel_data.push_back({
            .cum_count = count_sum,
            .event_buffer = EventBuffer(
                0,
                channel._event_file.particle_count(),
                DataLayout::of<EventIndicesRecord, ParticleRecord>()
            ),
            .weight_buffer = EventBuffer(
                0, 0, DataLayout::of<EventWeightRecord, EmptyParticleRecord>()
            ),
            .buffer_index = 0,
        });
    }
    return {channel_data, particle_count, _status.mean * count_sum / weight_sum};
}

void EventGenerator::read_and_combine(
    std::vector<EventGenerator::CombineChannelData>& channel_data,
    EventBuffer& buffer,
    double norm_factor
) {
    std::size_t batch_size = 1000;
    std::size_t event_count = std::min(batch_size, channel_data.back().cum_count);
    buffer.resize(event_count);

    std::random_device rand_device;
    std::mt19937 rand_gen(rand_device());
    for (std::size_t event_index = 0; event_index < event_count; ++event_index) {
        std::size_t random_index = std::uniform_int_distribution<
            std::size_t>(0, channel_data.back().cum_count - 1)(rand_gen);
        auto sampled_chan = std::lower_bound(
            channel_data.begin(),
            channel_data.end(),
            random_index,
            [](auto& chan, std::size_t val) { return chan.cum_count < val; }
        );
        std::for_each(sampled_chan, channel_data.end(), [](auto& chan) {
            --chan.cum_count;
        });
        auto& channel = _channels.at(sampled_chan - channel_data.begin());

        double weight = 0.;
        while (true) {
            if (sampled_chan->buffer_index ==
                sampled_chan->event_buffer.event_count()) {
                channel._event_file.read(sampled_chan->event_buffer, batch_size);
                channel._weight_file.read(sampled_chan->weight_buffer, batch_size);
                sampled_chan->buffer_index = 0;
            }
            weight = sampled_chan->weight_buffer
                         .event<EventWeightRecord>(sampled_chan->buffer_index)
                         .weight();
            if (weight != 0.) {
                break;
            }
            ++sampled_chan->buffer_index;
        }

        auto event_in = sampled_chan->event_buffer.event<EventIndicesRecord>(
            sampled_chan->buffer_index
        );
        auto event_out = buffer.event<EventFullRecord>(event_index);
        event_out.weight() = std::max(1., weight / channel._max_weight) * norm_factor;
        event_out.subprocess_index() = channel._status.subprocess;
        event_out.diagram_index() = event_in.diagram_index();
        event_out.color_index() = event_in.color_index();
        event_out.flavor_index() = event_in.flavor_index();
        event_out.helicity_index() = event_in.helicity_index();

        std::size_t i = 0;
        for (; i < sampled_chan->event_buffer.particle_count(); ++i) {
            auto particle_in = sampled_chan->event_buffer.particle<ParticleRecord>(
                sampled_chan->buffer_index, i
            );
            auto particle_out = buffer.particle<ParticleRecord>(event_index, i);
            particle_out.energy() = particle_in.energy();
            particle_out.px() = particle_in.px();
            particle_out.py() = particle_in.py();
            particle_out.pz() = particle_in.pz();
        }
        for (; i < buffer.particle_count(); ++i) {
            auto particle_out = buffer.particle<ParticleRecord>(event_index, i);
            particle_out.energy() = 0.;
            particle_out.px() = 0.;
            particle_out.py() = 0.;
            particle_out.pz() = 0.;
        }
        ++sampled_chan->buffer_index;
    }
}

void EventGenerator::fill_lhe_event(
    LHECompleter& lhe_completer,
    LHEEvent& lhe_event,
    EventBuffer& buffer,
    std::size_t event_index,
    std::mt19937& rand_gen
) {
    EventRecord event_in = buffer.event<EventFullRecord>(event_index);
    lhe_event.weight = event_in.weight();
    lhe_event.process_id = 0;
    lhe_event.scale = 0; // TODO: populate these
    lhe_event.alpha_qed = 0;
    lhe_event.alpha_qcd = 0;
    lhe_event.particles.clear();
    for (std::size_t i = 0; i < buffer.particle_count(); ++i) {
        auto particle_in = buffer.particle<ParticleRecord>(event_index, i);
        if (particle_in.energy() == 0.) {
            break;
        }
        lhe_event.particles.push_back(
            LHEParticle{
                .px = particle_in.px(),
                .py = particle_in.py(),
                .pz = particle_in.pz(),
                .energy = particle_in.energy(),
            }
        );
    }
    lhe_completer.complete_event_data(
        lhe_event,
        event_in.subprocess_index(),
        event_in.diagram_index(),
        event_in.color_index(),
        event_in.flavor_index(),
        event_in.helicity_index(),
        rand_gen
    );
}

void EventGenerator::init_status(const std::string& status) {
    _last_status_time = std::chrono::steady_clock::now();
    write_status(status, true);
}

void EventGenerator::write_status(const std::string& status, bool force_write) {
    auto now = std::chrono::steady_clock::now();
    using namespace std::chrono_literals;
    if (now - _last_status_time < 10s && !force_write) {
        return;
    }
    _last_status_time = now;

    std::string status_tmp_file = std::format("{}.tmp", _status_file);
    std::ofstream f(status_tmp_file);
    nlohmann::json j{
        {"status", status},
        {"process", _status},
        {"channels", channel_status()},
        {"run_times", _timing_data},
        {"histograms", histograms()},
    };
    f << j.dump();
    // rename atomically deletes the old file and replaces it with the new one
    // such that the status file exists at all times
    std::filesystem::rename(status_tmp_file, _status_file);
}

void EventGenerator::print_survey_init() {
    init_status("survey");
    _last_print_time = std::chrono::steady_clock::now();
    if (_config.verbosity != GeneratorConfig::pretty) {
        Logger::info("survey started");
        return;
    }
    _pretty_box_upper = PrettyBox("Survey", 4, {18, 0});
    _pretty_box_upper.set_column(
        0, {"Iteration:", "Result:", "Number of events:", "Run time:"}
    );
    _pretty_box_upper.print_first();
}

void EventGenerator::print_survey_update(
    bool done, std::size_t done_job_count, std::size_t total_job_count, std::size_t iter
) {
    if (done) {
        add_timing_data("survey");
    }
    write_status("survey", done);
    if (_config.verbosity == GeneratorConfig::pretty) {
        print_survey_update_pretty(done, done_job_count, total_job_count, iter);
    } else if (_config.verbosity == GeneratorConfig::log) {
        print_survey_update_log(done, done_job_count, total_job_count, iter);
    }
}

void EventGenerator::print_survey_update_pretty(
    bool done, std::size_t done_job_count, std::size_t total_job_count, std::size_t iter
) {
    std::string int_str = format_with_error(_status.mean, _status.error);
    std::string count_str = std::format(
        "{} before cuts, {} after",
        format_si_prefix(_status.count),
        format_si_prefix(_status.count_after_cuts)
    );
    if (done) {
        _pretty_box_upper.set_column(
            1,
            {std::format("{}", iter + 1), int_str, count_str, format_run_time("survey")}
        );
    } else {
        auto now = std::chrono::steady_clock::now();
        using namespace std::chrono_literals;
        if (now - _last_print_time < 0.1s) {
            return;
        }
        _last_print_time = now;

        _pretty_box_upper.set_column(
            1,
            {std::format(
                 "{:<15} {}",
                 iter + 1,
                 format_progress(
                     static_cast<double>(done_job_count) / total_job_count, 52
                 )
             ),
             int_str,
             count_str,
             std::format(
                 "{:%H:%M:%S}",
                 std::chrono::round<std::chrono::seconds>(now - _start_time)
             )}
        );
    }
    _pretty_box_upper.print_update();
}

void EventGenerator::print_survey_update_log(
    bool done, std::size_t done_job_count, std::size_t total_job_count, std::size_t iter
) {
    auto now = std::chrono::steady_clock::now();
    using namespace std::chrono_literals;
    if (now - _last_print_time < 10s && !done) {
        return;
    }
    _last_print_time = now;

    Logger::info(
        std::format(
            "survey, iter: {}, integral: {}, samps: {}, samps. after cuts: {}, "
            "time: {:%H:%M:%S}",
            iter + 1,
            format_with_error(_status.mean, _status.error),
            format_si_prefix(_status.count),
            format_si_prefix(_status.count_after_cuts),
            std::chrono::round<std::chrono::seconds>(now - _start_time)
        )
    );

    if (done) {
        Logger::info(std::format("survey done, {}", format_run_time("survey")));
    }
}

void EventGenerator::print_gen_init() {
    init_status("generate");
    _last_print_time = std::chrono::steady_clock::now();
    if (_config.verbosity != GeneratorConfig::pretty) {
        Logger::info("generating started");
        return;
    }

    std::size_t offset = 0;
    if (_channels.size() > 1) {
        _pretty_box_lower = PrettyBox(
            "Individual channels",
            _channels.size() < 21 ? _channels.size() + 1 : 22,
            {4, 16, 9, 9, 7, 7, 0}
        );
        _pretty_box_lower.set_row(
            0, {"#", "integral ↓", "RSD", "uweff", "N", "opt", "unweighted"}
        );
        if (_channels.size() > 20) {
            _pretty_box_lower.set_cell(21, 0, "..");
        }
        offset = _pretty_box_lower.line_count();
    }
    _pretty_box_upper = PrettyBox("Integration and unweighting", 7, {19, 0}, offset);
    _pretty_box_upper.set_column(
        0,
        {"Result:",
         "Rel. error:",
         "Rel. stddev:",
         "Number of events:",
         "Unweighting eff.:",
         "Unweighted events:",
         "Run time:"}
    );
    _pretty_box_upper.print_first();
    if (_channels.size() > 1) {
        _pretty_box_lower.print_first();
    }
}

void EventGenerator::print_gen_update(bool done) {
    if (done) {
        add_timing_data("generate");
    }
    write_status("generate", done);
    if (_config.verbosity == GeneratorConfig::pretty) {
        print_gen_update_pretty(done);
    } else if (_config.verbosity == GeneratorConfig::log) {
        print_gen_update_log(done);
    }
}

void EventGenerator::print_gen_update_pretty(bool done) {
    auto now = std::chrono::steady_clock::now();
    using namespace std::chrono_literals;
    if (now - _last_print_time < 0.1s && !done) {
        return;
    }
    _last_print_time = now;

    std::string int_str, rel_str, rsd_str, uweff_str, count_str;
    count_str = std::format(
        "{} before cuts, {} after",
        format_si_prefix(_status.count),
        format_si_prefix(_status.count_after_cuts)
    );

    if (!std::isnan(_status.error)) {
        double rel_err = _status.error / _status.mean;
        int_str = format_with_error(_status.mean, _status.error);
        rel_str = std::format("{:.4f} %", rel_err * 100);
        rsd_str = std::format("{:.3f}", _status.rel_std_dev);
        uweff_str = std::format(
            "{:.5f} before cuts, {:.5f} after",
            _status.count_unweighted / _status.count_opt,
            _status.count_unweighted / _status.count_after_cuts_opt
        );
    }
    std::string unw_str = std::format(
        "{} / {}",
        format_si_prefix(_status.count_unweighted),
        format_si_prefix(_status.count_target)
    );
    std::string time_str;
    if (done) {
        time_str = format_run_time("generate");
    } else {
        unw_str = std::format(
            "{:<15} {}",
            unw_str,
            format_progress(_status.count_unweighted / _status.count_target, 52)
        );
        time_str = std::format(
            "{:%H:%M:%S}", std::chrono::round<std::chrono::seconds>(now - _start_time)
        );
    }
    _pretty_box_upper.set_column(
        1, {int_str, rel_str, rsd_str, count_str, uweff_str, unw_str, time_str}
    );
    _pretty_box_upper.print_update();

    if (_channels.size() > 1) {
        auto channels = channel_status();
        std::sort(channels.begin(), channels.end(), [](auto& chan1, auto& chan2) {
            return chan1.mean > chan2.mean;
        });

        for (std::size_t row = 1; auto& channel : channels | std::views::take(20)) {
            std::string index_str = std::format("{}", channel.index);
            std::string int_str, rsd_str, count_str, unw_str, opt_str;
            if (!std::isnan(channel.error)) {
                int_str = format_with_error(channel.mean, channel.error);
                rsd_str = std::format("{:.3f}", channel.rel_std_dev);
                uweff_str = std::format(
                    "{:.5f}", channel.count_unweighted / channel.count_after_cuts_opt
                );
                count_str = format_si_prefix(channel.count_after_cuts);
                opt_str = std::format(
                    "{} {}",
                    channel.iterations,
                    channel.optimized || channel.done ? "✓" : ""
                );
                std::string unw_count_str = std::format(
                    "{:>5} / {:>5}",
                    format_si_prefix(channel.count_unweighted),
                    format_si_prefix(channel.count_target)
                );
                std::string progress;
                if (!_status.done) {
                    progress = format_progress(
                        channel.count_unweighted / channel.count_target, 19
                    );
                }
                unw_str = std::format("{:<15} {:<19}", unw_count_str, progress);
            }
            _pretty_box_lower.set_row(
                row,
                {index_str, int_str, rsd_str, uweff_str, count_str, opt_str, unw_str}
            );
            ++row;
        }
        _pretty_box_lower.print_update();
    }
}

void EventGenerator::print_gen_update_log(bool done) {
    auto now = std::chrono::steady_clock::now();
    using namespace std::chrono_literals;
    if (now - _last_print_time < 10s && !done) {
        return;
    }
    _last_print_time = now;

    Logger::info(
        std::format(
            "generating, events: {} / {}, integral: {}, rel. error: {:.4f} %, "
            "RSD: {:.3f}, samps: {}, samps. after cuts: {}, "
            "unw. eff.: {:.5f}, unw. eff. after cuts: {:.5f}, time: {:%H:%M:%S}",
            format_si_prefix(_status.count_unweighted),
            format_si_prefix(_status.count_target),
            format_with_error(_status.mean, _status.error),
            _status.error / _status.mean * 100,
            _status.rel_std_dev,
            format_si_prefix(_status.count),
            format_si_prefix(_status.count_after_cuts),
            _status.count_unweighted / _status.count_opt,
            _status.count_unweighted / _status.count_after_cuts_opt,
            std::chrono::round<std::chrono::seconds>(now - _start_time)
        )
    );

    if (done) {
        Logger::info(std::format("generating done, {}", format_run_time("generate")));
    }
}

void EventGenerator::print_combine_init() {
    init_status("combine");
    _last_print_time = std::chrono::steady_clock::now();
    if (_config.verbosity != GeneratorConfig::pretty) {
        Logger::info("combining started");
        return;
    }
    _pretty_box_upper = PrettyBox("Writing final output", 2, {10, 0});
    _pretty_box_upper.set_column(0, {"Events:", "Run time:"});
    _pretty_box_upper.print_first();
}

void EventGenerator::print_combine_update(std::size_t count) {
    if (count == _config.target_count) {
        add_timing_data("combine");
        write_status("done", true);
    } else {
        write_status("combine", false);
    }
    if (_config.verbosity == GeneratorConfig::pretty) {
        print_combine_update_pretty(count);
    } else if (_config.verbosity == GeneratorConfig::log) {
        print_combine_update_log(count);
    }
}

void EventGenerator::print_combine_update_pretty(std::size_t count) {
    if (count == _config.target_count) {
        _pretty_box_upper.set_column(
            1,
            {std::format(
                 "{:>5} / {:>5}",
                 format_si_prefix(count),
                 format_si_prefix(_config.target_count)
             ),
             format_run_time("combine")}
        );
    } else {
        auto now = std::chrono::steady_clock::now();
        using namespace std::chrono_literals;
        if (now - _last_print_time < 0.1s) {
            return;
        }
        _last_print_time = now;

        _pretty_box_upper.set_column(
            1,
            {std::format(
                 "{:>5} / {:>5}   {}",
                 format_si_prefix(count),
                 format_si_prefix(_config.target_count),
                 format_progress(static_cast<double>(count) / _config.target_count, 60)
             ),
             std::format(
                 "{:%H:%M:%S}",
                 std::chrono::round<std::chrono::seconds>(now - _start_time)
             )}
        );
    }
    _pretty_box_upper.print_update();
}

void EventGenerator::print_combine_update_log(std::size_t count) {
    auto now = std::chrono::steady_clock::now();
    using namespace std::chrono_literals;
    if (now - _last_print_time < 10s && count != _config.target_count) {
        return;
    }
    _last_print_time = now;

    Logger::info(
        std::format(
            "combining, events: {} / {}, time: {:%H:%M:%S}",
            format_si_prefix(count),
            format_si_prefix(_config.target_count),
            std::chrono::round<std::chrono::seconds>(now - _start_time)
        )
    );

    if (count == _config.target_count) {
        Logger::info(std::format("combining done, {}", format_run_time("combine")));
    }
}

void madspace::to_json(
    nlohmann::json& j, const EventGenerator::TimingData& timing_data
) {
    j = nlohmann::json{
        {"wall_time_sec", timing_data.wall_time_sec},
        {"cpu_time_sec", timing_data.cpu_time_sec},
    };
}

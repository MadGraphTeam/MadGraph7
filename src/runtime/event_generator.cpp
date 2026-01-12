#include "madevent/runtime/event_generator.h"
#include "madevent/runtime/logger.h"

#include <cmath>
#include <filesystem>
#include <format>
#include <ranges>
#include <sys/resource.h>

#include "madevent/util.h"

using namespace madevent;

namespace {

std::size_t cpu_time_microsec() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return 1000000 * (usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) +
        usage.ru_utime.tv_usec + usage.ru_stime.tv_usec;
}

} // namespace

const EventGenerator::Config EventGenerator::default_config = {};

EventGenerator::EventGenerator(
    ContextPtr context,
    const std::vector<Integrand>& channels,
    const std::string& temp_file_prefix,
    const std::string& status_file,
    const Config& config,
    const std::vector<std::size_t>& channel_subprocesses,
    const std::vector<std::string>& channel_names,
    const std::vector<ObservableHistograms>& channel_histograms
) :
    _context(context),
    _config(config),
    _unweighter(build_runtime(
        Unweighter({channels.at(0).return_types().begin(),
                    channels.at(0).return_types().begin() + 6})
            .function(),
        context,
        false
    )),
    _status_all(
        {0,
         0,
         "",
         0.,
         0.,
         0.,
         0,
         0,
         0,
         0,
         0.,
         static_cast<double>(config.target_count),
         false,
         false}
    ),
    _job_id(0),
    _status_file(status_file) {
    std::size_t i = 0;
    for (auto& channel : channels) {
        if (channel.flags() != integrand_flags) {
            throw std::invalid_argument(
                "Integrand flags must be sample | return_momenta | return_random | "
                "return_discrete"
            );
        }
        std::optional<VegasGridOptimizer> vegas_optimizer;
        RuntimePtr vegas_histogram = nullptr;
        if (const auto& name = channel.vegas_grid_name(); name) {
            vegas_optimizer = VegasGridOptimizer(context, *name, config.vegas_damping);
            VegasHistogram hist(channel.vegas_dimension(), channel.vegas_bin_count());
            vegas_histogram = build_runtime(hist.function(), context, false);
        }
        std::vector<std::string> prob_names;
        std::vector<std::size_t> option_counts;
        auto add_names = Overloaded{
            [&](const DiscreteSampler& sampler) {
                auto& names = sampler.prob_names();
                prob_names.insert(prob_names.end(), names.begin(), names.end());
                auto& opts = sampler.option_counts();
                option_counts.insert(option_counts.end(), opts.begin(), opts.end());
            },
            [](auto sampler) {}
        };
        std::visit(add_names, channel.discrete_before());
        std::visit(add_names, channel.discrete_after());
        std::optional<DiscreteOptimizer> discrete_optimizer;
        RuntimePtr discrete_histogram = nullptr;
        if (prob_names.size() > 0) {
            discrete_optimizer = DiscreteOptimizer(context, prob_names);
            DiscreteHistogram hist(option_counts);
            discrete_histogram = build_runtime(hist.function(), context, false);
        }
        RuntimePtr obs_histograms = nullptr;
        if (channel_histograms.size() > 0) {
            obs_histograms =
                build_runtime(channel_histograms.at(i).function(), context, false);
        }
        _channels.push_back({
            .index = i,
            .runtime = build_runtime(channel.function(), context, false),
            .event_file = EventFile(
                std::format("{}_channel{}_events.npy", temp_file_prefix, i),
                DataLayout::of<EventIndicesRecord, ParticleRecord>(),
                channel.particle_count(),
                EventFile::create,
                true
            ),
            .weight_file = EventFile(
                std::format("{}_channel{}_weights.npy", temp_file_prefix, i),
                DataLayout::of<EventWeightRecord, EmptyParticleRecord>(),
                0,
                EventFile::create,
                true
            ),
            .vegas_optimizer = vegas_optimizer,
            .vegas_histogram = std::move(vegas_histogram),
            .discrete_optimizer = discrete_optimizer,
            .discrete_histogram = std::move(discrete_histogram),
            .observable_histograms = std::move(obs_histograms),
            .batch_size = config.start_batch_size,
            .name =
                channel_names.size() > 0 ? channel_names.at(i) : std::format("{}", i),
            .subprocess_index =
                channel_subprocesses.size() > 0 ? channel_subprocesses.at(i) : 0,
        });
        ++i;
    }
    if (channel_histograms.size() > 0) {
        for (auto& item : channel_histograms.at(0).observables()) {
            _empty_histograms.push_back({
                .name = item.observable.name(),
                .min = item.min,
                .max = item.max,
                .bin_values = std::vector<double>(item.bin_count + 2),
                .bin_errors = std::vector<double>(item.bin_count + 2),
            });
        }
    }
}

void EventGenerator::survey() {
    reset_start_time();
    bool done = false;
    auto& thread_pool = default_thread_pool();
    std::size_t min_iters = _config.survey_min_iters;
    std::size_t max_iters = std::max(min_iters, _config.survey_max_iters);
    double target_precision = _config.survey_target_precision;

    std::size_t total_job_count = 0;
    std::size_t done_job_count = 0;
    for (auto& channel : _channels) {
        std::size_t chan_batch_size = channel.batch_size;
        for (std::size_t iter = 0; iter < min_iters; ++iter) {
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
            if (iter >= min_iters &&
                channel.cross_section.rel_error() < target_precision) {
                continue;
            }
            start_vegas_jobs(channel);
        }
        if (iter >= min_iters) {
            total_job_count += _running_jobs.size() - job_count_before;
        }
        done = true;
        while (auto job_id = thread_pool.wait()) {
            _abort_check_function();
            auto& job = _running_jobs.at(*job_id);
            auto& channel = _channels.at(job.channel_index);
            if (channel.job_count == job.vegas_job_count) {
                clear_channel(channel);
            }
            --channel.job_count;
            ++done_job_count;

            bool run_optim = channel.vegas_optimizer || channel.discrete_optimizer;
            auto [weights, events] =
                integrate_and_optimize(channel, job.events, run_optim);

            if (iter >= min_iters - 1) {
                update_max_weight(channel, weights);
                unweight_and_write(channel, events);
                if (channel.job_count == 0 &&
                    channel.cross_section.rel_error() < target_precision) {
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
    auto& thread_pool = default_thread_pool();
    std::size_t channel_index = 0;
    std::size_t target_job_count = 2 * thread_pool.thread_count();
    while (true) {
        _abort_check_function();

        std::size_t job_count_before;
        do {
            job_count_before = _running_jobs.size();
            for (std::size_t i = 0;
                 i < _channels.size() && _running_jobs.size() < target_job_count;
                 ++i, channel_index = (channel_index + 1) % _channels.size()) {
                auto& channel = _channels.at(channel_index);
                if (channel.eff_count >=
                    channel.integral_fraction * _config.target_count) {
                    continue;
                }
                if ((channel.vegas_optimizer || channel.discrete_optimizer) &&
                    channel.needs_optimization) {
                    if (channel.job_count == 0) {
                        start_vegas_jobs(channel);
                    }
                } else {
                    start_job(channel, _config.batch_size);
                }
            }
        } while (_running_jobs.size() - job_count_before > 0);

        if (auto job_id = thread_pool.wait()) {
            auto& job = _running_jobs.at(*job_id);
            auto& channel = _channels.at(job.channel_index);
            bool run_optim = (channel.vegas_optimizer || channel.discrete_optimizer) &&
                channel.needs_optimization;
            if (run_optim && channel.job_count == job.vegas_job_count) {
                clear_channel(channel);
            }
            --channel.job_count;

            auto [weights, events] =
                integrate_and_optimize(channel, job.events, run_optim);
            update_max_weight(channel, weights);
            unweight_and_write(channel, events);
            print_gen_update(false);
            _running_jobs.erase(*job_id);
        } else {
            if (_status_all.done) {
                unweight_all();
            }
            if (_status_all.done) {
                break;
            }
        }
    }
    print_gen_update(true);
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
            fill_lhe_event(lhe_completer, lhe_event, buffer, i);
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
    auto [channel_data, particle_count, norm_factor] = init_combine();
    auto& thread_pool = default_thread_pool();
    std::vector<std::pair<EventBuffer, std::string>> buffers;
    std::vector<std::size_t> idle_buffers;
    for (std::size_t i = 0; i < 2 * thread_pool.thread_count(); ++i) {
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
            thread_pool.submit([job_id, this, &in_buffer, &out_buffer, &lhe_completer] {
                LHEEvent lhe_event;
                out_buffer.clear();
                for (std::size_t i = 0; i < in_buffer.event_count(); ++i) {
                    fill_lhe_event(lhe_completer, lhe_event, in_buffer, i);
                    lhe_event.format_to(out_buffer);
                }
                return job_id;
            });
        }

        auto done_jobs = thread_pool.wait_multiple();
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
        unweight_channel(channel, rand_gen);

        double chan_target = channel.integral_fraction * _config.target_count;
        if (channel.eff_count < chan_target) {
            total_eff_count += channel.eff_count;
            done = false;
        } else {
            total_eff_count += chan_target;
        }
    }
    _status_all.count_unweighted = total_eff_count;
    _status_all.done = done;
}

void EventGenerator::unweight_channel(ChannelState& channel, std::mt19937 rand_gen) {
    std::size_t buf_size = 1000000;
    std::uniform_real_distribution<double> rand_dist;
    EventBuffer buffer(0, 0, DataLayout::of<EventWeightRecord, EmptyParticleRecord>());
    std::size_t accept_count = 0;
    for (std::size_t i = 0; i < channel.weight_file.event_count(); i += buf_size) {
        channel.weight_file.seek(i);
        channel.weight_file.read(buffer, buf_size);
        for (std::size_t j = 0; j < buffer.event_count(); ++j) {
            auto weight = buffer.event<EventWeightRecord>(j).weight();
            if (weight / channel.max_weight < rand_dist(rand_gen)) {
                weight = 0;
            } else {
                weight = std::max(weight.value(), channel.max_weight);
                ++accept_count;
            }
        }
        channel.weight_file.seek(i);
        channel.weight_file.write(buffer);
    }
    channel.eff_count = accept_count;
}

double
EventGenerator::channel_weight_sum(ChannelState& channel, std::size_t event_count) {
    std::size_t buf_size = 1000000;
    EventBuffer buffer(0, 0, DataLayout::of<EventWeightRecord, EmptyParticleRecord>());
    double weight_sum = 0;
    channel.weight_file.seek(0);
    std::size_t unweighted_count = 0;
    for (std::size_t i = 0; i < channel.weight_file.event_count(); i += buf_size) {
        channel.weight_file.read(buffer, buf_size);
        bool done = false;
        for (std::size_t j = 0; j < buffer.event_count(); ++j) {
            if (unweighted_count == event_count) {
                done = true;
                break;
            }
            double weight = buffer.event<EventWeightRecord>(j).weight();
            if (weight == 0.) {
                continue;
            }
            weight_sum += weight / channel.max_weight;
            ++unweighted_count;
        }
        if (done) {
            break;
        }
    }
    return weight_sum;
}

std::vector<EventGenerator::Status> EventGenerator::channel_status() const {
    std::vector<EventGenerator::Status> status;
    for (auto& channel : _channels) {
        double target_count = channel.integral_fraction * _config.target_count;
        status.push_back(
            {channel.index,
             channel.subprocess_index,
             channel.name,
             channel.cross_section.mean(),
             channel.cross_section.error(),
             channel.cross_section.rel_std_dev(),
             channel.total_sample_count,
             channel.total_sample_count_opt,
             channel.total_sample_count_after_cuts,
             channel.total_sample_count_after_cuts_opt,
             channel.eff_count,
             target_count,
             channel.iterations,
             !channel.needs_optimization,
             channel.eff_count >= target_count}
        );
    }
    return status;
}

std::vector<EventGenerator::Histogram> EventGenerator::histograms() const {
    auto hists = _empty_histograms;
    for (auto& channel : _channels) {
        for (auto [chan_hist, out_hist] : zip(channel.histograms, hists)) {
            for (auto [chan_w, val, err] :
                 zip(chan_hist, out_hist.bin_values, out_hist.bin_errors)) {
                auto [w, w2] = chan_w;
                auto n = channel.total_sample_count_opt;
                val += w / n;
                err += (w2 - w * w / n) / (n * n);
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

std::tuple<Tensor, std::vector<Tensor>> EventGenerator::integrate_and_optimize(
    ChannelState& channel, TensorVec& events, bool run_optim
) {
    auto& weights = events.at(0);
    auto weights_cpu = weights.cpu();
    auto w_view = weights_cpu.view<double, 1>();
    std::size_t sample_count_after_cuts = 0;
    for (std::size_t i = 0; i < w_view.size(); ++i) {
        if (w_view[i] != 0) {
            ++sample_count_after_cuts;
        }
        channel.cross_section.push(w_view[i]);
    }
    channel.total_sample_count += w_view.size();
    channel.total_sample_count_opt += w_view.size();
    channel.total_sample_count_after_cuts += sample_count_after_cuts;
    channel.total_sample_count_after_cuts_opt += sample_count_after_cuts;

    if (channel.observable_histograms) {
        auto hists = channel.observable_histograms->run({weights, events.at(1)});
        for (std::size_t i = 0; i < hists.size() / 2; ++i) {
            Tensor hist_cpu = hists.at(2 * i).cpu();
            Tensor hist2_cpu = hists.at(2 * i + 1).cpu();
            auto hist_view = hist_cpu.view<double, 2>()[0];
            auto hist2_view = hist2_cpu.view<double, 2>()[0];
            if (channel.histograms.size() <= i) {
                channel.histograms.emplace_back(hist_view.size());
            }
            auto& chan_hist = channel.histograms.at(i);
            for (std::size_t j = 0; j < hist_view.size(); ++j) {
                chan_hist.at(j).first += hist_view[j];
                chan_hist.at(j).second += hist2_view[j];
            }
        }
    }

    if (run_optim) {
        if (channel.vegas_optimizer) {
            auto hist = channel.vegas_histogram->run({events.at(6), weights});
            channel.vegas_optimizer->add_data(hist.at(0), hist.at(1));
        }
        if (channel.discrete_optimizer) {
            TensorVec args{events.begin() + 7, events.end()};
            args.push_back(weights);
            auto hist = channel.discrete_histogram->run(args);
            channel.discrete_optimizer->add_data(hist);
        }
        if (channel.job_count == 0) {
            if (channel.vegas_optimizer) {
                channel.vegas_optimizer->optimize();
            }
            if (channel.discrete_optimizer) {
                channel.discrete_optimizer->optimize();
            }
            double rsd = channel.cross_section.rel_std_dev();
            if (rsd < _config.optimization_threshold * channel.best_rsd) {
                channel.iters_without_improvement = 0;
            } else {
                ++channel.iters_without_improvement;
                if (channel.iters_without_improvement >=
                    _config.optimization_patience) {
                    channel.needs_optimization = false;
                }
            }
            channel.best_rsd = std::min(rsd, channel.best_rsd);
            ++channel.iterations;
        }
    }

    double total_mean = 0., total_var = 0.;
    std::size_t total_count = 0, total_count_opt = 0;
    std::size_t total_count_after_cuts = 0, total_count_after_cuts_opt = 0;
    std::size_t total_integ_count = 0;
    std::size_t iterations = 0;
    bool optimized = true;
    for (auto& channel : _channels) {
        total_mean += channel.cross_section.mean();
        total_var += channel.cross_section.variance() / channel.cross_section.count();
        total_count += channel.total_sample_count;
        total_count_opt += channel.total_sample_count_opt;
        total_count_after_cuts += channel.total_sample_count_after_cuts;
        total_count_after_cuts_opt += channel.total_sample_count_after_cuts_opt;
        total_integ_count += channel.cross_section.count();
        iterations = std::max(channel.iterations, iterations);
        if (channel.needs_optimization) {
            optimized = false;
        }
    }
    _status_all.mean = total_mean;
    _status_all.error = std::sqrt(total_var);
    _status_all.rel_std_dev = std::sqrt(total_var * total_integ_count) / total_mean;
    _status_all.count = total_count;
    _status_all.count_opt = total_count_opt;
    _status_all.count_after_cuts = total_count_after_cuts;
    _status_all.count_after_cuts_opt = total_count_after_cuts_opt;
    _status_all.iterations = iterations;
    _status_all.optimized = optimized;
    for (auto& channel : _channels) {
        channel.integral_fraction = channel.cross_section.mean() / total_mean;
    }

    return {weights_cpu, events};
}

void EventGenerator::start_job(
    ChannelState& channel, std::size_t batch_size, std::size_t vegas_job_count
) {
    std::size_t new_job_id = _job_id;
    ++_job_id;
    ++channel.job_count;
    auto& job =
        std::get<0>(_running_jobs.emplace(
                        new_job_id, RunningJob{channel.index, {}, vegas_job_count}
                    ))
            ->second;
    default_thread_pool().submit([new_job_id, batch_size, &job, &channel]() {
        job.events = channel.runtime->run({Tensor({batch_size})});
        return new_job_id;
    });
}

void EventGenerator::start_vegas_jobs(ChannelState& channel) {
    std::size_t vegas_job_count =
        (channel.batch_size + _config.batch_size - 1) / _config.batch_size;
    for (std::size_t i = 0; i < vegas_job_count; ++i) {
        std::size_t batch_size =
            std::min(_config.batch_size, channel.batch_size - i * _config.batch_size);
        start_job(channel, batch_size, vegas_job_count);
    }
    channel.batch_size = std::min(channel.batch_size * 2, _config.max_batch_size);
}

void EventGenerator::clear_channel(ChannelState& channel) {
    channel.eff_count = 0;
    channel.max_weight = 0;
    channel.total_sample_count_opt = 0;
    channel.total_sample_count_after_cuts_opt = 0;
    channel.event_file.clear();
    channel.weight_file.clear();
    channel.cross_section.reset();
    channel.large_weights.clear();
    channel.histograms.clear();
}

void EventGenerator::update_max_weight(ChannelState& channel, Tensor weights) {
    if (channel.eff_count > _config.freeze_max_weight_after) {
        return;
    }

    auto w_view = weights.view<double, 1>();
    double w_min_nonzero = 0.;
    for (std::size_t i = 0; i < w_view.size(); ++i) {
        double w = std::abs(w_view[i]);
        if (w != 0 && (w_min_nonzero == 0 || w < w_min_nonzero)) {
            w_min_nonzero = w;
        }
        if (w > channel.max_weight) {
            channel.large_weights.push_back(w);
        }
    }
    if (channel.max_weight == 0) {
        channel.max_weight = w_min_nonzero;
    }
    std::sort(
        channel.large_weights.begin(), channel.large_weights.end(), std::greater{}
    );

    double w_sum = 0, w_prev = 0;
    double max_truncation = _config.max_overweight_truncation *
        std::min(channel.integral_fraction * _config.target_count,
                 static_cast<double>(_config.freeze_max_weight_after));
    std::size_t count = 0;
    for (auto w : channel.large_weights) {
        if (w < channel.max_weight) {
            break;
        }
        w_sum += w;
        ++count;
        if (w_sum / w - count > max_truncation) {
            if (channel.max_weight < w) {
                channel.eff_count *= channel.max_weight / w_prev;
                channel.max_weight = w_prev;
            }
            break;
        }
        w_prev = w;
    }
    channel.large_weights.erase(
        channel.large_weights.begin() + count, channel.large_weights.end()
    );
}

void EventGenerator::unweight_and_write(
    ChannelState& channel, const std::vector<Tensor>& events
) {
    std::vector<Tensor> unweighter_args(events.begin(), events.begin() + 6);
    unweighter_args.push_back(Tensor(channel.max_weight, _context->device()));
    TensorVec unw_events = _unweighter->run(unweighter_args);
    TensorVec unw_events_cpu;
    for (auto& field : unw_events) {
        unw_events_cpu.push_back(field.cpu());
    }
    auto w_view = unw_events_cpu.at(0).view<double, 1>();
    auto mom_view = unw_events_cpu.at(1).view<double, 3>();
    auto colors_view = unw_events_cpu.at(2).view<me_int_t, 1>();
    auto helicities_view = unw_events_cpu.at(3).view<me_int_t, 1>();
    auto diagrams_view = unw_events_cpu.at(4).view<me_int_t, 1>();
    auto flavors_view = unw_events_cpu.at(5).view<me_int_t, 1>();

    EventBuffer event_buffer(
        w_view.size(),
        channel.event_file.particle_count(),
        DataLayout::of<EventIndicesRecord, ParticleRecord>()
    );
    EventBuffer weight_buffer(
        w_view.size(), 0, DataLayout::of<EventWeightRecord, EmptyParticleRecord>()
    );
    for (std::size_t i = 0; i < w_view.size(); ++i) {
        weight_buffer.event<EventWeightRecord>(i).weight() = w_view[i];
        auto event = event_buffer.event<EventIndicesRecord>(i);
        event.diagram_index() = diagrams_view[i];
        event.color_index() = colors_view[i];
        event.flavor_index() = flavors_view[i];
        event.helicity_index() = helicities_view[i];
        auto event_mom = mom_view[i];
        for (std::size_t j = 0; j < event_mom.size(); ++j) {
            auto particle_mom = event_mom[j];
            auto particle = event_buffer.particle<ParticleRecord>(i, j);
            particle.energy() = particle_mom[0];
            particle.px() = particle_mom[1];
            particle.py() = particle_mom[2];
            particle.pz() = particle_mom[3];
        }
    }
    channel.event_file.write(event_buffer);
    channel.weight_file.write(weight_buffer);

    channel.eff_count += w_view.size();
    double total_eff_count = 0.;
    bool done = true;
    for (auto& channel : _channels) {
        double chan_target = channel.integral_fraction * _config.target_count;
        if (channel.eff_count < chan_target) {
            total_eff_count += channel.eff_count;
            done = false;
        } else {
            total_eff_count += chan_target;
        }
    }
    _status_all.count_unweighted = total_eff_count;
    _status_all.done = done;
}

std::tuple<std::vector<EventGenerator::CombineChannelData>, std::size_t, double>
EventGenerator::init_combine() {
    std::vector<EventGenerator::CombineChannelData> channel_data;
    std::size_t count_sum = 0;
    std::size_t particle_count = 0;
    double weight_sum = 0.;
    for (auto& channel : _channels) {
        particle_count = std::max(particle_count, channel.event_file.particle_count());
        std::size_t count =
            std::round(channel.integral_fraction * _config.target_count);
        count_sum += count;
        channel.event_file.seek(0);
        weight_sum += channel_weight_sum(channel, count);
        channel.weight_file.seek(0);
        channel_data.push_back({
            .cum_count = count_sum,
            .event_buffer = EventBuffer(
                0,
                channel.event_file.particle_count(),
                DataLayout::of<EventIndicesRecord, ParticleRecord>()
            ),
            .weight_buffer = EventBuffer(
                0, 0, DataLayout::of<EventWeightRecord, EmptyParticleRecord>()
            ),
            .buffer_index = 0,
        });
    }
    return {channel_data, particle_count, _status_all.mean * count_sum / weight_sum};
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
                channel.event_file.read(sampled_chan->event_buffer, batch_size);
                channel.weight_file.read(sampled_chan->weight_buffer, batch_size);
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
        event_out.weight() = std::max(1., weight / channel.max_weight) * norm_factor;
        event_out.subprocess_index() = channel.subprocess_index;
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
    std::size_t event_index
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
        event_in.helicity_index()
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
        {"process", _status_all},
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
    if (_config.verbosity != EventGenerator::pretty) {
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
    if (_config.verbosity == EventGenerator::pretty) {
        print_survey_update_pretty(done, done_job_count, total_job_count, iter);
    } else if (_config.verbosity == EventGenerator::log) {
        print_survey_update_log(done, done_job_count, total_job_count, iter);
    }
}

void EventGenerator::print_survey_update_pretty(
    bool done, std::size_t done_job_count, std::size_t total_job_count, std::size_t iter
) {
    std::string int_str = format_with_error(_status_all.mean, _status_all.error);
    std::string count_str = std::format(
        "{} before cuts, {} after",
        format_si_prefix(_status_all.count),
        format_si_prefix(_status_all.count_after_cuts)
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
            format_with_error(_status_all.mean, _status_all.error),
            format_si_prefix(_status_all.count),
            format_si_prefix(_status_all.count_after_cuts),
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
    if (_config.verbosity != EventGenerator::pretty) {
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
    if (_config.verbosity == EventGenerator::pretty) {
        print_gen_update_pretty(done);
    } else if (_config.verbosity == EventGenerator::log) {
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
        format_si_prefix(_status_all.count),
        format_si_prefix(_status_all.count_after_cuts)
    );

    if (!std::isnan(_status_all.error)) {
        double rel_err = _status_all.error / _status_all.mean;
        int_str = format_with_error(_status_all.mean, _status_all.error);
        rel_str = std::format("{:.4f} %", rel_err * 100);
        rsd_str = std::format("{:.3f}", _status_all.rel_std_dev);
        uweff_str = std::format(
            "{:.5f} before cuts, {:.5f} after",
            _status_all.count_unweighted / _status_all.count_opt,
            _status_all.count_unweighted / _status_all.count_after_cuts_opt
        );
    }
    std::string unw_str = std::format(
        "{} / {}",
        format_si_prefix(_status_all.count_unweighted),
        format_si_prefix(_status_all.count_target)
    );
    std::string time_str;
    if (done) {
        time_str = format_run_time("generate");
    } else {
        unw_str = std::format(
            "{:<15} {}",
            unw_str,
            format_progress(_status_all.count_unweighted / _status_all.count_target, 52)
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
                if (!_status_all.done) {
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
            format_si_prefix(_status_all.count_unweighted),
            format_si_prefix(_status_all.count_target),
            format_with_error(_status_all.mean, _status_all.error),
            _status_all.error / _status_all.mean * 100,
            _status_all.rel_std_dev,
            format_si_prefix(_status_all.count),
            format_si_prefix(_status_all.count_after_cuts),
            _status_all.count_unweighted / _status_all.count_opt,
            _status_all.count_unweighted / _status_all.count_after_cuts_opt,
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
    if (_config.verbosity != EventGenerator::pretty) {
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
    if (_config.verbosity == EventGenerator::pretty) {
        print_combine_update_pretty(count);
    } else if (_config.verbosity == EventGenerator::log) {
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

void madevent::to_json(nlohmann::json& j, const EventGenerator::Status& status) {
    j = nlohmann::json{
        {"index", status.index},
        {"subprocess", status.subprocess},
        {"name", status.name},
        {"mean", status.mean},
        {"error", status.error},
        {"rel_std_dev", status.rel_std_dev},
        {"count", status.count},
        {"count_opt", status.count_opt},
        {"count_after_cuts", status.count_after_cuts},
        {"count_after_cuts_opt", status.count_after_cuts_opt},
        {"count_unweighted", status.count_unweighted},
        {"count_target", status.count_target},
        {"iterations", status.iterations},
        {"optimized", status.optimized},
        {"done", status.done},
    };
}

void madevent::to_json(
    nlohmann::json& j, const EventGenerator::TimingData& timing_data
) {
    j = nlohmann::json{
        {"wall_time_sec", timing_data.wall_time_sec},
        {"cpu_time_sec", timing_data.cpu_time_sec},
    };
}

void madevent::to_json(nlohmann::json& j, const EventGenerator::Histogram& hist) {
    j = nlohmann::json{
        {"name", hist.name},
        {"min", hist.min},
        {"max", hist.max},
        {"bin_values", hist.bin_values},
        {"bin_errors", hist.bin_errors},
    };
}

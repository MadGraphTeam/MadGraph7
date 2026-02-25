#include "madspace/runtime/channel_generator.h"

using namespace madspace;

ChannelEventGenerator::ChannelEventGenerator(
    const std::vector<ContextPtr>& contexts,
    const Integrand& integrand,
    const std::string& event_file,
    const std::string& weight_file,
    const GeneratorConfig& config,
    std::size_t index,
    std::size_t subprocess_index,
    const std::string& name,
    const std::optional<ObservableHistograms>& histograms,
    double integral_estimate
) :
    _contexts(contexts),
    _status{
        .index = index,
        .subprocess = subprocess_index,
        .name = name,
        .mean = integral_estimate,
        .error = 0.,
        .rel_std_dev = 0.,
        .count = 0,
        .count_opt = 0,
        .count_after_cuts = 0,
        .count_after_cuts_opt = 0,
        .count_unweighted = 0.,
        .count_target = 0.,
        .optimized = false,
        .done = false
    },
    _config(config),
    _event_file(
        event_file,
        DataLayout::of<EventIndicesRecord, ParticleRecord>(),
        integrand.particle_count(),
        EventFile::create,
        true
    ),
    _weight_file(
        weight_file,
        DataLayout::of<EventWeightRecord, EmptyParticleRecord>(),
        0,
        EventFile::create,
        true
    ),
    _batch_size(config.start_batch_size) {
    if (integrand.flags() != integrand_flags) {
        throw std::invalid_argument(
            "Integrand flags must be sample | return_momenta | return_random | "
            "return_discrete"
        );
    }
    for (auto& context : contexts) {
        _runtimes.push_back(
            {.integrand = build_runtime(integrand.function(), context, false),
             .unweighter = build_runtime(
                 Unweighter({integrand.return_types().begin(),
                             integrand.return_types().begin() + 6})
                     .function(),
                 context,
                 false
             )}
        );
    }
    if (const auto& grid_name = integrand.vegas_grid_name(); grid_name) {
        _vegas_optimizer =
            VegasGridOptimizer(contexts, grid_name.value(), config.vegas_damping);
        VegasHistogram hist(integrand.vegas_dimension(), integrand.vegas_bin_count());
        for (auto [context, runtimes] : zip(contexts, _runtimes)) {
            runtimes.vegas_histogram = build_runtime(hist.function(), context, false);
        }
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
    std::visit(add_names, integrand.discrete_before());
    std::visit(add_names, integrand.discrete_after());
    std::optional<DiscreteOptimizer> discrete_optimizer;
    RuntimePtr discrete_histogram = nullptr;
    if (prob_names.size() > 0) {
        discrete_optimizer = DiscreteOptimizer(contexts, prob_names);
        DiscreteHistogram hist(option_counts);
        for (auto [context, runtimes] : zip(contexts, _runtimes)) {
            runtimes.discrete_histogram =
                build_runtime(hist.function(), context, false);
        }
    }
    RuntimePtr obs_histograms = nullptr;
    if (histograms) {
        for (auto [context, runtimes] : zip(contexts, _runtimes)) {
            runtimes.observable_histograms =
                build_runtime(histograms.value().function(), context, false);
        }
        for (auto& item : histograms.value().observables()) {
            _histograms.push_back({
                .name = item.observable.name(),
                .min = item.min,
                .max = item.max,
                .bin_values = std::vector<double>(item.bin_count + 2),
                .bin_errors = std::vector<double>(item.bin_count + 2),
            });
        }
    }
}

void ChannelEventGenerator::unweight_file(std::mt19937 rand_gen) {
    std::size_t buf_size = 1000000;
    std::uniform_real_distribution<double> rand_dist;
    EventBuffer buffer(0, 0, DataLayout::of<EventWeightRecord, EmptyParticleRecord>());
    std::size_t accept_count = _unweighted_count;
    for (std::size_t i = _unweighted_count; i < _weight_file.event_count();
         i += buf_size) {
        _weight_file.seek(i);
        _weight_file.read(buffer, buf_size);
        for (std::size_t j = 0; j < buffer.event_count(); ++j) {
            auto weight = buffer.event<EventWeightRecord>(j).weight();
            if (weight / _max_weight < rand_dist(rand_gen)) {
                weight = 0;
            } else {
                weight = std::max(weight.value(), _max_weight);
                ++accept_count;
            }
        }
        _weight_file.seek(i);
        _weight_file.write(buffer);
    }
    _status.count_unweighted = accept_count;
}

void ChannelEventGenerator::integrate_and_optimize(
    const GeneratorBatchJob& job, bool run_optim
) {
    auto w_view = job.weights.view<double, 1>();
    std::size_t sample_count_after_cuts = 0;
    for (std::size_t i = 0; i < w_view.size(); ++i) {
        if (w_view[i] != 0) {
            ++sample_count_after_cuts;
        }
        _cross_section.push(w_view[i]);
    }
    _status.count += w_view.size();
    _status.count_opt += w_view.size();
    _status.count_after_cuts += sample_count_after_cuts;
    _status.count_after_cuts_opt += sample_count_after_cuts;

    if (job.hists.size() > 0) {
        for (std::size_t i = 0; i < job.hists.size() / 2; ++i) {
            auto hist_view = job.hists.at(2 * i).view<double, 2>()[0];
            auto hist2_view = job.hists.at(2 * i + 1).view<double, 2>()[0];
            auto& chan_hist = _histograms.at(i);
            for (std::size_t j = 0; j < hist_view.size(); ++j) {
                // note: we don't assign the actuals means and errors here. There are
                // still some normalization steps necessary that are performed later
                chan_hist.bin_values.at(j) += hist_view[j];
                chan_hist.bin_errors.at(j) += hist2_view[j];
            }
        }
    }

    if (job.vegas_hist.size() > 0) {
        _vegas_optimizer->add_data(job.vegas_hist.at(0), job.vegas_hist.at(1));
    }
    if (job.discrete_hist.size() > 0) {
        _discrete_optimizer->add_data(job.discrete_hist);
    }
    if (run_optim) {
        if (_vegas_optimizer) {
            _vegas_optimizer->optimize();
        }
        if (_discrete_optimizer) {
            _discrete_optimizer->optimize();
        }
        double rsd = _cross_section.rel_std_dev();
        if (rsd < _config.optimization_threshold * _best_rsd) {
            _iters_without_improvement = 0;
        } else {
            ++_iters_without_improvement;
            if (_iters_without_improvement >= _config.optimization_patience) {
                _needs_optimization = false;
            }
        }
        _best_rsd = std::min(rsd, _best_rsd);
        ++_status.iterations;
    }
}

double ChannelEventGenerator::channel_weight_sum(std::size_t event_count) {
    std::size_t buf_size = 1000000;
    EventBuffer buffer(0, 0, DataLayout::of<EventWeightRecord, EmptyParticleRecord>());
    double weight_sum = 0;
    _weight_file.seek(0);
    std::size_t unweighted_count = 0;
    for (std::size_t i = 0; i < _weight_file.event_count(); i += buf_size) {
        _weight_file.read(buffer, buf_size);
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
            weight_sum += weight / _max_weight;
            _unweighted_count = 0;
            ++unweighted_count;
        }
        if (done) {
            break;
        }
    }
    return weight_sum;
}

void ChannelEventGenerator::start_job(GeneratorBatchJob& job) {
    _contexts.at(job.context_index)->thread_pool().submit([this, &job]() {
        auto& runtimes = _runtimes.at(job.context_index);
        auto& context = _contexts.at(job.context_index);
        TensorVec events = runtimes.integrand->run({Tensor({job.batch_size})});
        job.weights = events.at(0).cpu();
        if (job.unweight) {
            std::vector<Tensor> unweighter_args(events.begin(), events.begin() + 6);
            unweighter_args.push_back(Tensor(_max_weight, context->device()));
            TensorVec unw_events = runtimes.unweighter->run(unweighter_args);
            for (auto& item : unw_events) {
                job.unweighted_events.push_back(item.cpu());
            }
        }
        if (runtimes.observable_histograms) {
            auto hists =
                runtimes.observable_histograms->run({events.at(0), events.at(1)});
            for (auto& item : hists) {
                job.hists.push_back(item.cpu());
            }
        }
        if (job.vegas_job_count > 0) {
            if (_vegas_optimizer) {
                auto hist = runtimes.vegas_histogram->run({events.at(6), events.at(0)});
                for (auto& item : hist) {
                    job.vegas_hist.push_back(item.cpu());
                }
            }
            if (_discrete_optimizer) {
                TensorVec args{events.begin() + 7, events.end()};
                args.push_back(events.at(0));
                auto hist = runtimes.discrete_histogram->run(args);
                for (auto& item : hist) {
                    job.discrete_hist.push_back(item.cpu());
                }
            }
        }
        return job.job_id;
    });
}

void ChannelEventGenerator::build_vegas_jobs(
    std::vector<GeneratorBatchJob>& ready_jobs, bool unweight
) {
    std::size_t vegas_job_count =
        (_batch_size + _config.batch_size - 1) / _config.batch_size;
    for (std::size_t i = 0; i < vegas_job_count; ++i) {
        std::size_t batch_size =
            std::min(_config.batch_size, _batch_size - i * _config.batch_size);
        ready_jobs.push_back(
            {.channel_index = i,
             .unweight = true,
             .batch_size = batch_size,
             .vegas_job_count = vegas_job_count}
        );
    }
    _batch_size = std::min(_batch_size * 2, _config.max_batch_size);
}

void ChannelEventGenerator::clear_events() {
    _status.count_unweighted = 0;
    _max_weight = 0;
    _unweighted_count = 0;
    _status.count_opt = 0;
    _status.count_after_cuts_opt = 0;
    _event_file.clear();
    _weight_file.clear();
    _cross_section.reset();
    _large_weights.clear();
    for (auto& hist : _histograms) {
        std::fill(hist.bin_values.begin(), hist.bin_values.end(), 0.);
        std::fill(hist.bin_errors.begin(), hist.bin_errors.end(), 0.);
    }
}

void ChannelEventGenerator::update_max_weight(Tensor weights) {
    if (_status.count_unweighted > _config.freeze_max_weight_after) {
        return;
    }

    auto w_view = weights.view<double, 1>();
    double w_min_nonzero = 0.;
    for (std::size_t i = 0; i < w_view.size(); ++i) {
        double w = std::abs(w_view[i]);
        if (w != 0 && (w_min_nonzero == 0 || w < w_min_nonzero)) {
            w_min_nonzero = w;
        }
        if (w > _max_weight) {
            _large_weights.push_back(w);
        }
    }
    if (_max_weight == 0) {
        _max_weight = w_min_nonzero;
    }
    std::sort(_large_weights.begin(), _large_weights.end(), std::greater{});

    double w_sum = 0, w_prev = 0;
    double max_truncation = _config.max_overweight_truncation *
        std::min(_integral_fraction * _config.target_count,
                 static_cast<double>(_config.freeze_max_weight_after));
    std::size_t count = 0;
    for (auto w : _large_weights) {
        if (w < _max_weight) {
            break;
        }
        w_sum += w;
        ++count;
        if (w_sum / w - count > max_truncation) {
            if (_max_weight < w) {
                _status.count_unweighted *= _max_weight / w_prev;
                _max_weight = w_prev;
                _unweighted_count = 0;
            }
            break;
        }
        w_prev = w;
    }
    _large_weights.erase(_large_weights.begin() + count, _large_weights.end());
}

void ChannelEventGenerator::unweight_and_write(
    const std::vector<Tensor>& unweighted_events
) {
    auto w_view = unweighted_events.at(0).view<double, 1>();
    auto mom_view = unweighted_events.at(1).view<double, 3>();
    auto colors_view = unweighted_events.at(2).view<me_int_t, 1>();
    auto helicities_view = unweighted_events.at(3).view<me_int_t, 1>();
    auto diagrams_view = unweighted_events.at(4).view<me_int_t, 1>();
    auto flavors_view = unweighted_events.at(5).view<me_int_t, 1>();

    EventBuffer event_buffer(
        w_view.size(),
        _event_file.particle_count(),
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
    _event_file.write(event_buffer);
    _weight_file.write(weight_buffer);
    _status.count_unweighted += w_view.size();
}

#include "madspace/driver/madnis_training.hpp"

#include "madspace/driver/random.hpp"
#include "madspace/phasespace/batch_sampler.hpp"

using namespace madspace;

MadnisTraining::MadnisTraining(
    ContextPtr generator_context,
    ContextPtr optimizer_context,
    const Config& config,
    const std::vector<std::shared_ptr<Integrand>>& integrands,
    const std::optional<ChannelWeightNetwork>& cwnet,
    std::optional<std::uint64_t> seed,
    std::size_t channel_index_offset,
    bool copy_optimizer_globals
) :
    _generator_context(generator_context),
    _optimizer_context(optimizer_context),
    _config(config),
    _channels(integrands.size()),
    _cwnet(cwnet),
    _seed(seed),
    _channel_index_offset(channel_index_offset),
    _copy_optimizer_globals(copy_optimizer_globals) {
    for (std::size_t index = 0;
         auto [integrand, channel] : zip(integrands, _channels)) {
        channel.index = index;
        channel.integrand = integrand;
        channel.integrand_prob = std::make_shared<IntegrandProbability>(*integrand);
        if (_arg_permutation.size() == 0) {
            auto& integ_args = channel.integrand->return_types().index_map();
            _arg_permutation.push_back(integ_args.at("weight"));
            _arg_permutation.push_back(integ_args.at("adaptive_prob"));
            if (cwnet) {
                _arg_permutation.push_back(integ_args.at("cwnet_input"));
                if (integ_args.contains("channel_weight_values")) {
                    _arg_permutation.push_back(integ_args.at("channel_weight_values"));
                    _arg_permutation.push_back(integ_args.at("channel_weight_indices"));
                } else {
                    _arg_permutation.push_back(integ_args.at("channel_weights"));
                }
                _arg_permutation.push_back(integ_args.at("channel_index"));
            }
            for (auto key : channel.integrand_prob->arg_types().keys()) {
                _arg_permutation.push_back(integ_args.at(key));
            }
        }
        auto& active_flavors = integrand->active_flavors();
        for (std::size_t flav_index : active_flavors) {
            if (_active_flavors_count.size() <= flav_index) {
                _active_flavors_count.resize(flav_index + 1);
            }
            ++_active_flavors_count.at(flav_index);
        }
        ++index;
    }
    build_runtimes_and_optimizer();
}

MadnisTraining::~MadnisTraining() {
    // the jobs still running write into this object
    if (_jobs_in_flight > 0) {
        _results.discard(_jobs_in_flight);
    }
}

void MadnisTraining::train_step(std::size_t batch_index) {
    _abort_check();
    _batch_index = batch_index;
    std::vector<std::size_t> channel_sizes = compute_channel_sizes();
    // delta-sigma modulation: accumulate the target fraction and spend one
    // quantum per buffered step, which spreads them out evenly
    _buffered_step_accumulator += buffered_step_target();
    bool try_buffered = _buffered_step_accumulator >= _buffered_step_scale;
    if (try_buffered) {
        _buffered_step_accumulator -= _buffered_step_scale;
    }
    TensorVec training_batch;
    bool used_buffered = false;
    while (true) {
        maybe_start_generator_jobs(channel_sizes, !try_buffered);
        if (try_buffered) {
            if (check_buffered_training_batch(channel_sizes)) {
                training_batch = build_buffered_training_batch(channel_sizes);
                used_buffered = true;
                break;
            }
            try_buffered = false;
        } else if (check_online_training_batch(channel_sizes)) {
            training_batch = build_online_training_batch(channel_sizes);
            break;
        }
        process_job_results({wait_for_job()});
    }
    double learning_rate = _optimizer->learning_rate();
    auto step_start = std::chrono::steady_clock::now();
    TensorVec results = _optimizer->step(training_batch);
    _optimizer_ns.fetch_add(
        std::chrono::nanoseconds(std::chrono::steady_clock::now() - step_start).count(),
        std::memory_order_relaxed
    );
    update_history(results, channel_sizes, learning_rate, used_buffered);
    if ((batch_index + 1) % _config.log_interval == 0) {
        _status_batches.push_back(batch_index + 1);
        _status_losses.push_back(average_loss());
        _status_channel_counts.push_back(active_channel_count());
        _status_learning_rates.push_back(average_learning_rate());
        _status_buffered_fractions.push_back(buffered_fraction());
        _status_generated_events.push_back(_generated_event_count);
        _status_buffer_sizes.push_back(buffer_event_count());
    }
    if (_channels.size() > 0 && _cwnet &&
        (batch_index + 1) % _config.channel_dropping_interval == 0) {
        process_all_jobs();
        drop_channels(batch_index + 1);
    }
    if (batch_index ==
        static_cast<std::size_t>(
            (1 - _config.fixed_cwnet_fraction) * _config.batches
        )) {
        process_all_jobs();
        freeze_cwnet();
    }
    if (batch_index + 1 == _config.batches) {
        process_all_jobs();
    }
}

std::size_t MadnisTraining::wait_for_job() {
    auto result = _results.wait(_abort_check);
    --_jobs_in_flight;
    if (result.exception) {
        std::rethrow_exception(result.exception);
    }
    return result.id;
}

void MadnisTraining::process_all_jobs() {
    while (_jobs_in_flight > 0) {
        process_job_results({wait_for_job()});
    }
}

std::vector<std::size_t> MadnisTraining::active_channels() const {
    std::vector<std::size_t> indices(_channels.size());
    for (auto [channel, index] : zip(_channels, indices)) {
        index = channel.index;
    }
    return indices;
}

double MadnisTraining::average_loss() const {
    double loss_sum = 0;
    std::size_t loss_count = 0, nan_count = 0;
    for (double loss : _loss_history) {
        if (!std::isnan(loss)) {
            loss_sum += loss;
            ++loss_count;
        } else {
            ++nan_count;
        }
    }
    return loss_sum / loss_count;
}

double MadnisTraining::average_learning_rate() const {
    if (_lr_history.size() == 0) {
        return 0.;
    }
    double lr_sum = 0;
    for (double lr : _lr_history) {
        lr_sum += lr;
    }
    return lr_sum / _lr_history.size();
}

double MadnisTraining::buffered_fraction() const {
    if (_buffered_history.size() == 0) {
        return 0.;
    }
    std::size_t buffered_count = 0;
    for (bool buffered : _buffered_history) {
        buffered_count += buffered;
    }
    return static_cast<double>(buffered_count) / _buffered_history.size();
}

// Target fraction of buffered steps in units of _buffered_step_scale, ramped up
// linearly between minimum_buffer_size and buffer_capacity of the least filled buffer
std::size_t MadnisTraining::buffered_step_target() const {
    if (_config.buffer_capacity == 0 || _config.buffered_steps_fraction <= 0.) {
        return 0;
    }
    std::size_t buffer_size = _config.buffer_capacity;
    for (auto& channel : _channels) {
        buffer_size = std::min(buffer_size, channel.buffer.size);
    }
    if (buffer_size <= _config.minimum_buffer_size) {
        return 0;
    }
    std::size_t range = _config.buffer_capacity > _config.minimum_buffer_size
        ? _config.buffer_capacity - _config.minimum_buffer_size
        : 1;
    std::size_t filled = std::min(buffer_size - _config.minimum_buffer_size, range);
    std::size_t max_target = static_cast<std::size_t>(
        _buffered_step_scale * std::min(_config.buffered_steps_fraction, 1.)
    );
    return max_target * filled / range;
}

std::size_t MadnisTraining::buffer_event_count() const {
    std::size_t count = 0;
    for (auto& channel : _channels) {
        count += channel.buffer.size;
    }
    return count;
}

void MadnisTraining::build_runtimes_and_optimizer() {
    std::vector<std::shared_ptr<FunctionGenerator>> functions;
    functions.reserve(_channels.size());
    _multi_channel_generator.reset();
    for (auto& channel : _channels) {
        functions.push_back(channel.integrand_prob);
        channel.generator_runtime.reset();
    }
    Function madnis_func =
        MadnisLoss(
            functions,
            _cwnet,
            _config.softclip_threshold,
            _config.compressed_channel_weight_count
        )
            .function();
    if (!_optimizer) {
        std::vector<std::string> global_names;
        for (auto& [name, value] : madnis_func.globals()) {
            global_names.push_back(name);
        }
        if (_cwnet) {
            global_names.push_back(_cwnet->mask_name());
        }
        if (_copy_optimizer_globals) {
            _optimizer_context->copy_globals_from(*_generator_context, global_names);
        }
    }
    if (_optimizer) {
        _optimizer->replace_function(madnis_func);
    } else {
        _optimizer.emplace(
            madnis_func,
            _optimizer_context,
            _config.learning_rate,
            _config.lr_schedule,
            _config.batches,
            _config.adam_beta1,
            _config.adam_beta2,
            _config.adam_eps,
            _config.grad_clip_threshold,
            _config.adam_weight_decay
        );
        // before freeze_cwnet takes the channel weight network out of them
        _trained_globals = _optimizer->param_names();
        if (_cwnet) {
            _trained_globals.push_back(_cwnet->mask_name());
        }
    }
    _generator_params =
        _generator_context->reallocate_globals_contiguously(_optimizer->param_names());
    if (_generator_context->device()->device_type() == DeviceType::cpu) {
        for (auto& channel : _channels) {
            channel.generator_runtime =
                build_runtime(channel.integrand->function(), _generator_context, false);
            if (_config.buffer_capacity > 0) {
                channel.unweighter_runtime = build_runtime(
                    BufferUnweighter(
                        channel.integrand->return_types(),
                        _config.buffer_unweighting_quantile
                    )
                        .function(),
                    _generator_context,
                    false
                );
            }
        }
    } else {
        std::vector<std::shared_ptr<Integrand>> integrands;
        integrands.reserve(_channels.size());
        for (auto& channel : _channels) {
            integrands.push_back(channel.integrand);
        }
        _multi_channel_generator = build_runtime(
            MultiChannelIntegrand(integrands, true).function(),
            _generator_context,
            false
        );
        if (_config.buffer_capacity > 0) {
            std::vector<std::shared_ptr<FunctionGenerator>> buf_unw_funcs;
            for (auto& channel : _channels) {
                buf_unw_funcs.push_back(
                    std::make_shared<BufferUnweighter>(
                        channel.integrand->return_types(),
                        _config.buffer_unweighting_quantile
                    )
                );
            }
            _multi_channel_unweighter = build_runtime(
                MultiChannelFunction(buf_unw_funcs, true).function(),
                _generator_context,
                false
            );
        }
    }
    if (_config.buffer_capacity > 0) {
        std::vector<NamedVector<Type>> types;
        for (auto& channel : _channels) {
            auto& chan_types = types.emplace_back();
            auto& ret_types_named = channel.integrand->return_types();
            auto ret_keys = ret_types_named.keys();
            auto& ret_types = ret_types_named.values();
            bool buffer_needs_init = channel.buffer.tensors.size() == 0;
            for (std::size_t index : _arg_permutation) {
                auto& type = ret_types.at(index);
                chan_types.push_back(ret_keys.at(index), type);
                if (buffer_needs_init) {
                    Sizes full_shape(type.shape.size() + 1);
                    full_shape[0] = _config.buffer_capacity;
                    std::copy(
                        type.shape.begin(), type.shape.end(), full_shape.begin() + 1
                    );
                    channel.buffer.tensors.emplace_back(
                        type.dtype, full_shape, _optimizer_context->device()
                    );
                }
            }
        }
        _multi_channel_sampler =
            build_runtime(BatchSampler(types).function(), _optimizer_context, false);
    }
}

std::vector<std::size_t> MadnisTraining::compute_channel_sizes() {
    std::size_t chan_count = _channels.size();
    std::size_t batch_size =
        _config.batch_size_per_channel * chan_count + _config.batch_size_offset;
    std::vector<std::size_t> sizes;
    sizes.reserve(chan_count);
    if (_channels.at(0).integration_history.size() <
        _config.integration_history_length) {
        sizes.assign(
            chan_count, std::ceil(static_cast<double>(batch_size) / chan_count)
        );
        return sizes;
    }

    std::vector<double> fractions;
    fractions.reserve(chan_count);
    double stddev_sum = 0.;
    for (auto& channel : _channels) {
        std::size_t total_count = 0;
        double total_var = 0.0;
        for (auto [count, abs_mean, var] : channel.integration_history) {
            if (!std::isnan(var)) {
                total_count += count;
                total_var += count * var;
            }
        }
        double stddev = std::sqrt(total_var / total_count);
        stddev_sum += stddev;
        fractions.push_back(stddev);
    }

    double frac_sum = 0.0;
    double uniform_per_chan = _config.uniform_channel_ratio / chan_count;
    for (double& fraction : fractions) {
        fraction = std::max(fraction / stddev_sum - uniform_per_chan, 0.0);
        frac_sum += fraction;
    }

    double weighted_part = 1.0 - _config.uniform_channel_ratio;
    for (double& fraction : fractions) {
        sizes.push_back(
            std::ceil(
                batch_size * (uniform_per_chan + weighted_part * fraction / frac_sum)
            )
        );
    }
    return sizes;
}

void MadnisTraining::start_generator_jobs(const std::vector<std::size_t>& counts) {
    if (_running_jobs.size() > 0) {
        return;
    }
    bool is_gpu = _generator_context->device()->device_type() != DeviceType::cpu;
    // flush buffer samples staged since the last round (see process_job_results)
    for (auto& channel : _channels) {
        for (auto& pending : channel.pending_buffer_samples) {
            buffer_store(channel, pending);
        }
        channel.pending_buffer_samples.clear();
    }
    _generator_params.copy_from(_optimizer->parameters());
    std::size_t chan_count = counts.size();
    std::size_t batch_size = is_gpu
        ? _config.gpu_generator_batch_granularity
        : _config.cpu_generator_batch_size;
    std::vector<std::size_t> missing_batch_counts(chan_count);
    std::vector<std::size_t> target_batch_counts(chan_count);
    for (auto [channel, count, missing_batch_count, target_batch_count] :
         zip(_channels, counts, missing_batch_counts, target_batch_counts)) {
        std::size_t target_count = count * _config.generator_target_size_factor;
        missing_batch_count = count > channel.sample_count
            ? (count - channel.sample_count + batch_size - 1) / batch_size
            : 0;
        target_batch_count = target_count > channel.sample_count
            ? (target_count - channel.sample_count + batch_size - 1) / batch_size
            : 0;
    }
    std::size_t gpu_subbatches =
        (_config.gpu_generator_batch_size + _config.gpu_generator_batch_granularity -
         1) /
        _config.gpu_generator_batch_granularity;
    // dispatch exactly what this round needs, uncapped by thread count, so
    // round contents don't depend on thread count
    std::size_t available_jobs = 0;
    for (auto count : missing_batch_counts) {
        available_jobs += count;
    }
    for (auto count : target_batch_counts) {
        available_jobs += count;
    }
    std::vector<std::size_t> channel_sizes;
    if (is_gpu) {
        channel_sizes.resize(chan_count, 0);
    }

    for (std::size_t chan_index = 0;
         auto [missing_batch_count, target_batch_count] :
         zip(missing_batch_counts, target_batch_counts)) {
        while (missing_batch_count > 0 && available_jobs > 0) {
            --available_jobs;
            if (is_gpu) {
                channel_sizes.at(chan_index) += batch_size;
                if (available_jobs % gpu_subbatches == 0) {
                    start_multi_job(channel_sizes);
                    channel_sizes.assign(chan_count, 0);
                }
            } else {
                start_single_job(chan_index, batch_size);
            }
            --missing_batch_count;
            --target_batch_count;
        }
        ++chan_index;
        if (available_jobs == 0) {
            break;
        }
    }
    std::vector<std::size_t> chan_indices(counts.size());
    std::iota(chan_indices.begin(), chan_indices.end(), 0);
    std::sort(chan_indices.begin(), chan_indices.end(), [&](auto i, auto j) {
        return target_batch_counts.at(i) > target_batch_counts.at(j);
    });
    for (std::size_t index = 0; available_jobs > 0;) {
        std::size_t chan_index = chan_indices.at(index);
        std::size_t& count = target_batch_counts.at(chan_index);
        if (count == 0) {
            break;
        }
        --available_jobs;
        if (is_gpu) {
            channel_sizes.at(chan_index) += batch_size;
            if (available_jobs % gpu_subbatches == 0) {
                start_multi_job(channel_sizes);
                channel_sizes.assign(chan_count, 0);
            }
        } else {
            start_single_job(chan_index, batch_size);
        }
        --count;
        if (chan_index == chan_count - 1) {
            index = 0;
        } else {
            std::size_t next_count = target_batch_counts.at(chan_indices.at(index + 1));
            if (count < next_count) {
                ++index;
            }
        }
    }
    if (available_jobs > 0 && is_gpu) {
        start_multi_job(channel_sizes);
    }
}

// Requires an online attempt and the online cache to be exhausted or about to
// be, both deterministic conditions, so round contents don't depend on thread
// scheduling.
void MadnisTraining::maybe_start_generator_jobs(
    const std::vector<std::size_t>& counts, bool is_online_attempt
) {
    if (_running_jobs.size() > 0) {
        return;
    }
    if (!is_online_attempt) {
        return;
    }
    bool depletion_imminent = false;
    for (auto [channel, count] : zip(_channels, counts)) {
        if (count >= channel.sample_count) {
            depletion_imminent = true;
            break;
        }
    }
    if (!depletion_imminent) {
        return;
    }
    start_generator_jobs(counts);
}

TensorVec MadnisTraining::permute_tensors(const TensorVec& tensors) const {
    TensorVec ret;
    ret.reserve(tensors.size());
    for (std::size_t index : _arg_permutation) {
        ret.push_back(tensors.at(index));
    }
    return ret;
}

// Seed depends only on channel index and channel-local sequence, not on which
// worker thread runs the job.
void MadnisTraining::start_single_job(
    std::size_t channel_index, std::size_t batch_size
) {
    std::size_t job_id = _job_id;
    ++_job_id;
    auto& channel = _channels.at(channel_index);
    std::size_t channel_seq = channel.next_dispatch_seq++;
    std::size_t global_channel_index = _channel_index_offset + channel_index;
    auto& job = std::get<0>(_running_jobs.emplace(job_id, SampleJob{}))->second;
    job.dispatch_seq = channel_seq;
    // captured by value: reading _batch_index inside the job would make the
    // buffer content depend on thread scheduling
    bool store_buffer = _batch_index >= _config.buffer_skip_batches;
    ++_jobs_in_flight;
    _results.submit(
        _generator_context->thread_pool(),
        job_id,
        [this, channel_index, global_channel_index, batch_size, store_buffer, &job]() {
            auto& channel = _channels.at(channel_index);
            if (_seed) {
                channel.generator_runtime->set_seed(DerivedSeed(
                    _seed,
                    DerivedSeed::madnis_generate,
                    job.dispatch_seq,
                    global_channel_index
                ));
            }
            auto samples = channel.generator_runtime->run({Tensor({batch_size})});
            job.samples.tensors = permute_tensors(samples);
            job.samples.size = samples.at(0).size(0);
            job.samples.channel_index = channel_index;
            if (channel.unweighter_runtime && store_buffer) {
                if (_seed) {
                    channel.unweighter_runtime->set_seed(DerivedSeed(
                        _seed,
                        DerivedSeed::madnis_unweight,
                        job.dispatch_seq,
                        global_channel_index
                    ));
                }
                auto unw_samples = channel.unweighter_runtime->run(samples);
                job.unweighted_samples.tensors = permute_tensors(unw_samples);
                job.unweighted_samples.size = unw_samples.at(0).size(0);
                job.unweighted_samples.channel_index = channel_index;
            }
        }
    );
}

// Dispatch sequence is global (not per-channel), since one job spans all channels.
void MadnisTraining::start_multi_job(const std::vector<std::size_t> batch_sizes) {
    std::size_t job_id = _job_id;
    ++_job_id;
    std::size_t dispatch_seq = _multi_job_next_dispatch_seq++;
    auto& job = std::get<0>(_running_jobs.emplace(job_id, SampleJob{}))->second;
    job.dispatch_seq = dispatch_seq;
    // see start_single_job
    bool store_buffer = _batch_index >= _config.buffer_skip_batches;
    ++_jobs_in_flight;
    _results.submit(
        _generator_context->thread_pool(),
        job_id,
        [this, batch_sizes, store_buffer, &job]() {
            if (_seed) {
                _multi_channel_generator->set_seed(DerivedSeed(
                    _seed,
                    DerivedSeed::madnis_generate,
                    job.dispatch_seq,
                    _channel_index_offset
                ));
            }
            auto samples = _multi_channel_generator->run({Tensor(batch_sizes)});
            job.samples.tensors = permute_tensors(samples);
            job.samples.channel_sizes = samples.back().batch_sizes();
            if (_multi_channel_unweighter && store_buffer) {
                if (_seed) {
                    _multi_channel_unweighter->set_seed(DerivedSeed(
                        _seed,
                        DerivedSeed::madnis_unweight,
                        job.dispatch_seq,
                        _channel_index_offset
                    ));
                }
                auto unw_samples = _multi_channel_unweighter->run(samples);
                job.unweighted_samples.tensors = permute_tensors(unw_samples);
                job.unweighted_samples.channel_sizes = unw_samples.back().batch_sizes();
            }
        }
    );
}

bool MadnisTraining::check_online_training_batch(
    const std::vector<std::size_t>& channel_sizes
) {
    for (auto [channel, count] : zip(_channels, channel_sizes)) {
        if (count > channel.sample_count) {
            return false;
        }
    }
    return true;
}

bool MadnisTraining::check_buffered_training_batch(
    const std::vector<std::size_t>& channel_sizes
) {
    for (auto [channel, count] : zip(_channels, channel_sizes)) {
        if (count > channel.buffer.size ||
            channel.buffer.size < _config.minimum_buffer_size) {
            return false;
        }
    }
    return true;
}

TensorVec
MadnisTraining::build_online_training_batch(const std::vector<size_t>& counts) {
    TensorVec training_batch;
    for (auto [channel, count] : zip(_channels, counts)) {
        auto& first_batch = channel.sample_batches.at(0);
        std::size_t consumed_batches = 0;
        if (first_batch.size - first_batch.consumed_count >= count) {
            for (auto& tensor : first_batch.tensors) {
                training_batch.push_back(tensor.slice(
                    0, first_batch.consumed_count, first_batch.consumed_count + count
                ));
            }
            if (first_batch.size - first_batch.consumed_count == count) {
                consumed_batches = 1;
            } else {
                first_batch.consumed_count += count;
            }
            channel.sample_count -= count;
        } else {
            for (auto& tensor : first_batch.tensors) {
                Sizes shape = tensor.shape();
                shape[0] = count;
                training_batch.emplace_back(tensor.dtype(), shape, tensor.device());
            }
            std::size_t remaining_count = count;
            for (auto& batch : channel.sample_batches) {
                std::size_t batch_size = batch.size - batch.consumed_count;
                std::size_t offset = count - remaining_count;
                if (batch_size >= remaining_count) {
                    for (auto [tensor_out, tensor_in] :
                         zip(std::span(
                                 training_batch.end() - batch.tensors.size(),
                                 training_batch.end()
                             ),
                             batch.tensors)) {
                        tensor_out.slice(0, offset, offset + remaining_count)
                            .copy_from(tensor_in.slice(
                                0,
                                batch.consumed_count,
                                batch.consumed_count + remaining_count
                            ));
                    }
                    batch.consumed_count += remaining_count;
                    channel.sample_count -= remaining_count;
                    break;
                } else {
                    for (auto [tensor_out, tensor_in] :
                         zip(std::span(
                                 training_batch.end() - batch.tensors.size(),
                                 training_batch.end()
                             ),
                             batch.tensors)) {
                        tensor_out.slice(0, offset, offset + batch_size)
                            .copy_from(
                                tensor_in.slice(0, batch.consumed_count, batch.size)
                            );
                    }
                    remaining_count -= batch_size;
                    channel.sample_count -= batch_size;
                    ++consumed_batches;
                }
            }
        }
        channel.sample_batches.erase(
            channel.sample_batches.begin(),
            channel.sample_batches.begin() + consumed_batches
        );
    }
    return training_batch;
}

TensorVec
MadnisTraining::build_buffered_training_batch(const std::vector<size_t>& counts) {
    TensorVec args;
    for (auto& channel : _channels) {
        for (auto& tensor : channel.buffer.tensors) {
            args.push_back(tensor.slice(0, 0, channel.buffer.size));
        }
    }
    args.emplace_back(counts);
    std::size_t batch_seq = _buffered_batch_seq++;
    if (_seed) {
        _multi_channel_sampler->set_seed(DerivedSeed(
            _seed, DerivedSeed::madnis_sample_buffer, batch_seq, _channel_index_offset
        ));
    }
    return _multi_channel_sampler->run(args);
}

void MadnisTraining::process_job_results(const std::vector<std::size_t>& job_ids) {
    for (auto job_id : job_ids) {
        auto& job = _running_jobs.at(job_id);
        // mark ready, committed below strictly in dispatch order
        if (job.samples.channel_sizes.size() == 0) {
            _channels.at(job.samples.channel_index)
                .ready_job_ids.emplace(job.dispatch_seq, job_id);
        } else {
            _multi_job_ready_job_ids.emplace(job.dispatch_seq, job_id);
        }
    }

    // commit each channel's ready single-channel jobs strictly in dispatch order
    for (auto& channel : _channels) {
        for (auto it = channel.ready_job_ids.find(channel.commit_cursor);
             it != channel.ready_job_ids.end();
             it = channel.ready_job_ids.find(channel.commit_cursor)) {
            auto committed_job = std::move(_running_jobs.extract(it->second).mapped());
            channel.ready_job_ids.erase(it);
            ++channel.commit_cursor;
            channel.sample_count += committed_job.samples.size;
            _generated_event_count += committed_job.samples.size;
            channel.sample_batches.push_back(std::move(committed_job.samples));
            if (committed_job.unweighted_samples.size > 0) {
                // flushed into buffer at the start of the next round (see
                // start_generator_jobs)
                channel.pending_buffer_samples.push_back(
                    std::move(committed_job.unweighted_samples)
                );
            }
        }
    }

    // commit ready multi-channel (GPU) jobs strictly in dispatch order
    for (auto it = _multi_job_ready_job_ids.find(_multi_job_commit_cursor);
         it != _multi_job_ready_job_ids.end();
         it = _multi_job_ready_job_ids.find(_multi_job_commit_cursor)) {
        auto multi_job = std::move(_running_jobs.extract(it->second).mapped());
        _multi_job_ready_job_ids.erase(it);
        ++_multi_job_commit_cursor;
        std::size_t offset = 0, unw_offset = 0, chan_index = 0;
        SampleBatch chan_unweighted_samples;
        for (auto [channel, chan_size] :
             zip(_channels, multi_job.samples.channel_sizes)) {
            if (chan_size == 0) {
                ++chan_index;
                continue;
            }
            channel.sample_count += chan_size;
            _generated_event_count += chan_size;
            channel.sample_batches.emplace_back();
            auto& batch = channel.sample_batches.back();
            batch.tensors.reserve(multi_job.samples.tensors.size());
            for (auto& tensor : multi_job.samples.tensors) {
                batch.tensors.push_back(tensor.slice(0, offset, offset + chan_size));
            }
            if (multi_job.unweighted_samples.channel_sizes.size() > 0) {
                std::size_t unw_chan_size =
                    multi_job.unweighted_samples.channel_sizes.at(chan_index);
                chan_unweighted_samples.tensors.clear();
                chan_unweighted_samples.size = unw_chan_size;
                for (auto& tensor : multi_job.unweighted_samples.tensors) {
                    chan_unweighted_samples.tensors.push_back(
                        tensor.slice(0, unw_offset, unw_offset + unw_chan_size)
                    );
                }
                channel.pending_buffer_samples.push_back(
                    std::move(chan_unweighted_samples)
                );
                unw_offset += unw_chan_size;
            }
            batch.size = chan_size;
            offset += chan_size;
            ++chan_index;
        }
    }
}

void MadnisTraining::buffer_store(ChannelData& channel, SampleBatch& samples) {
    std::size_t empty_count = _config.buffer_capacity - channel.buffer.size;

    std::size_t size = std::min(samples.size, _config.buffer_capacity);
    std::size_t end_index = channel.buffer.consumed_count + size;
    std::size_t store_begin1, store_end1, load_begin1, load_end1;
    std::size_t store_begin2 = 0, store_end2 = 0, load_begin2 = 0, load_end2 = 0;
    if (end_index < _config.buffer_capacity) {
        store_begin1 = channel.buffer.consumed_count;
        store_end1 = end_index;
        load_begin1 = 0;
        load_end1 = size;
    } else {
        store_begin1 = channel.buffer.consumed_count;
        store_end1 = _config.buffer_capacity;
        load_begin1 = 0;
        load_end1 = _config.buffer_capacity - channel.buffer.consumed_count;
        store_begin2 = 0;
        store_end2 = end_index - _config.buffer_capacity;
        load_begin2 = _config.buffer_capacity - channel.buffer.consumed_count;
        load_end2 = size;
    }
    channel.buffer.consumed_count = end_index % _config.buffer_capacity;
    channel.buffer.size = std::min(channel.buffer.size + size, _config.buffer_capacity);
    for (auto [buf, tensor] : zip(channel.buffer.tensors, samples.tensors)) {
        buf.slice(0, store_begin1, store_end1)
            .copy_from(tensor.slice(0, load_begin1, load_end1));
        if (load_end2 - load_begin2 > 0) {
            buf.slice(0, store_begin2, store_end2)
                .copy_from(tensor.slice(0, load_begin2, load_end2));
        }
    }
}

void MadnisTraining::update_history(
    const TensorVec& results,
    const std::vector<std::size_t>& counts,
    double learning_rate,
    bool buffered
) {
    Tensor loss_cpu = results.at(0).cpu();
    Tensor abs_means_cpu = results.at(1).cpu();
    Tensor variances_cpu = results.at(2).cpu();
    double loss = loss_cpu.view<double, 1>()[0];
    if (loss > 1e6) {
        ++_diverged_batch_count;
        if (_diverged_batch_count > 10) {
            throw std::runtime_error("MadNIS training diverged. Please restart");
        }
    } else {
        _diverged_batch_count = 0;
    }
    if (_loss_history.size() < _config.log_interval) {
        _loss_history.push_back(loss);
        _lr_history.push_back(learning_rate);
        _buffered_history.push_back(buffered);
    } else {
        _loss_history.at(_loss_history_index) = loss;
        _lr_history.at(_loss_history_index) = learning_rate;
        _buffered_history.at(_loss_history_index) = buffered;
    }
    if (++_loss_history_index == _config.log_interval) {
        _loss_history_index = 0;
    }

    auto abs_means = abs_means_cpu.view<double, 2>()[0];
    auto variances = variances_cpu.view<double, 2>()[0];
    if (abs_means.size() != _channels.size() || variances.size() != _channels.size()) {
        throw std::logic_error("wrong channel count returned");
    }
    for (std::size_t i = 0; auto [channel, count] : zip(_channels, counts)) {
        if (channel.integration_history.size() < _config.integration_history_length) {
            channel.integration_history.push_back({count, abs_means[i], variances[i]});
        } else {
            channel.integration_history.at(channel.history_index) = {
                count, abs_means[i], variances[i]
            };
        }
        if (++channel.history_index == _config.log_interval) {
            channel.history_index = 0;
        }
        ++i;
    }
}

void MadnisTraining::drop_channels(std::size_t batch) {
    std::vector<double> abs_means;
    abs_means.reserve(_channels.size());
    double abs_mean_sum = 0.;
    for (auto& channel : _channels) {
        std::size_t chan_count = 0;
        double chan_abs_mean = 0.0;
        for (auto [count, abs_mean, var] : channel.integration_history) {
            if (!std::isnan(abs_mean)) {
                chan_count += count;
                chan_abs_mean += count * abs_mean;
            }
        }
        abs_mean_sum += chan_abs_mean;
        abs_means.push_back(chan_abs_mean);
    }
    std::vector<std::size_t> indices(_channels.size());
    std::iota(indices.begin(), indices.end(), 0);
    // stable_sort: exact ties (e.g. charge-conjugate channels) must break consistently
    std::stable_sort(indices.begin(), indices.end(), [&](auto i, auto j) {
        return abs_means.at(i) < abs_means.at(j);
    });

    Tensor active_mask_glob = _optimizer_context->global(_cwnet.value().mask_name());
    Tensor active_mask = active_mask_glob.cpu();
    auto mask_view = active_mask.view<double, 2>()[0];

    double drop_sum = 0.;
    double drop_threshold =
        _config.channel_dropping_threshold * std::min(1000. / batch, 1.);
    std::size_t drop_count = 0;
    for (std::size_t chan_index : indices) {
        drop_sum += abs_means.at(chan_index);
        if (drop_sum / abs_mean_sum > drop_threshold) {
            break;
        }
        auto& channel = _channels.at(chan_index);

        if (_active_flavors_count.size() > 0) {
            auto& active_flavors = channel.integrand->active_flavors();
            if (std::any_of(
                    active_flavors.begin(),
                    active_flavors.end(),
                    [&](std::size_t flav_index) {
                        return _active_flavors_count.at(flav_index) == 0;
                    }
                )) {
                // a flavor of this channel has no other channel to fall back to
                continue;
            }
            for (std::size_t flav_index : active_flavors) {
                --_active_flavors_count.at(flav_index);
            }
        }

        for (me_int_t index : channel.integrand->channel_indices()) {
            if (index < 0 || index > mask_view.size()) {
                throw std::out_of_range("channel index out of bounds");
            }
            mask_view[index] = 0;
        }
        channel.integrand = nullptr;
        ++drop_count;
    }
    if (drop_count == 0) {
        return;
    }
    _channels.erase(
        std::remove_if(
            _channels.begin(),
            _channels.end(),
            [&](auto& channel) { return !channel.integrand; }
        ),
        _channels.end()
    );
    if (_optimizer_context->device()->device_type() != DeviceType::cpu) {
        active_mask_glob.copy_from(active_mask);
    }
    _generator_context->global(_cwnet.value().mask_name()).copy_from(active_mask);
    build_runtimes_and_optimizer();
}

void MadnisTraining::freeze_cwnet() {
    if (!_cwnet) {
        return;
    }
    for (auto& name : _cwnet->mlp().global_names()) {
        _optimizer_context->set_global_requires_grad(name, false);
    }
    build_runtimes_and_optimizer();
}

namespace {

// thrown into the trainings still running when MultiMadnisTraining stops them
struct TrainingStopped : std::runtime_error {
    TrainingStopped() : std::runtime_error("MadNIS training stopped") {}
};

std::string device_name(Context& context) {
    auto device = context.device();
    switch (device->device_type()) {
    case DeviceType::cpu:
        return "cpu";
    case DeviceType::cuda:
        return std::format("cuda:{}", device->device_index());
    case DeviceType::hip:
        return std::format("hip:{}", device->device_index());
    }
    return "?";
}

} // namespace

MultiMadnisTraining::MultiMadnisTraining(
    const std::vector<ContextPtr>& generator_contexts,
    const std::vector<TrainingArgs>& training_args,
    const ParallelConfig& parallel_config,
    Verbosity verbosity,
    std::shared_ptr<StatusFile> status_file,
    std::optional<std::uint64_t> seed
) :
    _generator_contexts(generator_contexts),
    _training_args(training_args),
    _parallel_config(parallel_config),
    _verbosity(verbosity),
    _seed(seed),
    _status_file(status_file) {
    if (_generator_contexts.empty()) {
        throw std::invalid_argument("MultiMadnisTraining needs a generator context");
    }
    _trainings_status = nlohmann::json::array();
    for (std::size_t i = 0; i < _training_args.size(); ++i) {
        _trainings_status.push_back(
            {{"subprocess", i},
             {"batch", nlohmann::json::array()},
             {"batch_count", _training_args.at(i).config.batches},
             {"losses", nlohmann::json::array()},
             {"channel_counts", nlohmann::json::array()},
             {"learning_rates", nlohmann::json::array()},
             {"buffered_fractions", nlohmann::json::array()},
             {"generated_events", nlohmann::json::array()},
             {"buffer_sizes", nlohmann::json::array()}}
        );
    }
}

MultiMadnisTraining::~MultiMadnisTraining() { stop_all(); }

void MultiMadnisTraining::train() {
    std::size_t subproc_count = _training_args.size();
    _subprocs.assign(subproc_count, {});
    _active_channels.assign(subproc_count, {});
    // each subprocess gets its own channel_index slice so their derived seed
    // streams don't collide, independently of the order the trainings start in
    for (std::size_t offset = 0; auto [subproc, args] : zip(_subprocs, _training_args)) {
        subproc.channel_index_offset = offset;
        offset += args.integrands.size();
    }
    _devices.assign(_generator_contexts.size(), {});
    for (std::size_t device_index = 0; auto& device : _devices) {
        // the thread pools may have been busy before, e.g. with the survey
        device.window_busy_ns = device_busy_ns(device_index);
        ++device_index;
    }
    for (auto& context : _generator_contexts) {
        context->reset_cache();
    }
    _stop = false;
    _status_written = false;
    _done_count = 0;

    print_progress_init();
    _start_time = std::chrono::steady_clock::now();
    _start_cpu_microsec = cpu_time_microsec();
    _window_start = _start_time;
    if (_subprocs.empty()) {
        print_done();
        return;
    }

    using namespace std::chrono_literals;
    auto check_interval = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(_parallel_config.check_interval)
    );
    try {
        while (true) {
            MadnisTraining::check_abort();
            process_events();
            if (collect_finished()) {
                break;
            }
            start_trainings_on_idle_devices();
            auto now = std::chrono::steady_clock::now();
            if (now - _window_start >= check_interval) {
                start_training_by_occupancy(now);
            }
            std::unique_lock<std::mutex> lock(_mutex);
            _cv.wait_for(lock, 100ms, [&] { return _finished_count > 0; });
        }
    } catch (...) {
        stop_all();
        throw;
    }
    for (auto& context : _generator_contexts) {
        context->set_frozen(false);
    }
    print_done();
}

void MultiMadnisTraining::stop_all() {
    _stop = true;
    for (auto& run : _running) {
        if (run.thread.joinable()) {
            run.thread.join();
        }
    }
    _running.clear();
    for (auto& context : _generator_contexts) {
        context->set_frozen(false);
    }
}

void MultiMadnisTraining::start_trainings_on_idle_devices() {
    for (std::size_t device_index = 0; device_index < _devices.size(); ++device_index) {
        if (_devices.at(device_index).running_count == 0) {
            start_training(device_index);
        }
    }
}

void MultiMadnisTraining::start_training_by_occupancy(
    std::chrono::steady_clock::time_point now
) {
    double window_sec = std::chrono::duration<double>(now - _window_start).count();
    std::optional<std::size_t> best_device;
    double best_occupancy = _parallel_config.target_occupancy;
    for (std::size_t device_index = 0; auto& device : _devices) {
        auto& context = *_generator_contexts.at(device_index);
        std::int64_t busy_ns = device_busy_ns(device_index);
        double occupancy = (busy_ns - device.window_busy_ns) * 1e-9 /
            (window_sec * context.thread_pool().thread_count());
        device.window_busy_ns = busy_ns;
        // started within the window: its occupancy does not show the new
        // training fully yet
        bool settled = device.last_start < _window_start;
        if (settled && occupancy < best_occupancy) {
            best_occupancy = occupancy;
            best_device = device_index;
        }
        ++device_index;
    }
    _window_start = now;
    if (best_device) {
        start_training(*best_device);
    }
}

std::int64_t MultiMadnisTraining::device_busy_ns(std::size_t device_index) const {
    auto& context = *_generator_contexts.at(device_index);
    bool is_gpu = context.device()->device_type() != DeviceType::cpu;
    std::int64_t busy_ns = context.thread_pool().busy_time().count() +
        _devices.at(device_index).finished_busy_ns;
    for (auto& run : _running) {
        if (run.device_index == device_index) {
            busy_ns += run.optimizer_context->thread_pool().busy_time().count();
            if (is_gpu) {
                busy_ns += run.training->optimizer_time().count();
            }
        }
    }
    return busy_ns;
}

// The GPUs take the subprocesses with the highest multiplicity (the costly
// ones), the CPUs those with the lowest; ties are broken by the number of
// batches times channels in the same direction, and then by subprocess index.
std::optional<std::size_t> MultiMadnisTraining::next_subprocess(bool is_gpu) const {
    std::optional<std::size_t> best;
    std::pair<std::size_t, std::size_t> best_key;
    for (std::size_t index = 0; auto& subproc : _subprocs) {
        if (subproc.state == SubprocState::queued) {
            auto& args = _training_args.at(index);
            std::pair<std::size_t, std::size_t> key{
                args.multiplicity, args.config.batches * args.integrands.size()
            };
            if (!best || (is_gpu ? key > best_key : key < best_key)) {
                best = index;
                best_key = key;
            }
        }
        ++index;
    }
    return best;
}

void MultiMadnisTraining::start_training(std::size_t device_index) {
    std::size_t max_running = std::max<std::size_t>(
        _parallel_config.max_parallel_trainings, 1
    );
    if (_running.size() >= max_running) {
        return;
    }
    auto& gen_context = _generator_contexts.at(device_index);
    bool is_gpu = gen_context->device()->device_type() != DeviceType::cpu;
    auto subproc_index = next_subprocess(is_gpu);
    if (!subproc_index) {
        return;
    }
    auto& subproc = _subprocs.at(*subproc_index);
    auto& args = _training_args.at(*subproc_index);
    auto& run = _running.emplace_back();
    run.subproc_index = *subproc_index;
    run.device_index = device_index;
    run.optimizer_context = std::make_shared<Context>(
        gen_context->device(), gen_context->thread_pool().thread_count()
    );
    // built here, not on the thread of the training: the first training on a
    // context adds the resources its runtimes need to it, which is only safe
    // before the context is frozen and shared
    run.training = std::make_unique<MadnisTraining>(
        gen_context,
        run.optimizer_context,
        args.config,
        args.integrands,
        args.cwnet,
        _seed,
        subproc.channel_index_offset,
        true
    );
    run.training->set_abort_check([this] {
        if (_stop) {
            throw TrainingStopped();
        }
    });
    gen_context->set_frozen(true);
    subproc.state = SubprocState::running;
    subproc.device = device_name(*gen_context);
    auto& device = _devices.at(device_index);
    ++device.running_count;
    device.last_start = std::chrono::steady_clock::now();
    if (_verbosity == Verbosity::log) {
        Logger::info(
            std::format(
                "training, subproc: {} / {}, started on {}",
                *subproc_index + 1,
                _training_args.size(),
                subproc.device
            )
        );
    }
    run.thread = std::thread([this, &run] { run_training(run); });
}

void MultiMadnisTraining::run_training(Running& run) {
    try {
        auto& training = *run.training;
        _generator_contexts.at(run.device_index)->device()->activate();
        std::size_t batch_count = training.config().batches;
        std::size_t log_interval = training.config().log_interval;
        for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
            training.train_step(batch_index);
            ProgressEvent event{
                run.subproc_index,
                batch_index,
                training.average_loss(),
                training.active_channel_count()
            };
            bool done = batch_index + 1 == batch_count;
            if (done || (batch_index + 1) % log_interval == 0) {
                event.status = {
                    {"subprocess", run.subproc_index},
                    {"batch", training.status_batches()},
                    {"batch_count", batch_count},
                    {"losses", training.status_losses()},
                    {"channel_counts", training.status_channel_counts()},
                    {"learning_rates", training.status_learning_rates()},
                    {"buffered_fractions", training.status_buffered_fractions()},
                    {"generated_events", training.status_generated_events()},
                    {"buffer_sizes", training.status_buffer_sizes()}
                };
            }
            std::lock_guard<std::mutex> lock(_mutex);
            _events.push_back(std::move(event));
        }
        // the event generators run on any of the contexts
        for (auto& context : _generator_contexts) {
            context->copy_globals_from(*run.optimizer_context, training.trained_globals());
        }
        std::lock_guard<std::mutex> lock(_mutex);
        _active_channels.at(run.subproc_index) = training.active_channels();
    } catch (...) {
        std::lock_guard<std::mutex> lock(_mutex);
        run.exception = std::current_exception();
    }
    {
        std::lock_guard<std::mutex> lock(_mutex);
        run.done = true;
        ++_finished_count;
    }
    _cv.notify_all();
}

bool MultiMadnisTraining::collect_finished() {
    std::exception_ptr exception;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_finished_count == 0) {
            return false;
        }
        _finished_count = 0;
    }
    for (auto it = _running.begin(); it != _running.end();) {
        bool done;
        {
            std::lock_guard<std::mutex> lock(_mutex);
            done = it->done;
        }
        if (!done) {
            ++it;
            continue;
        }
        it->thread.join();
        if (it->exception && !exception) {
            exception = it->exception;
        }
        auto& device = _devices.at(it->device_index);
        bool is_gpu = _generator_contexts.at(it->device_index)->device()->device_type() !=
            DeviceType::cpu;
        device.finished_busy_ns += it->optimizer_context->thread_pool().busy_time().count();
        if (is_gpu) {
            device.finished_busy_ns += it->training->optimizer_time().count();
        }
        --device.running_count;
        auto& subproc = _subprocs.at(it->subproc_index);
        subproc.state = SubprocState::done;
        ++_done_count;
        it = _running.erase(it);
    }
    if (exception) {
        std::rethrow_exception(exception);
    }
    // process the last progress events of the trainings that finished
    process_events();
    return _done_count == _subprocs.size();
}

void MultiMadnisTraining::process_events() {
    std::deque<ProgressEvent> events;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        events.swap(_events);
    }
    bool status_due = false;
    for (auto& event : events) {
        auto& subproc = _subprocs.at(event.subproc_index);
        subproc.batch_index = event.batch_index;
        subproc.loss = event.loss;
        subproc.channel_count = event.channel_count;
        auto& config = _training_args.at(event.subproc_index).config;
        if (_verbosity == Verbosity::log &&
            (event.batch_index + 1) % config.log_interval == 0) {
            Logger::info(
                std::format(
                    "training, subproc: {} / {}, batch: {} / {}, loss: {:.4f}, "
                    "channels: {}, time: {:%H:%M:%S}",
                    event.subproc_index + 1,
                    _subprocs.size(),
                    event.batch_index + 1,
                    config.batches,
                    event.loss,
                    event.channel_count,
                    std::chrono::round<std::chrono::seconds>(
                        std::chrono::steady_clock::now() - _start_time
                    )
                )
            );
        }
        if (!event.status.is_null()) {
            _trainings_status.at(event.subproc_index) = std::move(event.status);
            status_due = true;
        }
    }
    if (status_due || !_status_written) {
        write_status(false);
    }
    if (_verbosity == Verbosity::pretty && !events.empty()) {
        using namespace std::chrono_literals;
        auto now = std::chrono::steady_clock::now();
        if (now - _last_print_time >= 0.1s) {
            _last_print_time = now;
            print_progress();
        }
    }
}

void MultiMadnisTraining::print_progress_init() {
    _last_print_time = std::chrono::steady_clock::now();
    if (_verbosity != Verbosity::pretty) {
        Logger::info("training started");
        return;
    }

    if (_training_args.size() > 1) {
        _pretty_box_lower = PrettyBox(
            "Subprocesses", _training_args.size() + 1, {12, 10, 12, 12, 0}
        );
        _pretty_box_lower.set_row(
            0, {"Subprocess", "Device", "Loss", "Channels", "Batch"}
        );
        for (std::size_t i = 0; i < _training_args.size(); ++i) {
            _pretty_box_lower.set_row(i + 1, {std::format("{}", i), "queued"});
        }
        _pretty_box_upper =
            PrettyBox("MadNIS training", 3, {18, 0}, _pretty_box_lower.line_count());
        _pretty_box_upper.set_column(0, {"Subprocesses:", "Running:", "Run time:"});
        _pretty_box_upper.print_first();
        _pretty_box_lower.print_first();
    } else {
        _pretty_box_upper = PrettyBox("MadNIS training", 4, {18, 0});
        _pretty_box_upper.set_column(0, {"Batch:", "Loss:", "Channels:", "Run time:"});
        _pretty_box_upper.print_first();
    }
}

void MultiMadnisTraining::print_progress() { print_progress_update(false); }

void MultiMadnisTraining::print_done() {
    using namespace std::chrono_literals;
    auto now = std::chrono::steady_clock::now();
    double wall_time_sec = (now - _start_time) / 1.0s;
    double cpu_time_sec = (cpu_time_microsec() - _start_cpu_microsec) / 1e6;
    if (_verbosity == Verbosity::log) {
        Logger::info(
            std::format(
                "training done, {}", format_run_time(wall_time_sec, cpu_time_sec)
            )
        );
    } else if (_verbosity == Verbosity::pretty) {
        print_progress_update(true);
    }
    write_status(true);
}

void MultiMadnisTraining::print_progress_update(bool done) {
    using namespace std::chrono_literals;
    auto now = std::chrono::steady_clock::now();
    std::string time_str;
    if (done) {
        double wall_time_sec = (now - _start_time) / 1.0s;
        double cpu_time_sec = (cpu_time_microsec() - _start_cpu_microsec) / 1e6;
        time_str = format_run_time(wall_time_sec, cpu_time_sec);
    } else {
        time_str = std::format(
            "{:%H:%M:%S}", std::chrono::round<std::chrono::seconds>(now - _start_time)
        );
    }

    if (_subprocs.size() == 1) {
        auto& subproc = _subprocs.at(0);
        std::size_t batches = _training_args.at(0).config.batches;
        std::size_t batch = subproc.state == SubprocState::queued
            ? 0
            : subproc.batch_index + 1;
        std::string progress_bar =
            done ? "" : format_progress(static_cast<double>(batch) / batches, 52);
        _pretty_box_upper.set_column(
            1,
            {std::format("{:<15} {}", std::format("{} / {}", batch, batches), progress_bar),
             std::format("{:>.4f}", subproc.loss),
             std::format("{}", subproc.channel_count),
             time_str}
        );
        _pretty_box_upper.print_update();
        return;
    }

    // the global progress bar weights each subprocess by its batch count
    std::size_t total_batches = 0, batches_done = 0;
    for (auto [subproc, args] : zip(_subprocs, _training_args)) {
        total_batches += args.config.batches;
        if (subproc.state == SubprocState::done) {
            batches_done += args.config.batches;
        } else if (subproc.state == SubprocState::running) {
            batches_done += subproc.batch_index + 1;
        }
    }
    std::string progress_bar_all = done
        ? ""
        : format_progress(static_cast<double>(batches_done) / total_batches, 52);
    _pretty_box_upper.set_column(
        1,
        {std::format(
             "{:<15} {}",
             std::format("{} / {}", _done_count, _subprocs.size()),
             progress_bar_all
         ),
         std::format(
             "{} (max {})",
             _running.size(),
             std::max<std::size_t>(_parallel_config.max_parallel_trainings, 1)
         ),
         time_str}
    );
    for (std::size_t index = 0; auto [subproc, args] : zip(_subprocs, _training_args)) {
        std::size_t batches = args.config.batches;
        std::string device, loss, channels, batch;
        if (subproc.state == SubprocState::queued) {
            device = "queued";
        } else {
            device = subproc.device;
            loss = std::format("{:>.4f}", subproc.loss);
            channels = std::format("{}", subproc.channel_count);
            std::string batch_str =
                std::format("{} / {}", subproc.batch_index + 1, batches);
            batch = subproc.state == SubprocState::done
                ? batch_str
                : std::format(
                      "{:<15} {}",
                      batch_str,
                      format_progress(
                          static_cast<double>(subproc.batch_index + 1) / batches, 30
                      )
                  );
        }
        _pretty_box_lower.set_row(
            index + 1, {std::format("{}", index), device, loss, channels, batch}
        );
        ++index;
    }
    _pretty_box_upper.print_update();
    _pretty_box_lower.print_update();
}

void MultiMadnisTraining::write_status(bool done) {
    if (!_status_file) {
        return;
    }
    using namespace std::chrono_literals;
    auto now = std::chrono::steady_clock::now();
    nlohmann::json j{
        {"status", done ? "done" : "training"},
        {"madnis_trainings", _trainings_status},
        {"run_times",
         {{"training",
           {{"wall_time_sec", (now - _start_time) / 1.0s},
            {"cpu_time_sec", (cpu_time_microsec() - _start_cpu_microsec) / 1e6}}}}},
    };
    _status_file->write(j, done || !_status_written);
    _status_written = true;
}

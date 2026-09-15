#include "madspace/phasespace/integrand.hpp"

#include "madspace/util.hpp"

#include <set>

using namespace madspace;

namespace {

std::size_t final_channel_count(
    const std::vector<DifferentialCrossSection>& diff_xs,
    const nested_vector2<me_int_t>& first_chan_weight_remap,
    std::size_t first_remapped_chan_count,
    const std::vector<me_int_t>& second_chan_weight_remap,
    std::size_t second_remapped_chan_count,
    const std::optional<PropagatorChannelWeights>& prop_chan_weights,
    const std::optional<SubchannelWeights>& subchan_weights
) {
    if (second_chan_weight_remap.size() > 0) {
        return second_remapped_chan_count;
    } else if (subchan_weights) {
        return subchan_weights->channel_count();
    } else if (first_chan_weight_remap.size() > 0 || prop_chan_weights) {
        // with the denominators sde_strategy, prop_chan_weights already
        // produces first_remapped_chan_count weights without needing a remap
        return first_remapped_chan_count;
    } else {
        return diff_xs.at(0).matrix_element().diagram_count();
    }
}

} // namespace

static const BatchSize acc_batch_size("acc_batch_size");

Integrand::Integrand(
    const PhaseSpaceMapping& mapping,
    const std::vector<DifferentialCrossSection>& diff_xs,
    const AdaptiveMapping& adaptive_map,
    const AdaptiveDiscrete& discrete_sym,
    const AdaptiveDiscrete& discrete_flavor,
    const nested_vector2<me_int_t>& pid_options,
    const std::optional<PdfGrid>& pdf_grid,
    const std::optional<RunningCoupling>& running_coupling,
    const std::optional<EnergyScale>& energy_scale,
    const std::optional<PropagatorChannelWeights>& prop_chan_weights,
    const std::optional<SubchannelWeights>& subchan_weights,
    const std::optional<ChannelWeightNetwork>& chan_weight_net,
    const nested_vector2<me_int_t>& first_chan_weight_remap,
    std::size_t first_remapped_chan_count,
    const std::vector<me_int_t>& second_chan_weight_remap,
    std::size_t second_remapped_chan_count,
    bool madnis_training,
    bool drop_cuts_and_rescale,
    bool partial_weights,
    const std::vector<std::size_t>& channel_indices,
    const nested_vector2<std::size_t>& active_flavors,
    const std::vector<std::size_t>& flavor_remap,
    const std::vector<double>& flavor_factors,
    const std::vector<bool>& flavor_mirror,
    const std::vector<std::size_t>& flavor_diff_xs_indices,
    const std::vector<std::size_t>& flavor_subproc_indices,
    const std::vector<std::size_t>& flavor_per_subproc_remap,
    std::size_t compressed_channel_weight_count
) :
    FunctionGenerator(
        "Integrand",
        {{"batch_size", Type({batch_size})}},
        [&] {
            NamedVector<Type> ret_types;
            auto& diff_xs_first = diff_xs.at(0);
            auto flav_count = pid_options.size();
            if (madnis_training) {
                if (std::holds_alternative<std::monostate>(adaptive_map)) {
                    throw std::invalid_argument(
                        "madnis_training requires an adaptive mapping"
                    );
                }
                ret_types.push_back("full_weight", batch_float);
                ret_types.push_back("weight", batch_float);
                ret_types.push_back("latent", batch_float_array(mapping.random_dim()));
                ret_types.push_back("adaptive_prob", batch_float);
                ret_types.push_back("channel_index", batch_int);

                std::size_t madnis_channel_count = final_channel_count(
                    diff_xs,
                    first_chan_weight_remap,
                    first_remapped_chan_count,
                    second_chan_weight_remap,
                    second_remapped_chan_count,
                    prop_chan_weights,
                    subchan_weights
                );
                if (madnis_channel_count > 1 &&
                    madnis_channel_count * 8 > compressed_channel_weight_count * 12) {
                    ret_types.push_back(
                        "channel_weight_values",
                        batch_float_array(compressed_channel_weight_count)
                    );
                    ret_types.push_back(
                        "channel_weight_indices",
                        batch_int_array(compressed_channel_weight_count)
                    );
                } else {
                    ret_types.push_back(
                        "channel_weights", batch_float_array(madnis_channel_count)
                    );
                }
                ret_types.push_back(
                    "cwnet_input",
                    batch_float_array(
                        chan_weight_net.value().preprocessing().output_dim()
                    )
                );
                ret_types.push_back("channel_index_in_group", batch_int);
                if (flav_count > 1 &&
                    !std::holds_alternative<std::monostate>(discrete_flavor)) {
                    ret_types.push_back("discrete_flavor_index", batch_int);
                    if (pdf_grid && energy_scale) {
                        ret_types.push_back("pdf_prior", batch_float_array(flav_count));
                    }
                }
            } else {
                ret_types.push_back("weight", batch_float);
                ret_types.push_back(
                    "momenta", batch_four_vec_array(mapping.particle_count())
                );
                ret_types.push_back("color_index", batch_int);
                ret_types.push_back("helicity_index", batch_int);
                ret_types.push_back("diagram_index", batch_int);
                ret_types.push_back("flavor_index", batch_int);
                if (flavor_diff_xs_indices.size() > 0) {
                    ret_types.push_back("subprocess_index", batch_int);
                }
                ret_types.push_back("ren_scale", batch_float);
                ret_types.push_back("alpha_qcd", batch_float);
                // The per-beam factorisation scales ride along with x1/x2 in
                // the event record. They are worth keeping for any hadronic
                // run, not just a reweighted one: a dynamical scale choice can
                // give the two beams different scales, and without them the
                // LHE writer has nothing to report but mu_R.
                if (diff_xs_first.has_pdf(0)) {
                    ret_types.push_back("x1", batch_float);
                    ret_types.push_back("fact_scale1", batch_float);
                }
                if (diff_xs_first.has_pdf(1)) {
                    ret_types.push_back("x2", batch_float);
                    ret_types.push_back("fact_scale2", batch_float);
                }
                if (partial_weights &&
                    (diff_xs_first.has_pdf(0) || diff_xs_first.has_pdf(1))) {
                    ret_types.push_back("partial_weight_product", batch_float);
                }
                if (energy_scale && energy_scale->is_mlm()) {
                    ret_types.push_back(
                        "cluster_scales",
                        batch_float_array(mapping.particle_count() - 2)
                    );
                }
                ret_types.push_back("random", batch_float_array(mapping.random_dim()));
                if (mapping.channel_count() > 1 &&
                    !std::holds_alternative<std::monostate>(discrete_sym)) {
                    ret_types.push_back("channel_index_in_group", batch_int);
                }
                if (flav_count > 1 &&
                    !std::holds_alternative<std::monostate>(discrete_flavor)) {
                    ret_types.push_back("discrete_flavor_index", batch_int);
                }
            }
            return ret_types;
        }()
    ),
    _mapping(mapping),
    _diff_xs(diff_xs),
    _adaptive_map(adaptive_map),
    _discrete_sym(discrete_sym),
    _discrete_flavor(discrete_flavor),
    _pid_options(pid_options),
    _running_coupling(running_coupling),
    _energy_scale(energy_scale),
    _prop_chan_weights(prop_chan_weights),
    _subchan_weights(subchan_weights),
    _chan_weight_net(chan_weight_net),
    _first_chan_weight_remap(first_chan_weight_remap),
    _first_remapped_chan_count(first_remapped_chan_count),
    _second_chan_weight_remap(second_chan_weight_remap),
    _second_remapped_chan_count(second_remapped_chan_count),
    _compressed_channel_weight_count(compressed_channel_weight_count),
    _madnis_training(madnis_training),
    _drop_cuts_and_rescale(drop_cuts_and_rescale),
    _partial_weights(partial_weights),
    _channel_indices(channel_indices.begin(), channel_indices.end()),
    _random_dim(
        mapping.random_dim() +          // phasespace
        (mapping.channel_count() > 1) + // symmetric channel
        (pid_options.size() > 1) +      // flavor
        // flipped initial state
        std::any_of(flavor_mirror.begin(), flavor_mirror.end(), std::identity{})
    ),
    _flavor_remap(flavor_remap.begin(), flavor_remap.end()),
    _flavor_factors(flavor_factors),
    _flavor_diff_xs_indices(
        flavor_diff_xs_indices.begin(), flavor_diff_xs_indices.end()
    ),
    _flavor_subproc_indices(
        flavor_subproc_indices.begin(), flavor_subproc_indices.end()
    ),
    _flavor_per_subproc_remap(
        flavor_per_subproc_remap.begin(), flavor_per_subproc_remap.end()
    ) {
    if (pdf_grid) {
        for (std::size_t i = 0; i < 2; ++i) {
            std::set<int> pids;
            for (auto& option : pid_options) {
                pids.insert(option.at(i));
            }
            for (auto& option : pid_options) {
                _pdf_indices.at(i).push_back(
                    std::distance(pids.begin(), pids.find(option.at(i)))
                );
            }
            _pdfs.at(i) = PartonDensity(pdf_grid.value(), {pids.begin(), pids.end()});
        }
        if (energy_scale && energy_scale->mlm_pdf_reweighting()) {
            // Everything the reweighting can ask for: the gluon, whatever the
            // beams can be, and the flavours the clustering pinned down along
            // the way. The kernel names these by class, not by pdg, because
            // one clustering serves every flavour channel; the table turns
            // (class, sampled option) into an index into this one density.
            auto& absolute = energy_scale->mlm_pdf_absolute_pdgs();
            std::vector<int> rw_pids{21};
            auto index_of = [&](int pid) {
                auto found = std::find(rw_pids.begin(), rw_pids.end(), pid);
                if (found == rw_pids.end()) {
                    rw_pids.push_back(pid);
                    return static_cast<me_int_t>(rw_pids.size() - 1);
                }
                return static_cast<me_int_t>(found - rw_pids.begin());
            };
            std::size_t class_count = 3 + absolute.size();
            std::size_t option_count = std::max<std::size_t>(pid_options.size(), 1);
            _pdf_rw_table.resize(class_count * option_count);
            for (std::size_t f = 0; f < option_count; ++f) {
                _pdf_rw_table.at(0 * option_count + f) = index_of(21);
                for (std::size_t beam = 0; beam < 2; ++beam) {
                    _pdf_rw_table.at((1 + beam) * option_count + f) = index_of(
                        pid_options.size() > 0 ? pid_options.at(f).at(beam) : 21
                    );
                }
                for (std::size_t c = 0; c < absolute.size(); ++c) {
                    _pdf_rw_table.at((3 + c) * option_count + f) =
                        index_of(absolute.at(c));
                }
            }
            for (std::size_t c = 0; c < class_count; ++c) {
                _pdf_rw_class_offsets.push_back(
                    static_cast<me_int_t>(c * option_count)
                );
            }
            _pdf_rw = PartonDensity(pdf_grid.value(), rw_pids, true);
        }
    }

    if (energy_scale && energy_scale->mlm_history_per_diagram()) {
        // The diagram is picked from the matrix element's own diagram weights,
        // and those only line up with the clustering's diagram numbering for
        // a single matrix element.
        if (diff_xs.size() != 1) {
            throw std::invalid_argument(
                "an MLM clustering history per diagram needs a single matrix element"
            );
        }
        auto& starts = energy_scale->mlm_diagram_start_states();
        std::size_t diagram_count = diff_xs.at(0).matrix_element().diagram_count();
        std::vector<double> mask(diagram_count, 0.);
        _mlm_start_states.assign(diagram_count, 0);
        for (std::size_t i = 0; i < std::min(diagram_count, starts.size()); ++i) {
            mask.at(i) = starts.at(i) != 0 ? 1. : 0.;
            _mlm_start_states.at(i) = starts.at(i);
        }
        _mlm_diagram_mask.push_back(mask);
    }

    if (active_flavors.size() > 0) {
        if (active_flavors.size() != mapping.channel_count()) {
            throw std::invalid_argument(
                "a list of active flavors must be provided for each permutation"
            );
        }
        _active_flavors_mask.resize(mapping.channel_count());
        std::vector<bool> mask_all(pid_options.size());
        for (auto [mask, active] : zip(_active_flavors_mask, active_flavors)) {
            mask.resize(pid_options.size());
            for (auto index : active) {
                mask.at(index) = 1.;
                mask_all.at(index) = 1.;
            }
        }
        for (std::size_t i = 0; bool active : mask_all) {
            _active_flavors.push_back(i);
            ++i;
        }
    }

    _flavor_mirror.reserve(flavor_mirror.size());
    _has_mirror = false;
    for (bool mirror : flavor_mirror) {
        _flavor_mirror.push_back(mirror ? 2 : 1);
        _has_mirror |= mirror;
    }
    _channel_part_ret_types = compute_channel_part_ret_types();
}

std::tuple<std::vector<std::size_t>, std::vector<bool>> Integrand::latent_dims() const {
    std::vector<std::size_t> dims{_mapping.random_dim(), 1};
    std::vector<bool> is_float{true, false};

    auto flav_count = _pid_options.size();
    if (flav_count > 1 && !std::holds_alternative<std::monostate>(_discrete_flavor)) {
        dims.push_back(1);
        is_float.push_back(false);
        if ((_pdfs.at(0) || _pdfs.at(1)) && _energy_scale) {
            dims.push_back(flav_count);
            is_float.push_back(true);
        }
    }

    return {dims, is_float};
}

NamedVector<Value> Integrand::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    auto channel_out = build_channel_part(fb, args);
    return build_common_part(fb, channel_out);
}

NamedVector<Type> Integrand::compute_channel_part_ret_types() const {
    auto acc_float = Type(DataType::dt_float, acc_batch_size, {});
    auto acc_int = Type(DataType::dt_int, acc_batch_size, {});
    auto acc_float_array = [](int n) {
        return Type(DataType::dt_float, acc_batch_size, {n});
    };
    auto acc_four_vec_array = [](int n) {
        return Type(DataType::dt_float, acc_batch_size, {n, 4});
    };

    bool has_multi_flavor = _pid_options.size() > 1;
    int particle_count = static_cast<int>(_mapping.particle_count());
    int random_dim = static_cast<int>(_mapping.random_dim());

    NamedVector<Type> ret;

    // outputs before cuts
    ret.push_back("r", batch_float_array(random_dim));
    ret.push_back("latent", batch_float_array(random_dim));
    ret.push_back("weight_before_cuts", batch_float);
    ret.push_back("adaptive_prob", batch_float);
    ret.push_back("chan_index", batch_int);
    ret.push_back("chan_index_in_group", batch_int);
    if (!_madnis_training) {
        ret.push_back("momenta", batch_four_vec_array(particle_count));
    }
    if (_madnis_training) {
        ret.push_back("extra_weight_before_cuts", batch_float);
    }

    // outputs after cuts
    ret.push_back("indices_acc", Type(DataType::dt_int, acc_batch_size, {}));
    ret.push_back("momenta_acc", acc_four_vec_array(particle_count));
    if (_has_mirror) {
        if (!_madnis_training) {
            ret.push_back("momenta_mirror_acc", acc_four_vec_array(particle_count));
        }
        ret.push_back("mirror_id_acc", acc_int);
    }
    ret.push_back("x1_acc", acc_float);
    ret.push_back("x2_acc", acc_float);
    ret.push_back("flavor_id", acc_int);
    ret.push_back("weight_after_cuts", acc_float);
    if (_madnis_training && !std::holds_alternative<std::monostate>(_discrete_flavor)) {
        ret.push_back("extra_weight_after_cuts", acc_float);
    }
    ret.push_back("ren_scale", acc_float);
    if ((_pdfs.at(0) || _pdfs.at(1)) && _energy_scale) {
        auto flav_count = static_cast<int>(_pid_options.size());
        if (has_multi_flavor) {
            ret.push_back("pdf_prior", acc_float_array(flav_count));
        }
        for (std::size_t i = 0; i < 2; ++i) {
            if (_diff_xs.at(0).has_pdf(i)) {
                ret.push_back(std::format("pdf{}", i + 1), acc_float);
                ret.push_back(std::format("fact_scale{}", i + 1), acc_float);
            }
        }
    }

    if (_energy_scale && _energy_scale->is_mlm()) {
        ret.push_back("cluster_scales_acc", acc_float_array(particle_count - 2));
        ret.push_back("scale_diagram_index_acc", acc_int);
    }

    return ret;
}

NamedVector<Value> Integrand::build_channel_part(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    bool has_multi_flavor = _pid_options.size() > 1;
    bool has_permutations = _mapping.channel_count() > 1;
    auto batch_size_val = args.at("batch_size");

    Value r = fb.random(batch_size_val, _random_dim);
    ValueVec weights_before_cuts, weights_after_cuts, adaptive_probs;
    ValueVec extra_weights_before_cuts;

    // Split off discrete random numbers
    Value chan_random, flavor_random, mirror_random;
    if (has_permutations) {
        auto [r_rest, r_val] = fb.pop(r);
        r = r_rest;
        chan_random = r_val;
    }
    if (has_multi_flavor) {
        auto [r_rest, r_val] = fb.pop(r);
        r = r_rest;
        flavor_random = r_val;
    }
    if (_has_mirror) {
        auto [r_rest, r_val] = fb.pop(r);
        r = r_rest;
        mirror_random = r_val;
    }

    // Apply adaptive map (VEGAS or MadNIS flow)
    Value latent = r;
    ValueVec mapping_conditions, flow_conditions;
    std::visit(
        Overloaded{
            [&](std::monostate) {},
            [&](const auto& admap) {
                auto admap_result = admap.build_forward(fb, {r}, {});
                latent = admap_result["data"];
                adaptive_probs.push_back(admap_result["det"]);
                if (_madnis_training) {
                    extra_weights_before_cuts.push_back(admap_result["det"]);
                } else {
                    weights_before_cuts.push_back(admap_result["det"]);
                }
                flow_conditions.push_back(latent);
            }
        },
        _adaptive_map
    );

    // Sample channel permutation
    Value chan_index, chan_index_in_group;
    if (has_permutations) {
        me_int_t opt_count = _channel_indices.size();
        std::visit(
            Overloaded{
                [&](std::monostate) {
                    auto [index, chan_det] = fb.sample_discrete(chan_random, opt_count);
                    chan_index_in_group = index;
                    weights_before_cuts.push_back(chan_det);
                },
                [&](const auto& discrete_sym) {
                    ValueVec discrete_condition;
                    using TDiscrete = std::decay_t<decltype(discrete_sym)>;
                    if constexpr (std::is_same_v<TDiscrete, DiscreteFlow>) {
                        if (flow_conditions.size() == 1) {
                            discrete_condition.push_back(flow_conditions.at(0));
                        } else if (flow_conditions.size() > 1) {
                            discrete_condition.push_back(fb.cat(flow_conditions));
                        }
                    }
                    auto discrete_result = discrete_sym.build_forward(
                        fb, {chan_random}, discrete_condition
                    );
                    chan_index_in_group = discrete_result.at(0);
                    if (_madnis_training) {
                        extra_weights_before_cuts.push_back(discrete_result["det"]);
                    } else {
                        weights_before_cuts.push_back(discrete_result["det"]);
                    }
                    adaptive_probs.push_back(discrete_result["det"]);
                    flow_conditions.push_back(
                        fb.one_hot(chan_index_in_group, opt_count)
                    );
                }
            },
            _discrete_sym
        );
        chan_index = fb.gather_int(chan_index_in_group, _channel_indices);
        mapping_conditions.push_back(chan_index_in_group);
    } else {
        chan_index =
            fb.full({static_cast<me_int_t>(_channel_indices.at(0)), batch_size_val});
        chan_index_in_group = fb.full({static_cast<me_int_t>(0), batch_size_val});
    }

    // Apply phase space mapping
    auto mapping_result = _mapping.build_forward(fb, {latent}, mapping_conditions);
    weights_before_cuts.push_back(mapping_result["det"]);
    Value momenta = mapping_result["momenta"];
    Value x0 = mapping_result["x1"];
    Value x1 = mapping_result["x2"];

    // Filter events that pass cuts
    Value weight_before_cuts = fb.product(weights_before_cuts);
    Value extra_weight_before_cuts;
    if (!extra_weights_before_cuts.empty()) {
        extra_weight_before_cuts = fb.product(extra_weights_before_cuts);
    }
    Value indices_acc = fb.nonzero(weight_before_cuts);
    Value momenta_acc = fb.batch_gather(indices_acc, momenta);
    std::array<Value, 2> x_acc{
        {fb.batch_gather(indices_acc, x0), fb.batch_gather(indices_acc, x1)}
    };
    for (auto& cond : flow_conditions) {
        cond = fb.batch_gather(indices_acc, cond);
    }

    // Evaluate PDF prior for adaptive flavor sampling
    Value pdf_prior;
    auto scales = _energy_scale.value().build_function(fb, {momenta_acc});
    std::array<Value, 2> pdf_results;
    bool has_pdf_prior = false;
    if ((_pdfs.at(0) || _pdfs.at(1)) && _energy_scale) {
        ValueVec pdf_priors;
        for (std::size_t i = 0; i < 2; ++i) {
            if (_diff_xs.at(0).has_pdf(i)) {
                // Under pdf reweighting the density is asked for low on the
                // clustering ladder rather than at the factorisation scale,
                // and walked back up by the ratios applied further down. This
                // is madevent setting q2fact below q2bck before calling DSIG,
                // and it is what the flavour prior should see too, since that
                // is the density the event is actually generated with.
                auto& pdf_scale = _energy_scale->mlm_pdf_reweighting()
                    ? scales.at(std::format("pdf_scale{}", i + 1))
                    : scales.at(i + 1);
                auto pdf = _pdfs.at(i)
                               .value()
                               .build_function(fb, {x_acc.at(i), pdf_scale})
                               .at(0);
                pdf_results.at(i) = pdf;
                pdf_priors.push_back(fb.select(pdf, _pdf_indices.at(i)));
            }
        }
        if (has_multi_flavor) {
            pdf_prior = fb.abs(fb.product(pdf_priors));
            if (_active_flavors_mask.size() > 0) {
                Value index = fb.batch_gather(indices_acc, chan_index_in_group);
                Value mask = fb.gather_vector(index, _active_flavors_mask);
                pdf_prior = fb.mul(pdf_prior, mask);
            }
            has_pdf_prior = true;
        }
    }

    // Flavor sampling
    auto batch_size_acc = fb.batch_size({momenta_acc});
    Value flavor_id = fb.full({static_cast<me_int_t>(0), batch_size_acc});
    Value extra_weight_after_cuts;
    if (has_multi_flavor) {
        auto flavor_random_acc = fb.batch_gather(indices_acc, flavor_random);
        std::visit(
            Overloaded{
                [&](std::monostate) {
                    if (has_pdf_prior) {
                        auto [index, flavor_det] =
                            fb.sample_discrete_probs(flavor_random_acc, pdf_prior);
                        flavor_id = index;
                        weights_after_cuts.push_back(flavor_det);
                    } else {
                        auto [index, flavor_det] = fb.sample_discrete(
                            flavor_random_acc,
                            static_cast<me_int_t>(_pid_options.size())
                        );
                        flavor_id = index;
                        weights_after_cuts.push_back(flavor_det);
                    }
                },
                [&](const auto& discrete_flavor) {
                    ValueVec discrete_condition;
                    using TDiscrete = std::decay_t<decltype(discrete_flavor)>;
                    if constexpr (std::is_same_v<TDiscrete, DiscreteFlow>) {
                        if (flow_conditions.size() == 1) {
                            discrete_condition.push_back(flow_conditions.at(0));
                        } else if (flow_conditions.size() > 1) {
                            discrete_condition.push_back(fb.cat(flow_conditions));
                        }
                    }
                    if (has_pdf_prior) {
                        discrete_condition.push_back(pdf_prior);
                    }
                    auto discrete_result = discrete_flavor.build_forward(
                        fb, {flavor_random_acc}, discrete_condition
                    );
                    flavor_id = discrete_result.at(0);
                    if (_madnis_training) {
                        extra_weight_after_cuts = discrete_result["det"];
                    } else {
                        weights_after_cuts.push_back(discrete_result["det"]);
                    }
                    auto ones = fb.full({1., batch_size_val});
                    adaptive_probs.push_back(
                        fb.batch_scatter(indices_acc, ones, discrete_result["det"])
                    );
                }
            },
            _discrete_flavor
        );
        for (auto [pdf, indices] : zip(pdf_results, _pdf_indices)) {
            if (pdf) {
                pdf = fb.gather(fb.gather_int(flavor_id, indices), pdf);
            }
        }
    } else {
        for (auto& pdf : pdf_results) {
            if (pdf) {
                pdf = fb.squeeze(pdf);
            }
        }
    }

    Value momenta_mirror_acc, mirror_id_acc;
    if (_has_mirror) {
        Value option_count;
        if (std::all_of(
                _flavor_mirror.begin(), _flavor_mirror.end(), [](me_int_t mirror) {
                    return mirror == 2;
                }
            )) {
            option_count = static_cast<me_int_t>(2);
        } else {
            option_count = fb.gather_int(flavor_id, _flavor_mirror);
        }
        Value mirror_random_acc = fb.batch_gather(indices_acc, mirror_random);
        auto [index, mirror_det] = fb.sample_discrete(mirror_random_acc, option_count);
        mirror_id_acc = index;
        momenta_mirror_acc = fb.mirror_momenta(momenta_acc, mirror_id_acc);
        weights_after_cuts.push_back(mirror_det);
    }

    // With a clustering history per diagram these scales only serve the pdf
    // prior above: the history, and every weight read off it, is settled after
    // the matrix element has given the diagram weights, in the common part.
    bool mlm_history_per_diagram =
        _energy_scale && _energy_scale->mlm_history_per_diagram();
    if (_energy_scale && _energy_scale->is_mlm() && !mlm_history_per_diagram) {
        for (auto& weight : mlm_weights(fb, scales, x_acc, flavor_id)) {
            weights_after_cuts.push_back(weight);
        }
    }
    if (_energy_scale && _energy_scale->has_scale_range() && !mlm_history_per_diagram) {
        // Same for the floor on the scales themselves, which applies to every
        // dynamical scale choice rather than only to the merging one.
        weights_after_cuts.push_back(scales.at("scale_weight"));
    }

    Value weight_after_cuts = weights_after_cuts.empty()
        ? fb.full({1., batch_size_acc})
        : fb.product(weights_after_cuts);
    Value adaptive_prob = adaptive_probs.empty()
        ? fb.full({1., batch_size_val})
        : fb.product(adaptive_probs);
    if (_drop_cuts_and_rescale) {
        adaptive_prob =
            fb.div(fb.accept_norm(indices_acc, adaptive_prob), adaptive_prob);
    }

    NamedVector<Value> out;

    // outputs before cuts
    out.push_back("r", r);
    out.push_back("latent", latent);
    out.push_back("weight_before_cuts", weight_before_cuts);
    out.push_back("adaptive_prob", adaptive_prob);
    out.push_back("chan_index", chan_index);
    out.push_back("chan_index_in_group", chan_index_in_group);
    if (!_madnis_training) {
        out.push_back("momenta", momenta);
    }
    if (_madnis_training) {
        out.push_back("extra_weight_before_cuts", extra_weight_before_cuts);
    }

    // outputs after cuts
    out.push_back("indices_acc", indices_acc);
    out.push_back("momenta_acc", momenta_acc);
    if (_has_mirror) {
        if (!_madnis_training) {
            out.push_back("momenta_mirror_acc", momenta_mirror_acc);
        }
        out.push_back("mirror_id_acc", mirror_id_acc);
    }
    out.push_back("x1_acc", x_acc.at(0));
    out.push_back("x2_acc", x_acc.at(1));
    out.push_back("flavor_id", flavor_id);
    out.push_back("weight_after_cuts", weight_after_cuts);
    if (_madnis_training && extra_weight_after_cuts) {
        out.push_back("extra_weight_after_cuts", extra_weight_after_cuts);
    }
    out.push_back("ren_scale", scales.at(0));
    if (has_pdf_prior) {
        out.push_back("pdf_prior", pdf_prior);
    }
    if (_diff_xs.at(0).has_pdf(0)) {
        out.push_back("pdf1", pdf_results.at(0));
        out.push_back("fact_scale1", scales.at(1));
    }
    if (_diff_xs.at(0).has_pdf(1)) {
        out.push_back("pdf2", pdf_results.at(1));
        out.push_back("fact_scale2", scales.at(2));
    }

    if (_energy_scale && _energy_scale->is_mlm()) {
        out.push_back("cluster_scales_acc", scales.at("outgoing_scales"));
        out.push_back("scale_diagram_index_acc", scales.at("diagram_index"));
    }

    return out;
}

ValueVec Integrand::mlm_weights(
    FunctionBuilder& fb,
    const NamedVector<Value>& scales,
    const std::array<Value, 2>& x,
    Value flavor_id
) const {
    ValueVec weights;
    auto batch_size_acc = fb.batch_size({flavor_id});
    // The merging cut comes out of the clustering, which only runs for
    // events that already passed the phase-space cuts, so it enters as a
    // factor on the weight rather than as one of the cuts themselves. An
    // event below xqcut ends up with weight zero and is never unweighted.
    weights.push_back(scales.at("xqcut_weight"));

    // alpha_s reweighting, the CKKW-style factor madevent applies in
    // Template/LO/SubProcesses/reweight.f: every clustering vertex that
    // produced a parton is evaluated at its own scale rather than at the
    // event's, so the weight carries prod_i alphas(pt_i) instead of
    // alphas(mu_R)^n. Without it a merged sample is short by one factor
    // per emission, compounding with multiplicity.
    //
    // The kernel hands back mu_R for any vertex it does not reweight, so
    // that vertex's ratio is one and no mask is needed here.
    if (_energy_scale->mlm_alphas_reweighting() && _running_coupling) {
        auto vertex_scales = scales.at("alphas_scales");
        std::size_t vertex_count = vertex_scales.type.shape.at(0);
        if (vertex_count > 0) {
            auto reference = _running_coupling.value()
                                 .build_function(fb, {scales.at("ren_scale")})
                                 .at(0);
            Value factor;
            for (std::size_t i = 0; i < vertex_count; ++i) {
                auto [rest, one_scale] = fb.pop(vertex_scales);
                vertex_scales = rest;
                auto alpha =
                    _running_coupling.value().build_function(fb, {one_scale}).at(0);
                auto ratio = fb.div(alpha, reference);
                factor = factor ? fb.mul(factor, ratio) : ratio;
            }
            weights.push_back(factor);
            weights.push_back(scales.at("alphas_weight"));
        }
    }

    // pdf reweighting: the beam density, evaluated above at the bottom of
    // the clustering ladder, walked back up one clustering at a time. Each
    // slot the kernel marked active contributes f(x, Q_i) / f(x, Q_i-1)
    // for the flavour its beam line carried at that point.
    //
    // The flavour is only knowable here, after the sampling: the kernel
    // hands out a class - the gluon, one of the two beams' own flavours, or
    // a flavour the diagram pinned down - and the table turns that plus the
    // sampled option into an index into the reweighting density.
    if (_pdf_rw && _pid_options.size() > 0) {
        auto flavor_classes = scales.at("pdf_rw_flavor");
        std::size_t slot_count = flavor_classes.type.shape.at(0);
        double low = _energy_scale->min_scale();
        double high = _energy_scale->max_scale() > 0.
            ? _energy_scale->max_scale()
            : 1e30;
        auto ones = fb.full({1., batch_size_acc});
        Value factor, veto;
        for (std::size_t i = 0; i < slot_count; ++i) {
            std::vector<me_int_t> column{static_cast<me_int_t>(i)};
            auto pick = [&](const char* name) {
                return fb.squeeze(fb.select(scales.at(name), column));
            };
            auto active = pick("pdf_rw_active");
            auto beam = pick("pdf_rw_beam");
            // x of the beam line: the beam's own momentum fraction times
            // everything the clustering z has taken off it since.
            auto x_beam = fb.add(
                x.at(0),
                fb.mul(beam, fb.sub(x.at(1), x.at(0)))
            );
            auto x = fb.mul(x_beam, pick("pdf_rw_x"));
            auto scale_of = [&](const char* name) {
                auto q = pick(name);
                if (low > 0.) {
                    q = fb.max(q, fb.full({low, batch_size_acc}));
                }
                return fb.min(q, fb.full({high, batch_size_acc}));
            };
            auto index = fb.gather_int(
                fb.add_int(
                    fb.gather_int(
                        fb.squeeze(fb.select_int(flavor_classes, column)),
                        _pdf_rw_class_offsets
                    ),
                    flavor_id
                ),
                _pdf_rw_table
            );
            auto density = [&](const char* name) {
                return _pdf_rw.value()
                    .build_function(fb, {x, scale_of(name), index})
                    .at(0);
            };
            auto numerator = density("pdf_rw_q_num");
            auto denominator = density("pdf_rw_q_den");
            // madevent drops the event outright when the density it is
            // dividing by falls under 1e-10, where the grid is no longer
            // saying anything. Same threshold, as a ramp rather than a
            // branch, and the floor under the division keeps the term
            // finite so that a vetoed slot multiplies to zero and not to
            // a NaN.
            auto floor = fb.full({1e-10, batch_size_acc});
            auto usable = fb.min(
                fb.max(
                    fb.mul(denominator, fb.full({1e10, batch_size_acc})),
                    fb.full({0., batch_size_acc})
                ),
                ones
            );
            auto ratio = fb.div(numerator, fb.max(denominator, floor));
            // An inert slot contributes exactly one, whatever its density
            // came out as.
            auto term = fb.add(ones, fb.mul(active, fb.sub(ratio, ones)));
            auto pass = fb.add(ones, fb.mul(active, fb.sub(usable, ones)));
            factor = factor ? fb.mul(factor, term) : term;
            veto = veto ? fb.mul(veto, pass) : pass;
        }
        if (factor) {
            weights.push_back(factor);
            weights.push_back(veto);
        }
    }
    return weights;
}

std::array<Value, 2> Integrand::evaluate_pdfs(
    FunctionBuilder& fb,
    const NamedVector<Value>& scales,
    const std::array<Value, 2>& x,
    Value flavor_id
) const {
    std::array<Value, 2> pdfs;
    for (std::size_t i = 0; i < 2; ++i) {
        if (!_diff_xs.at(0).has_pdf(i)) {
            continue;
        }
        auto& pdf_scale = _energy_scale->mlm_pdf_reweighting()
            ? scales.at(std::format("pdf_scale{}", i + 1))
            : scales.at(std::format("fact_scale{}", i + 1));
        auto pdf = _pdfs.at(i).value().build_function(fb, {x.at(i), pdf_scale}).at(0);
        pdfs.at(i) = _pid_options.size() > 1
            ? fb.gather(fb.gather_int(flavor_id, _pdf_indices.at(i)), pdf)
            : fb.squeeze(pdf);
    }
    return pdfs;
}

NamedVector<Value> Integrand::build_common_part(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    bool has_multi_flavor = _pid_options.size() > 1;
    bool has_permutations = _mapping.channel_count() > 1;
    bool has_pdf_prior =
        (_pdfs.at(0) || _pdfs.at(1)) && _energy_scale && has_multi_flavor;

    auto indices_acc = args.at("indices_acc");
    auto momenta_acc = args.at("momenta_acc");
    auto x1_acc = args.at("x1_acc");
    auto x2_acc = args.at("x2_acc");
    auto flavor_id = args.at("flavor_id");
    auto batch_size_val = fb.batch_size({args.at("weight_before_cuts")});

    auto scatter_or_drop = [&](Value default_value, Value value) -> Value {
        if (_drop_cuts_and_rescale) {
            return value;
        }
        return fb.batch_scatter(indices_acc, default_value, value);
    };
    auto optional_cut = [&](Value value) -> Value {
        if (_drop_cuts_and_rescale) {
            return fb.batch_gather(indices_acc, value);
        }
        return value;
    };

    // Channel weight computation
    std::size_t channel_count = final_channel_count(
        _diff_xs,
        _first_chan_weight_remap,
        _first_remapped_chan_count,
        _second_chan_weight_remap,
        _second_remapped_chan_count,
        _prop_chan_weights,
        _subchan_weights
    );
    bool use_compressed_channel_weights =
        channel_count > 1 && channel_count * 8 > _compressed_channel_weight_count * 12;
    Value chan_weights_acc;
    if (channel_count > 1 && _prop_chan_weights) {
        chan_weights_acc = _prop_chan_weights->build_function(fb, {momenta_acc}).at(0);
        if (_first_chan_weight_remap.size() > 0) {
            chan_weights_acc = fb.collect_channel_weights(
                chan_weights_acc,
                _first_chan_weight_remap.at(0),
                _first_remapped_chan_count
            );
        }
    }

    // Compute running coupling
    Value alpha_qcd_acc =
        _running_coupling.value().build_function(fb, {args.at("ren_scale")}).at(0);

    // The scales and densities the event is evaluated with. They come from the
    // channel part, unless the MLM history is picked per diagram further down.
    Value ren_scale_acc = args.at("ren_scale");
    std::array<Value, 2> fact_scales_acc, pdfs_acc;
    for (std::size_t i = 0; i < 2; ++i) {
        if (_diff_xs.at(0).has_pdf(i)) {
            fact_scales_acc.at(i) = args.at(std::format("fact_scale{}", i + 1));
            pdfs_acc.at(i) = args.at(std::format("pdf{}", i + 1));
        }
    }
    Value cluster_scales_acc;
    if (_energy_scale && _energy_scale->is_mlm()) {
        cluster_scales_acc = args.at("cluster_scales_acc");
    }

    // Evaluate differential cross section
    auto make_xs_args = [&](Value diagram, std::array<Value, 2>& pdfs, Value alpha) {
        ValueVec xs_args{
            momenta_acc,
            _flavor_remap.size() > 0 ? fb.gather_int(flavor_id, _flavor_remap)
                                     : flavor_id,
        };
        if (_energy_scale && _energy_scale->is_mlm()) {
            xs_args.push_back(diagram);
        }
        xs_args.push_back(x1_acc);
        xs_args.push_back(x2_acc);
        xs_args.push_back(flavor_id);
        for (std::size_t i = 0; i < 2; ++i) {
            if (_diff_xs.at(0).has_pdf(i)) {
                xs_args.push_back(pdfs.at(i));
            }
        }
        xs_args.push_back(alpha);
        return xs_args;
    };
    ValueVec xs_args = make_xs_args(
        _energy_scale && _energy_scale->is_mlm() ? args.at("scale_diagram_index_acc")
                                                 : Value(),
        pdfs_acc,
        alpha_qcd_acc
    );
    ValueVec dxs_vec;
    Value ps_flavor_id;
    Value subproc_id;
    if (_diff_xs.size() == 1) {
        dxs_vec = _diff_xs.at(0).build_function(fb, xs_args).values();
        ps_flavor_id = flavor_id;
        if (channel_count > 1 && !_prop_chan_weights) {
            chan_weights_acc = dxs_vec.at(1);
            if (_first_chan_weight_remap.size() > 0) {
                chan_weights_acc = fb.collect_channel_weights(
                    chan_weights_acc,
                    _first_chan_weight_remap.at(0),
                    _first_remapped_chan_count
                );
            }
        }
    } else {
        Value dxs_index = fb.gather_int(flavor_id, _flavor_diff_xs_indices);
        ps_flavor_id = fb.gather_int(flavor_id, _flavor_per_subproc_remap);
        subproc_id = fb.gather_int(flavor_id, _flavor_subproc_indices);
        ValueVec split_indices =
            fb.batch_split_by_index(dxs_index, static_cast<me_int_t>(_diff_xs.size()));
        std::vector<ValueVec> split_outputs(_diff_xs.at(0).return_types().size());
        ValueVec split_channel_weights;
        for (std::size_t i = 0;
             auto [diff_xs, indices] : zip(_diff_xs, split_indices)) {
            ValueVec split_args;
            for (Value& arg : xs_args) {
                split_args.push_back(fb.batch_gather(indices, arg));
            }
            ValueVec outputs = diff_xs.build_function(fb, split_args).values();
            for (auto [out, split_out] : zip(outputs, split_outputs)) {
                split_out.push_back(out);
                split_out.push_back(indices);
            }
            if (channel_count > 1 && !_prop_chan_weights) {
                Value split_cw = outputs.at(1);
                if (_first_chan_weight_remap.size() > 0) {
                    split_cw = fb.collect_channel_weights(
                        split_cw,
                        _first_chan_weight_remap.at(i),
                        _first_remapped_chan_count
                    );
                }
                split_channel_weights.push_back(split_cw);
                split_channel_weights.push_back(indices);
            }
            ++i;
        }
        for (std::size_t i = 0; auto& split_out : split_outputs) {
            if (i == 1) {
                dxs_vec.push_back(Value());
            } else {
                dxs_vec.push_back(fb.batch_merge_by_index(split_out));
            }
            ++i;
        }
        if (channel_count > 1 && !_prop_chan_weights) {
            chan_weights_acc = fb.batch_merge_by_index(split_channel_weights);
        }
    }

    // A clustering history per diagram. The evaluation above, at the scales of
    // the history over every diagram, has supplied the diagram weights
    // |A_i|^2 / sum_j |A_j|^2 (and the channel weights, which stay as they
    // are). One diagram is picked from them - a choice between histories, not
    // an importance sample, so it puts no factor on the weight - the event is
    // clustered along it, and the matrix element is evaluated again at the
    // scales that history gives. Evaluating it again rather than rescaling by
    // alpha_s ratios keeps this exact when coupling orders mix, as they do for
    // gluon fusion and VBF in one process.
    ValueVec mlm_history_weights;
    if (_energy_scale && _energy_scale->mlm_history_per_diagram()) {
        auto batch_size_acc = fb.batch_size({momenta_acc});
        auto diagram_mask = fb.gather_vector(
            fb.full({static_cast<me_int_t>(0), batch_size_acc}), _mlm_diagram_mask
        );
        auto [diagram, diagram_det] = fb.sample_discrete_probs(
            fb.squeeze(fb.random(batch_size_acc, 1)),
            fb.mul(dxs_vec.at(1), diagram_mask)
        );
        auto scales = _energy_scale->build_mlm_from_start_state(
            fb, momenta_acc, fb.gather_int(diagram, _mlm_start_states)
        );
        std::array<Value, 2> x_acc{x1_acc, x2_acc};
        mlm_history_weights = mlm_weights(fb, scales, x_acc, flavor_id);
        if (_energy_scale->has_scale_range()) {
            mlm_history_weights.push_back(scales.at("scale_weight"));
        }
        ren_scale_acc = scales.at("ren_scale");
        for (std::size_t i = 0; i < 2; ++i) {
            if (_diff_xs.at(0).has_pdf(i)) {
                fact_scales_acc.at(i) = scales.at(std::format("fact_scale{}", i + 1));
            }
        }
        pdfs_acc = evaluate_pdfs(fb, scales, x_acc, flavor_id);
        cluster_scales_acc = scales.at("outgoing_scales");
        alpha_qcd_acc =
            _running_coupling.value().build_function(fb, {ren_scale_acc}).at(0);
        dxs_vec = _diff_xs.at(0)
                      .build_function(
                          fb,
                          make_xs_args(scales.at("diagram_index"), pdfs_acc, alpha_qcd_acc)
                      )
                      .values();
    }

    auto diff_xs_acc = dxs_vec.at(0);
    if (_flavor_factors.size() > 0) {
        diff_xs_acc = fb.mul(diff_xs_acc, fb.gather(flavor_id, _flavor_factors));
    }
    ValueVec weights_after_cuts{args.at("weight_after_cuts"), diff_xs_acc};
    for (auto& weight : mlm_history_weights) {
        weights_after_cuts.push_back(weight);
    }
    ValueVec extra_weights_after_cuts;
    if (args.index_map().contains("extra_weight_after_cuts")) {
        extra_weights_after_cuts.push_back(args.at("extra_weight_after_cuts"));
    }
    if (channel_count > 1 && _subchan_weights) {
        chan_weights_acc =
            _subchan_weights->build_function(fb, {momenta_acc, chan_weights_acc}).at(0);
    }
    if (_second_chan_weight_remap.size() > 0) {
        chan_weights_acc = fb.collect_channel_weights(
            chan_weights_acc, _second_chan_weight_remap, _second_remapped_chan_count
        );
    }

    // Apply channel weight network
    auto prior_chan_weights_acc = chan_weights_acc;
    if (channel_count > 1 && _chan_weight_net) {
        auto& preproc = _chan_weight_net.value().preprocessing();
        auto cw_preproc_acc =
            preproc.build_function(fb, {momenta_acc, x1_acc, x2_acc}).at(0);
        chan_weights_acc =
            _chan_weight_net.value()
                .build_function(fb, {cw_preproc_acc, chan_weights_acc})
                .at(0);
    }

    // Compute full phase-space weight
    if (channel_count > 1 && !_madnis_training) {
        Value chan_index_acc = fb.batch_gather(indices_acc, args.at("chan_index"));
        weights_after_cuts.push_back(fb.gather(chan_index_acc, chan_weights_acc));
    }
    auto weight = fb.mul(
        args.at("weight_before_cuts"),
        fb.batch_scatter(
            indices_acc, args.at("weight_before_cuts"), fb.product(weights_after_cuts)
        )
    );

    NamedVector<Value> outputs;
    if (_madnis_training) {
        Value full_weight = weight;
        if (!extra_weights_after_cuts.empty()) {
            full_weight = fb.mul(
                weight,
                fb.batch_scatter(
                    indices_acc, full_weight, fb.product(extra_weights_after_cuts)
                )
            );
        }
        if (args.index_map().contains("extra_weight_before_cuts")) {
            full_weight = fb.mul(full_weight, args.at("extra_weight_before_cuts"));
        }
        outputs.push_back("full_weight", optional_cut(full_weight));
        outputs.push_back("weight", optional_cut(weight));
        outputs.push_back("latent", optional_cut(args.at("latent")));
        outputs.push_back("adaptive_prob", optional_cut(args.at("adaptive_prob")));
        outputs.push_back("channel_index", optional_cut(args.at("chan_index")));
        if (channel_count > 1) {
            if (use_compressed_channel_weights) {
                Value chan_index_acc =
                    fb.batch_gather(indices_acc, args.at("chan_index"));
                auto [chan_weight_values_acc, chan_weight_indices_acc] =
                    fb.compress_channel_weights(
                        chan_index_acc,
                        prior_chan_weights_acc,
                        static_cast<me_int_t>(_compressed_channel_weight_count)
                    );
                auto cw_values_default = fb.full(
                    {0.,
                     batch_size_val,
                     static_cast<me_int_t>(_compressed_channel_weight_count)}
                );
                auto cw_indices_default = fb.full(
                    {static_cast<me_int_t>(-1),
                     batch_size_val,
                     static_cast<me_int_t>(_compressed_channel_weight_count)}
                );
                outputs.push_back(
                    "channel_weight_values",
                    scatter_or_drop(cw_values_default, chan_weight_values_acc)
                );
                outputs.push_back(
                    "channel_weight_indices",
                    scatter_or_drop(cw_indices_default, chan_weight_indices_acc)
                );
            } else {
                auto cw_flat = fb.full(
                    {1. / channel_count,
                     batch_size_val,
                     static_cast<me_int_t>(channel_count)}
                );
                outputs.push_back(
                    "channel_weights", scatter_or_drop(cw_flat, prior_chan_weights_acc)
                );
            }
        } else {
            outputs.push_back(
                "channel_weights",
                fb.full(
                    {1. / channel_count,
                     fb.batch_size({outputs.at(0)}),
                     static_cast<me_int_t>(channel_count)}
                )
            );
        }
        auto& preproc = _chan_weight_net.value().preprocessing();
        auto cw_preproc_acc =
            preproc.build_function(fb, {momenta_acc, x1_acc, x2_acc}).at(0);
        outputs.push_back(
            "cwnet_input",
            scatter_or_drop(
                fb.full(
                    {0., batch_size_val, static_cast<me_int_t>(preproc.output_dim())}
                ),
                cw_preproc_acc
            )
        );
        outputs.push_back(
            "channel_index_in_group", optional_cut(args.at("chan_index_in_group"))
        );
        if (has_multi_flavor &&
            !std::holds_alternative<std::monostate>(_discrete_flavor)) {
            auto zeros = fb.full({static_cast<me_int_t>(0), batch_size_val});
            outputs.push_back(
                "discrete_flavor_index", scatter_or_drop(zeros, flavor_id)
            );
            if (has_pdf_prior) {
                auto flav_count = static_cast<me_int_t>(_pid_options.size());
                outputs.push_back(
                    "pdf_prior",
                    scatter_or_drop(
                        fb.full({1. / flav_count, batch_size_val, flav_count}),
                        args.at("pdf_prior")
                    )
                );
            }
        }
    } else {
        outputs.push_back("weight", optional_cut(weight));
        if (_has_mirror) {
            outputs.push_back(
                "momenta",
                scatter_or_drop(args.at("momenta"), args.at("momenta_mirror_acc"))
            );
        } else {
            outputs.push_back("momenta", args.at("momenta"));
        }
        auto zeros_int = fb.full({static_cast<me_int_t>(0), batch_size_val});
        auto zeros_float = fb.full({0., batch_size_val});
        outputs.push_back("color_index", scatter_or_drop(zeros_int, dxs_vec.at(2)));
        outputs.push_back("helicity_index", scatter_or_drop(zeros_int, dxs_vec.at(3)));
        outputs.push_back("diagram_index", scatter_or_drop(zeros_int, dxs_vec.at(4)));
        outputs.push_back("flavor_index", scatter_or_drop(zeros_int, ps_flavor_id));
        if (subproc_id) {
            outputs.push_back(
                "subprocess_index", scatter_or_drop(zeros_int, subproc_id)
            );
        }

        outputs.push_back(
            "ren_scale", scatter_or_drop(zeros_float, ren_scale_acc)
        );
        outputs.push_back("alpha_qcd", scatter_or_drop(zeros_float, alpha_qcd_acc));
        ValueVec pdf_vals;
        if (_diff_xs.at(0).has_pdf(0)) {
            outputs.push_back("x1", scatter_or_drop(zeros_float, args.at("x1_acc")));
            outputs.push_back(
                "fact_scale1", scatter_or_drop(zeros_float, fact_scales_acc.at(0))
            );
            pdf_vals.push_back(pdfs_acc.at(0));
        }
        if (_diff_xs.at(0).has_pdf(1)) {
            outputs.push_back("x2", scatter_or_drop(zeros_float, args.at("x2_acc")));
            outputs.push_back(
                "fact_scale2", scatter_or_drop(zeros_float, fact_scales_acc.at(1))
            );
            pdf_vals.push_back(pdfs_acc.at(1));
        }
        if (_partial_weights &&
            (_diff_xs.at(0).has_pdf(0) || _diff_xs.at(0).has_pdf(1))) {
            outputs.push_back(
                "partial_weight_product",
                scatter_or_drop(zeros_float, fb.product(pdf_vals))
            );
        }
        if (_energy_scale && _energy_scale->is_mlm()) {
            auto outgoing_count = static_cast<me_int_t>(_mapping.particle_count() - 2);
            outputs.push_back(
                "cluster_scales",
                scatter_or_drop(
                    fb.full({0., batch_size_val, outgoing_count}),
                    cluster_scales_acc
                )
            );
        }
        outputs.push_back("random", optional_cut(args.at("r")));
        if (has_permutations &&
            !std::holds_alternative<std::monostate>(_discrete_sym)) {
            outputs.push_back(
                "channel_index_in_group", optional_cut(args.at("chan_index_in_group"))
            );
        }
        if (has_multi_flavor &&
            !std::holds_alternative<std::monostate>(_discrete_flavor)) {
            outputs.push_back(
                "discrete_flavor_index", scatter_or_drop(zeros_int, flavor_id)
            );
        }
    }

    return outputs;
}

IntegrandChannelPart::IntegrandChannelPart(const Integrand& integrand) :
    FunctionGenerator(
        "IntegrandChannelPart",
        {{"batch_size", Type({batch_size})}},
        integrand._channel_part_ret_types
    ),
    _integrand(integrand) {}

NamedVector<Value> IntegrandChannelPart::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    return _integrand.build_channel_part(fb, args);
}

IntegrandCommonPart::IntegrandCommonPart(const Integrand& integrand) :
    FunctionGenerator(
        "IntegrandCommonPart",
        integrand._channel_part_ret_types,
        integrand.return_types()
    ),
    _integrand(integrand) {}

NamedVector<Value> IntegrandCommonPart::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    return _integrand.build_common_part(fb, args);
}

IntegrandConcatenator::IntegrandConcatenator(const Integrand& integrand) :
    FunctionGenerator(
        "IntegrandConcatenator",
        [&] {
            NamedVector<Type> arg_types;
            arg_types.reserve(2 * integrand._channel_part_ret_types.size());
            auto keys = integrand._channel_part_ret_types.keys();
            for (auto [key, type] : zip(keys, integrand._channel_part_ret_types)) {
                arg_types.push_back(std::format("arg1_{}", key), type);
            }
            for (auto [key, type] : zip(keys, integrand._channel_part_ret_types)) {
                arg_types.push_back(std::format("arg2_{}", key), type);
            }
            return arg_types;
        }(),
        integrand._channel_part_ret_types
    ),
    _integrand(integrand) {}

NamedVector<Value> IntegrandConcatenator::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    // Combine per-channel results into a single NamedVector
    ValueVec outputs;
    auto keys = return_types().keys();
    std::size_t half_count = args.size() / 2;
    Value batch_sizes = fb.stack_sizes(
        {fb.batch_size({args.at("arg1_momenta")}),
         fb.batch_size({args.at("arg2_momenta")})}
    );
    for (auto [key, val1, val2] :
         zip(std::span(keys.begin(), keys.begin() + half_count),
             std::span(args.begin(), args.begin() + half_count),
             std::span(args.begin() + half_count, args.end()))) {
        auto [cat, cat_sizes] = fb.batch_cat({val1, val2});
        if (key == "indices_acc" && !_integrand._drop_cuts_and_rescale) {
            outputs.push_back(
                fb.add_int(fb.offset_indices(batch_sizes, cat_sizes), cat)
            );
        } else {
            outputs.push_back(cat);
        }
    }
    return {keys, outputs};
}

MultiChannelIntegrand::MultiChannelIntegrand(
    const std::vector<std::shared_ptr<Integrand>>& integrands, bool return_sizes
) :
    FunctionGenerator(
        "MultiChannelIntegrand",
        {{"batch_sizes", multichannel_batch_size(integrands.size())}},
        [&] {
            NamedVector<Type> ret_types = integrands.at(0)->return_types();
            if (return_sizes) {
                ret_types.push_back(
                    "return_batch_sizes", multichannel_batch_size(integrands.size())
                );
            }
            return ret_types;
        }()
    ),
    _integrands(integrands),
    _return_sizes(return_sizes) {
    auto& first_function = integrands.at(0);
    std::size_t arg_count = first_function->arg_types().size();
    std::size_t return_count = first_function->return_types().size();
    for (auto& integrand : integrands) {
        if (integrand->arg_types().size() != arg_count ||
            integrand->return_types().size() != return_count) {
            throw std::invalid_argument(
                "All integrands must have the same number of inputs and outputs"
            );
        }
    }
}

NamedVector<Value> MultiChannelIntegrand::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    auto batch_sizes = args.at(0);
    auto all_batch_sizes = fb.unstack_sizes(batch_sizes);

    std::vector<NamedVector<Value>> results;
    ValueVec ret_batch_sizes;

    for (std::size_t index = 0;
         auto [integrand, chan_size] : zip(_integrands, all_batch_sizes)) {
        fb.set_current_stream(index + 1);
        results.push_back(
            integrand->build_channel_part(fb, {{"batch_size", chan_size}})
        );
        if (_return_sizes) {
            ret_batch_sizes.push_back(
                fb.batch_size({results.back().at("indices_acc")})
            );
        }
        ++index;
    }
    fb.set_current_stream(0);

    // Combine per-channel results into a single NamedVector
    NamedVector<Value> common_results;
    for (const auto& key : results.at(0).keys()) {
        ValueVec values;
        for (auto& result : results) {
            values.push_back(result.at(key));
        }
        auto [cat, cat_sizes] = fb.batch_cat(values);
        if (key == "indices_acc") {
            common_results.push_back(
                key, fb.add_int(fb.offset_indices(batch_sizes, cat_sizes), cat)
            );
        } else {
            common_results.push_back(key, cat);
        }
    }

    auto output = _integrands.at(0)->build_common_part(fb, common_results);
    if (_return_sizes) {
        output.push_back(
            "return_batch_sizes",
            _integrands.at(0)->_drop_cuts_and_rescale
                ? fb.stack_sizes(ret_batch_sizes)
                : batch_sizes
        );
    }
    return output;
}

IntegrandProbability::IntegrandProbability(const Integrand& integrand) :
    FunctionGenerator(
        "IntegrandProbability",
        [&] {
            NamedVector<Type> arg_types{
                {"latent", batch_float_array(integrand._mapping.random_dim())},
                {"channel_index_in_group", batch_int}
            };
            auto flavor_count = integrand._pid_options.size();
            if (flavor_count > 1 &&
                !std::holds_alternative<std::monostate>(integrand._discrete_flavor)) {
                arg_types.push_back("discrete_flavor_index", batch_int);
                if ((integrand._pdfs.at(0) || integrand._pdfs.at(1)) &&
                    integrand._energy_scale) {
                    arg_types.push_back("pdf_prior", batch_float_array(flavor_count));
                }
            }
            return arg_types;
        }(),
        {{"prob", batch_float}}
    ),
    _adaptive_map(integrand._adaptive_map),
    _discrete_sym(integrand._discrete_sym),
    _discrete_flavor(integrand._discrete_flavor),
    _permutation_count(integrand._mapping.channel_count()),
    _flavor_count(integrand._pid_options.size()),
    _has_pdf_prior(
        (integrand._pdfs.at(0) || integrand._pdfs.at(1)) && integrand._energy_scale
    ) {}

NamedVector<Value> IntegrandProbability::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    ValueVec probs, flow_conditions;

    auto latent = args.at(0);
    std::visit(
        Overloaded{
            [&](std::monostate) {},
            [&](const auto& admap) {
                ValueVec cond;
                auto admap_result = admap.build_inverse(fb, {latent}, {});
                probs.push_back(admap_result["det"]);
                flow_conditions.push_back(latent);
            }
        },
        _adaptive_map
    );

    if (_permutation_count > 1) {
        auto chan_index = args.at(1);
        std::visit(
            Overloaded{
                [](std::monostate) {},
                [&](const auto& discrete_sym) {
                    ValueVec discrete_condition;
                    using TDiscrete = std::decay_t<decltype(discrete_sym)>;
                    if constexpr (std::is_same_v<TDiscrete, DiscreteFlow>) {
                        if (flow_conditions.size() == 1) {
                            discrete_condition.push_back(flow_conditions.at(0));
                        } else if (flow_conditions.size() > 1) {
                            discrete_condition.push_back(fb.cat(flow_conditions));
                        }
                    }
                    auto discrete_result = discrete_sym.build_inverse(
                        fb, {chan_index}, discrete_condition
                    );
                    probs.push_back(discrete_result["det"]);
                    flow_conditions.push_back(fb.one_hot(
                        chan_index, static_cast<me_int_t>(_permutation_count)
                    ));
                }
            },
            _discrete_sym
        );
    }

    std::size_t arg_index = 2;
    if (_flavor_count > 1) {
        std::visit(
            Overloaded{
                [&](std::monostate) {},
                [&](const auto& discrete_flavor) {
                    auto flavor = args.at(arg_index);
                    ++arg_index;
                    ValueVec discrete_condition;
                    using TDiscrete = std::decay_t<decltype(discrete_flavor)>;
                    if constexpr (std::is_same_v<TDiscrete, DiscreteFlow>) {
                        if (flow_conditions.size() == 1) {
                            discrete_condition.push_back(flow_conditions.at(0));
                        } else if (flow_conditions.size() > 1) {
                            discrete_condition.push_back(fb.cat(flow_conditions));
                        }
                    }
                    if (_has_pdf_prior) {
                        auto pdf_prior = args.at(arg_index);
                        ++arg_index;
                        discrete_condition.push_back(pdf_prior);
                    }
                    auto discrete_result =
                        discrete_flavor.build_inverse(fb, {flavor}, discrete_condition);
                    probs.push_back(discrete_result["det"]);
                }
            },
            _discrete_flavor
        );
    }

    return {{"prob", fb.product(probs)}};
}

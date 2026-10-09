#include "madspace/driver/event_utils.hpp"

#include <cmath>
#include <format>
#include <stdexcept>

using namespace madspace;

DataLayout
madspace::combined_event_layout(int extra_event_flags, int extra_particle_flags) {
    return DataLayout(
        EventRecord::layout(
            EventRecord::f_weight | EventRecord::f_subproc_index |
            EventRecord::f_event_data | extra_event_flags
        ),
        ParticleRecord::layout(ParticleRecord::f_particle_data | extra_particle_flags)
    );
}

void madspace::copy_channel_event(
    EventBuffer& in_buffer,
    std::size_t in_index,
    EventBuffer& out_buffer,
    std::size_t out_index,
    double weight,
    int subprocess_index,
    int extra_flags
) {
    auto event_in = in_buffer.event(in_index);
    auto event_out = out_buffer.event(out_index);
    event_out.weight() = weight;
    event_out.subprocess_index() = extra_flags & EventRecord::f_subproc_index
        ? event_in.subprocess_index().value()
        : subprocess_index;
    event_out.diagram_index() = event_in.diagram_index();
    event_out.color_index() = event_in.color_index();
    event_out.flavor_index() = event_in.flavor_index();
    event_out.helicity_index() = event_in.helicity_index();
    event_out.ren_scale() = event_in.ren_scale();
    event_out.alpha_qcd() = event_in.alpha_qcd();

    if (extra_flags & EventRecord::f_beam1) {
        event_out.x1() = event_in.x1();
        event_out.fact_scale1() = event_in.fact_scale1();
    }
    if (extra_flags & EventRecord::f_beam2) {
        event_out.x2() = event_in.x2();
        event_out.fact_scale2() = event_in.fact_scale2();
    }
    if (extra_flags & EventRecord::f_partial_weights) {
        event_out.partial_weight_product() = event_in.partial_weight_product();
    }

    std::size_t i = 0;
    for (; i < in_buffer.particle_count(); ++i) {
        auto particle_in = in_buffer.particle(in_index, i);
        auto particle_out = out_buffer.particle(out_index, i);
        particle_out.energy() = particle_in.energy();
        particle_out.px() = particle_in.px();
        particle_out.py() = particle_in.py();
        particle_out.pz() = particle_in.pz();
    }
    for (; i < out_buffer.particle_count(); ++i) {
        auto particle_out = out_buffer.particle(out_index, i);
        particle_out.energy() = 0.;
        particle_out.px() = 0.;
        particle_out.py() = 0.;
        particle_out.pz() = 0.;
    }
}

void madspace::fill_lhe_event(
    LHECompleter& lhe_completer,
    LHEEvent& lhe_event,
    EventBuffer& buffer,
    std::size_t event_index,
    MixMaxRandom& rand_gen
) {
    EventRecord event_in = buffer.event(event_index);
    lhe_event.weight = event_in.weight();
    lhe_event.process_id = 0;
    lhe_event.scale = event_in.ren_scale();
    lhe_event.alpha_qed = 0; // TODO: populate this
    lhe_event.alpha_qcd = event_in.alpha_qcd();
    lhe_event.particles.clear();
    for (std::size_t i = 0; i < buffer.particle_count(); ++i) {
        auto particle_in = buffer.particle(event_index, i);
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

void madspace::write_lhe_particles(
    const LHEEvent& lhe_event, EventBuffer& buffer, std::size_t event_index
) {
    std::size_t i = 0;
    for (; i < lhe_event.particles.size(); ++i) {
        buffer.particle(event_index, i).from_lhe_particle(lhe_event.particles[i]);
    }
    for (; i < buffer.particle_count(); ++i) {
        buffer.particle(event_index, i).from_lhe_particle(LHEParticle{});
    }
}

void madspace::sum_channel_status(
    const std::vector<std::shared_ptr<ChannelEventGenerator>>& channels,
    GeneratorStatus& status
) {
    double total_mean = 0., total_var = 0.;
    double total_mean_abs = 0., total_var_abs = 0.;
    std::size_t total_count = 0, total_count_opt = 0;
    std::size_t total_count_after_cuts = 0, total_count_after_cuts_opt = 0;
    std::size_t total_integ_count = 0;
    std::size_t iterations = 0;
    bool optimized = true;
    const ChannelEventGenerator* bad_channel = nullptr;
    for (auto& channel : channels) {
        auto& chan_status = channel->status();
        auto& cross_section = channel->cross_section();
        auto& abs_cross_section = channel->abs_cross_section();
        // special case for channels with 0 samples, as they have nan variance
        if (bad_channel == nullptr && cross_section.count() > 0 &&
            (!std::isfinite(cross_section.mean()) ||
             !std::isfinite(cross_section.variance()) ||
             !std::isfinite(abs_cross_section.mean()) ||
             !std::isfinite(abs_cross_section.variance()))) {
            bad_channel = channel.get();
        }
        total_mean += cross_section.mean();
        total_mean_abs += abs_cross_section.mean();
        if (cross_section.count() > 0) {
            total_var += cross_section.variance() / cross_section.count();
            total_var_abs += abs_cross_section.variance() / abs_cross_section.count();
        }
        total_count += chan_status.count;
        total_count_opt += chan_status.count_opt;
        total_count_after_cuts += chan_status.count_after_cuts;
        total_count_after_cuts_opt += chan_status.count_after_cuts_opt;
        total_integ_count += cross_section.count();
        iterations = std::max(chan_status.iterations, iterations);
        if (!chan_status.optimized) {
            optimized = false;
        }
    }
    if (bad_channel != nullptr || !std::isfinite(total_mean) ||
        !std::isfinite(total_mean_abs)) {
        std::string where = bad_channel != nullptr
            ? std::format(
                  "channel '{}' after {} samples (mean={}, variance={})",
                  bad_channel->status().name,
                  bad_channel->cross_section().count(),
                  bad_channel->cross_section().mean(),
                  bad_channel->cross_section().variance()
              )
            : std::format("the sum over channels (mean={})", total_mean);
        throw std::runtime_error(
            std::format(
                "non-finite integral in {}. A non-finite weight cannot be recovered "
                "from by sampling further, so the integration is aborted. This usually "
                "means the matrix element or the phase-space mapping returned nan/inf "
                "for "
                "some phase-space point -- check the process for a zero or negative "
                "width, "
                "a parameter point outside the model's validity, or a kinematic "
                "configuration at a threshold.",
                where
            )
        );
    }

    status.mean = total_mean;
    status.error = std::sqrt(total_var);
    status.mean_abs = total_mean_abs;
    status.error_abs = std::sqrt(total_var_abs);
    status.rel_std_dev = std::sqrt(total_var_abs * total_integ_count) / total_mean_abs;
    status.count = total_count;
    status.count_opt = total_count_opt;
    status.count_after_cuts = total_count_after_cuts;
    status.count_after_cuts_opt = total_count_after_cuts_opt;
    status.iterations = iterations;
    status.optimized = optimized;
}

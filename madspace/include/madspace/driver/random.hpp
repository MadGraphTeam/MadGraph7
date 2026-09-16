#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>

#include "madspace/mixmax/mixmax.hpp"

namespace madspace {

struct DerivedSeed {
    static constexpr std::size_t max_channel_count = 1ULL << 12;
    static constexpr std::size_t max_job_count = 1ULL << 32;
    static constexpr std::size_t max_stream_count = 1ULL << 16;
    enum SeedType {
        none,
        first_survey_generate,
        first_survey_unweight,
        second_survey_generate,
        second_survey_unweight,
        generator_generate,
        generator_unweight,
        combine_select,
        lhe_complete,
        unweight_pass,
        madnis_generate,
        madnis_unweight,
        madnis_sample_buffer,
        global_init,
    };

    std::array<std::uint32_t, 4> seed_parts;

    DerivedSeed(
        const std::optional<std::uint64_t>& seed = std::nullopt,
        SeedType seed_type = none,
        std::size_t job_index = 0,
        std::size_t channel_index = 0,
        std::size_t stream_index = 0
    );
};

// MIXMAX vielbein with the 64 global-run-seed bits applied (seed_parts[0..1], i.e.
// effective seed bits [64,128)). These do not change within a run, so each RNG caches
// this and only recomputes it if the run seed actually differs (e.g. when madspace is
// driven as a library). The per-call bits [0,64) are applied on top.
struct RunSeedSkip {
    std::array<std::uint64_t, mixmax_engine::state_size> state;
    std::uint64_t run_seed = 0;
    bool valid = false;

    // refresh `state` for `run_seed_hi:run_seed_lo` if it is not already cached
    void update(std::uint32_t run_seed_hi, std::uint32_t run_seed_lo) {
        std::uint64_t key = (std::uint64_t(run_seed_hi) << 32) | run_seed_lo;
        if (valid && key == run_seed) {
            return;
        }
        mixmax_engine::run_seed_prefix(state.data(), run_seed_lo, run_seed_hi);
        run_seed = key;
        valid = true;
    }
};

class MixMaxRandom {
public:
    MixMaxRandom() : MixMaxRandom(DerivedSeed()) {}
    MixMaxRandom(DerivedSeed seed) { apply_seed(seed); }
    explicit MixMaxRandom(std::uint64_t seed) : MixMaxRandom(DerivedSeed(seed)) {}
    void set_seed(DerivedSeed seed) { apply_seed(seed); }
    double generate_double() { return _mixmax.flat(); }
    std::size_t generate_int(std::size_t max_int) {
        return std::min<std::size_t>(_mixmax.flat() * max_int, max_int - 1);
    }

private:
    void apply_seed(const DerivedSeed& seed) {
        _run_skip.update(seed.seed_parts[0], seed.seed_parts[1]);
        // remaining bits: seed_parts[3] at [0,32), seed_parts[2] at [32,64)
        _mixmax.seed_from_state(
            _run_skip.state.data(), seed.seed_parts[3], seed.seed_parts[2]
        );
    }

    mixmax_engine _mixmax;
    RunSeedSkip _run_skip;
};

} // namespace madspace

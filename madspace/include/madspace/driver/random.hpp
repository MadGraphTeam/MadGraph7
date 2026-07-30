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

class MixMaxRandom {
public:
    MixMaxRandom() : MixMaxRandom(DerivedSeed()) {}
    MixMaxRandom(DerivedSeed seed) :
        _mixmax(
            seed.seed_parts[0],
            seed.seed_parts[1],
            seed.seed_parts[2],
            seed.seed_parts[3]
        ) {}
    explicit MixMaxRandom(std::uint64_t seed) : MixMaxRandom(DerivedSeed(seed)) {}
    void set_seed(DerivedSeed seed) {
        _mixmax.seed_uniquestream(
            seed.seed_parts[0],
            seed.seed_parts[1],
            seed.seed_parts[2],
            seed.seed_parts[3]
        );
    }
    double generate_double() { return _mixmax.flat(); }
    std::size_t generate_int(std::size_t max_int) {
        return std::min<std::size_t>(_mixmax.flat() * max_int, max_int - 1);
    }

private:
    mixmax_engine _mixmax;
};

} // namespace madspace

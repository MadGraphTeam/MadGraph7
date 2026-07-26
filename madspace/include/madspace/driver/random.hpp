#pragma once

#include <cstdint>
#include <initializer_list>
#include <random>
#include <string_view>

namespace madspace {

// Mixes `seed` with `salts` (splitmix64-based) to derive an independent
// sub-seed, e.g. for keeping different random streams derived from the same
// base seed uncorrelated. `seed == 0` is a sentinel for "non-deterministic"
// and passes through unchanged.
std::uint64_t
mix_seed(std::uint64_t seed, std::initializer_list<std::uint64_t> salts = {});

// Builds a seeded std::mt19937 from `seed` and `salts`. `seed == 0` seeds
// non-deterministically from std::random_device instead.
std::mt19937
seeded_rng(std::uint64_t seed, std::initializer_list<std::uint32_t> salts = {});

// Deterministic (platform/compiler-independent) 64-bit FNV-1a hash of `s`,
// used to fold a name (e.g. a global's name) into a mix_seed()/seeded_rng()
// salt. Not for anything security-sensitive -- just needs to be stable across
// runs/machines and have a low accidental-collision rate.
std::uint64_t hash_string(std::string_view s);

// Registry of the top-level salts used across madspace, so independent RNG
// streams derived from the same base seed by different subsystems never
// collide, even if they happen to be given the same base seed. Each is used
// at its documented call site; add new ones here (with a comment pointing at
// the call site and the salt tuple shape it's used in) rather than picking an
// arbitrary local value.
namespace salt {

// driver/channel_generator.cpp -- ChannelEventGenerator::start_job's per-job
// base seed: mix_seed(seed, {job.channel_index, job_kind_*, ...}).
inline constexpr std::uint64_t job_kind_survey = 1;
inline constexpr std::uint64_t job_kind_generate = 2;

// driver/channel_generator.cpp -- sub-seed for an individual Runtime::run()
// call within a job: mix_seed(job.rng_seed, {phase_*, call_index}).
inline constexpr std::uint64_t phase_generate = 1;
inline constexpr std::uint64_t phase_unweight = 2;

// driver/event_generator.cpp -- EventGenerator's post-generation passes:
// seeded_rng(seed, {combine_select}) / seeded_rng(seed, {lhe_complete}) /
// seeded_rng(seed, {lhe_complete, batch_seq}) /
// seeded_rng(seed, {unweight_pass, call_index, channel_index}).
inline constexpr std::uint64_t combine_select = 1;
inline constexpr std::uint64_t lhe_complete = 2;
// Named unweight_pass, not unweight_all, to not collide with the unrelated
// member function EventGenerator::unweight_all().
inline constexpr std::uint64_t unweight_pass = 3;

// driver/madnis_training.cpp -- MadnisTraining::start_single_job's per-job
// base seed: mix_seed(seed, {job_kind_madnis_train, channel_index, channel_seq}).
inline constexpr std::uint64_t job_kind_madnis_train = 100;

// phasespace/*.cpp -- random initialization of a named global (MLP weights
// and biases): see global_init_seed().
inline constexpr std::uint64_t global_init = 200;

} // namespace salt

// Derives the seed used to randomly initialize the global named `name`,
// salted (via salt::kGlobalInit and a hash of `name`) so it collides neither
// between different globals nor with any other subsystem's stream derived
// from the same base `seed`. `seed == 0` passes through unchanged (the usual
// non-deterministic sentinel).
std::uint64_t global_init_seed(std::uint64_t seed, std::string_view name);

} // namespace madspace

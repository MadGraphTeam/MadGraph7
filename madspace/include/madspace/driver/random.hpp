#pragma once

#include <cstdint>
#include <initializer_list>
#include <random>

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

} // namespace madspace

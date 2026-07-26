#include "madspace/driver/random.hpp"

#include <vector>

namespace madspace {

std::uint64_t mix_seed(std::uint64_t seed, std::initializer_list<std::uint64_t> salts) {
    if (seed == 0) {
        return 0;
    }
    auto splitmix = [](std::uint64_t x) {
        x += 0x9E3779B97F4A7C15ull;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        return x ^ (x >> 31);
    };
    std::uint64_t h = splitmix(seed);
    for (auto s : salts) {
        h = splitmix(h ^ splitmix(s));
    }
    return h ? h : 1;
}

std::mt19937
seeded_rng(std::uint64_t seed, std::initializer_list<std::uint32_t> salts) {
    if (seed == 0) {
        std::random_device rand_device;
        return std::mt19937(rand_device());
    }
    std::vector<std::uint32_t> data;
    data.reserve(2 + salts.size());
    data.push_back(static_cast<std::uint32_t>(seed & 0xFFFFFFFFu));
    data.push_back(static_cast<std::uint32_t>(seed >> 32));
    data.insert(data.end(), salts.begin(), salts.end());
    std::seed_seq seq(data.begin(), data.end());
    return std::mt19937(seq);
}

std::uint64_t hash_string(std::string_view s) {
    // FNV-1a, 64-bit.
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (unsigned char c : s) {
        h ^= c;
        h *= 0x100000001b3ull;
    }
    return h;
}

std::uint64_t global_init_seed(std::uint64_t seed, std::string_view name) {
    return mix_seed(seed, {salt::global_init, hash_string(name)});
}

} // namespace madspace

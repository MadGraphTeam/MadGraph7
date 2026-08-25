#pragma once

#include "device.hpp"
#include "gpu_abstraction.cuh"
#include "madspace/driver/random.hpp"
#include "tensor.cuh" // MAX_THREADS_PER_BLOCK, THREADS_MULTIPLE

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace madspace {
namespace gpu {

// One MIXMAX substream per DerivedSeed stream index; the bank covers all of them so
// every RNG-consuming instruction draws from its own non-overlapping substream.
constexpr std::size_t RNG_ENGINE_COUNT = DerivedSeed::max_stream_count;

// Grid geometry for a draw of `count` numbers from the engine bank: split evenly over
// as few threads as cover `count` in ceil(count / RNG_ENGINE_COUNT) draws per thread,
// so the grid never has a large idle remainder.
inline std::pair<std::size_t, std::size_t> rng_grid(std::size_t count) {
    std::size_t draws = (count + RNG_ENGINE_COUNT - 1) / RNG_ENGINE_COUNT;
    std::size_t needed = (count + draws - 1) / draws;
    std::size_t threads = std::min<std::size_t>(
        MAX_THREADS_PER_BLOCK,
        ((needed + THREADS_MULTIPLE - 1) / THREADS_MULTIPLE) * THREADS_MULTIPLE
    );
    std::size_t blocks = (needed + threads - 1) / threads;
    return {blocks, threads};
}

__global__ void kernel_rng_seed(
    std::size_t count,
    mixmax_engine* engines,
    std::uint32_t s0,
    std::uint32_t s1,
    std::uint32_t s2,
    std::uint32_t s3
) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    // low bits of s3 are the reserved DerivedSeed stream index; OR in the engine index
    // to give every engine its own non-colliding MIXMAX substream.
    engines[i] = mixmax_engine(s0, s1, s2, s3 | static_cast<std::uint32_t>(i));
}

// Bank of RNG_ENGINE_COUNT MIXMAX engines living in device memory, seeded from a
// DerivedSeed. Re-seeding only records the seed; the actual (re-)seed kernel runs
// lazily on the next run() call, on the main (RNG) stream.
class GpuRandom {
public:
    GpuRandom() {
        void* ptr;
        check_error(gpuMalloc(&ptr, sizeof(mixmax_engine) * RNG_ENGINE_COUNT));
        _engines = static_cast<mixmax_engine*>(ptr);
    }
    ~GpuRandom() {
        if (_engines) {
            gpuFree(_engines);
        }
    }
    GpuRandom(const GpuRandom&) = delete;
    GpuRandom& operator=(const GpuRandom&) = delete;
    GpuRandom(GpuRandom&& other) noexcept :
        _engines(other._engines), _seed(other._seed), _pending(other._pending) {
        other._engines = nullptr;
    }
    GpuRandom& operator=(GpuRandom&& other) noexcept {
        if (_engines) {
            gpuFree(_engines);
        }
        _engines = other._engines;
        _seed = other._seed;
        _pending = other._pending;
        other._engines = nullptr;
        return *this;
    }

    void set_seed(DerivedSeed seed) {
        // the engine index is folded into the low bits of the stream field (see
        // kernel_rng_seed), so callers must leave it unset
        if (seed.seed_parts[3] & 0xFFFFu) {
            throw std::invalid_argument(
                "GpuRandom::set_seed: stream index is reserved for the GPU engine bank"
            );
        }
        _seed = seed;
        _pending = true;
    }

    void reseed_if_needed(gpuStream_t stream) {
        if (!_pending) {
            return;
        }
        auto [blocks, threads] = rng_grid(RNG_ENGINE_COUNT);
        kernel_rng_seed<<<blocks, threads, 0, stream>>>(
            RNG_ENGINE_COUNT,
            _engines,
            _seed.seed_parts[0],
            _seed.seed_parts[1],
            _seed.seed_parts[2],
            _seed.seed_parts[3]
        );
        check_error();
        _pending = false;
    }

    mixmax_engine* engines() { return _engines; }

private:
    mixmax_engine* _engines = nullptr;
    DerivedSeed _seed;
    bool _pending = true;
};

template <typename F, typename... Args>
void launch_rng_kernel(
    F kernel, std::size_t count, GpuRandom& rng, gpuStream_t stream, Args... args
) {
    if (count == 0) {
        return;
    }
    auto [blocks, threads] = rng_grid(count);
    kernel<<<blocks, threads, 0, stream>>>(count, rng.engines(), args...);
    check_error();
}

} // namespace gpu
} // namespace madspace

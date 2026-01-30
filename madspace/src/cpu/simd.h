#pragma once

#ifdef USE_SIMD

#ifdef USE_SIMD_NEON
#include "simd_arm.h"
#endif

#ifdef USE_SIMD_AVX2
#include "simd_x86_256.h"
#endif

#ifdef USE_SIMD_AVX512
#include "simd_x86_512.h"
#endif

#else // USE_SIMD

constexpr int simd_vec_size = 1;

#endif // USE_SIMD

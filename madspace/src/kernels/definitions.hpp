#pragma once

#include "madspace/constants.hpp"

#if defined(__CUDACC__) || defined(__HIPCC__)
#include "../gpu/kernel_definitions.cuh"
#else
#include "../cpu/kernel_definitions.hpp"
#endif

namespace madspace {
namespace kernels {

inline constexpr double EPS = 1e-12;
inline constexpr double EPS2 = 1e-24;

template <typename T, int dim>
using FIn = typename T::template FIn<dim>;
template <typename T, int dim>
using IIn = typename T::template IIn<dim>;
template <typename T, int dim>
using BIn = typename T::template BIn<dim>;
// inputs marked as single in the instruction set (no batch dimension)
template <typename T, int dim>
using FSIn = typename T::template FSIn<dim>;
template <typename T, int dim>
using ISIn = typename T::template ISIn<dim>;
template <typename T, int dim>
using FOut = typename T::template FOut<dim>;
template <typename T, int dim>
using IOut = typename T::template IOut<dim>;
template <typename T, int dim>
using BOut = typename T::template BOut<dim>;
template <typename T>
using FVal = typename T::FVal;
template <typename T>
using IVal = typename T::IVal;
template <typename T>
using BVal = typename T::BVal;

// The smallest normal double, the floor of quantities such as 1 - x that have
// to stay positive and normal where a power of a random number underflows.
inline constexpr double SMALLEST_NORMAL = 2.2250738585072014e-308;

// base^exponent, zero for a base that is not positive or not a number. The
// base is replaced before pow sees it: with a fractional exponent a negative
// base gives NaN, and one NaN in a SIMD lane is enough to spoil the whole
// batch.
template <typename T>
KERNELSPEC FVal<T> positive_pow(FVal<T> base, FVal<T> exponent) {
    auto positive = base > 0.;
    return where(positive, pow(where(positive, base, 1.), exponent), 0.);
}

} // namespace kernels
} // namespace madspace

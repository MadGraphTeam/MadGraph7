#pragma once

#include "simd.hpp"
#include "tensor.hpp"

#include <cmath>

#define KERNELSPEC

namespace madspace {
namespace kernels {

template <typename F>
struct CpuTypesT {
    template <int dim>
    using FIn = const TensorView<F, dim>;
    template <int dim>
    using IIn = const TensorView<me_int_t, dim>;
    template <int dim>
    using FOut = TensorView<F, dim>;
    template <int dim>
    using IOut = TensorView<me_int_t, dim>;
    template <int dim>
    using FSIn = const TensorView<F, dim>;
    template <int dim>
    using ISIn = const TensorView<me_int_t, dim>;
    using FVal = F;
    using IVal = me_int_t;
    using BVal = bool;
    // the same policy with a different float type, for conversion kernels
    using F32 = CpuTypesT<float>;
    using F64 = CpuTypesT<double>;
};
using CpuTypes = CpuTypesT<double>;
using CpuTypes32 = CpuTypesT<float>;

inline double where(bool condition, double val_true, double val_false) {
    return condition ? val_true : val_false;
}
inline float where(bool condition, float val_true, float val_false) {
    return condition ? val_true : val_false;
}
inline me_int_t where(bool condition, me_int_t val_true, me_int_t val_false) {
    return condition ? val_true : val_false;
}
inline double min(double arg1, double arg2) { return arg1 < arg2 ? arg1 : arg2; }
inline double max(double arg1, double arg2) { return arg1 > arg2 ? arg1 : arg2; }
inline float min(float arg1, float arg2) { return arg1 < arg2 ? arg1 : arg2; }
inline float max(float arg1, float arg2) { return arg1 > arg2 ? arg1 : arg2; }
inline std::size_t single_index(me_int_t arg) { return arg; }

// Kernels with tables per configuration (or another per-event index) run
// their body once per value of the index among the lanes of a vector, with
// the mask of the lanes of that value, so that the tables are those of one
// configuration in each pass. Outside of SIMD, and on the GPU, the body runs
// once with the index of the event.
template <class F>
inline void for_each_value(me_int_t value, F&& body) {
    body(static_cast<std::size_t>(value), true);
}

// a single (not batched) input of dimension zero as a plain value, e.g. a
// table entry of the configuration of a pass of for_each_value
template <ScalarType T>
inline T single_value(const TensorView<T, 0>& view) {
    return view;
}

// writes value to an output in the lanes of mask (all of them outside of SIMD)
template <ScalarType T>
inline void masked_store(TensorView<T, 0> view, bool, T value) {
    view = value;
}

// the value rounded to single precision, as a cast to float and back
inline double round_to_float32(double arg) { return static_cast<float>(arg); }

using std::acos;
using std::asinh;
using std::atan;
using std::atan2;
using std::atanh;
using std::cos;
using std::cosh;
using std::erf;
using std::exp;
using std::expm1;
using std::fabs;
using std::fma;
using std::log;
using std::log1p;
using std::pow;
using std::sin;
using std::sinh;
using std::sqrt;
using std::tan;
using std::tanh;

using std::isnan;

#ifdef USE_SIMD
struct SimdTypes {
    template <int dim>
    using FIn = const cpu::VectorizedTensorView<FVec, double, dim, false>;
    template <int dim>
    using IIn = const cpu::VectorizedTensorView<IVec, me_int_t, dim, false>;
    template <int dim>
    using FOut = cpu::VectorizedTensorView<FVec, double, dim, false>;
    template <int dim>
    using IOut = cpu::VectorizedTensorView<IVec, me_int_t, dim, false>;
    template <int dim>
    using FSIn = const cpu::SingleVectorizedTensorView<FVec, double, dim, false>;
    template <int dim>
    using ISIn = const cpu::SingleVectorizedTensorView<IVec, me_int_t, dim, false>;
    using FVal = FVec;
    using IVal = IVec;
    using BVal = BVec;
};

struct SimdTypes32 {
    template <int dim>
    using FIn = const cpu::VectorizedTensorView<FVec32, float, dim, false>;
    template <int dim>
    using IIn = const cpu::VectorizedTensorView<IVec32, me_int_t, dim, false>;
    template <int dim>
    using FOut = cpu::VectorizedTensorView<FVec32, float, dim, false>;
    template <int dim>
    using IOut = cpu::VectorizedTensorView<IVec32, me_int_t, dim, false>;
    template <int dim>
    using FSIn = const cpu::SingleVectorizedTensorView<FVec32, float, dim, false>;
    template <int dim>
    using ISIn = const cpu::SingleVectorizedTensorView<IVec32, me_int_t, dim, false>;
    using FVal = FVec32;
    using IVal = IVec32;
    using BVal = BVec32;
};

template <class F>
inline void for_each_value(IVec value, F&& body) {
    int lanes[simd_vec_size];
    vstore(lanes, value);
    for (int k = 0; k < simd_vec_size; ++k) {
        bool seen = false;
        for (int l = 0; l < k; ++l) {
            seen = seen || lanes[l] == lanes[k];
        }
        if (!seen) {
            body(static_cast<std::size_t>(lanes[k]), value == IVec(lanes[k]));
        }
    }
}

template <class V, ScalarType T, bool is_batch>
inline T single_value(const cpu::SingleVectorizedTensorView<V, T, 0, is_batch>& view) {
    return view.value();
}

template <class V, ScalarType T, bool is_batch>
inline void
masked_store(cpu::VectorizedTensorView<V, T, 0, is_batch> view, BVec mask, V value) {
    view = where(mask, value, V(view));
}
#else  // USE_SIMD
using SimdTypes = CpuTypes;
using SimdTypes32 = CpuTypes32;
#endif // USE_SIMD

} // namespace kernels
} // namespace madspace

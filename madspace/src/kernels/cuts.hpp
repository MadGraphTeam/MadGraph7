#pragma once

#include "definitions.hpp"

namespace madspace {
namespace kernels {

// Kernels

template <typename T>
KERNELSPEC void kernel_cut_unphysical(
    FIn<T, 0> w_in, FIn<T, 2> p, FIn<T, 0> x1, FIn<T, 0> x2, FOut<T, 0> w_out
) {
    FVal<T> w = where(isnan(w_in), 0., w_in);
    for (std::size_t i = 0; i < p.size(); ++i) {
        auto p_i = p[i];
        for (std::size_t j = 0; j < 4; ++j) {
            w = where(isnan(p_i[j]), 0., w);
        }
    }
    w_out = where(
        (x1 < 0.) | (x1 > 1.) | isnan(x1) | (x2 < 0.) | (x2 > 1.) | isnan(x2), 0., w
    );
}

template <typename T>
KERNELSPEC void
kernel_cut_one(FIn<T, 0> obs, FIn<T, 0> min, FIn<T, 0> max, FOut<T, 0> w) {
    w = where((obs < min) | (obs > max), FVal<T>(0.), 1.);
}

// min and max are ONE bound applied to every observable of the vector
// (instruction_set.yaml: type [float]), not one bound per observable: viewed as
// vectors, min[i] for i > 0 read through a stride the scalar tensor does not
// have (uninitialised Sizes entries) -- an illegal address on the GPU as soon as
// a cut acts on two objects or more (the jets of p p > j j).
template <typename T>
KERNELSPEC void
kernel_cut_all(FIn<T, 1> obs, FIn<T, 0> min, FIn<T, 0> max, FOut<T, 0> w) {
    FVal<T> cut = 1.;
    for (std::size_t i = 0; i < obs.size(); ++i) {
        cut = where((obs[i] < min) | (obs[i] > max), 0., cut);
    }
    w = cut;
}

template <typename T>
KERNELSPEC void
kernel_cut_any(FIn<T, 1> obs, FIn<T, 0> min, FIn<T, 0> max, FOut<T, 0> w) {
    FVal<T> cut = 0.;
    for (std::size_t i = 0; i < obs.size(); ++i) {
        cut = where((obs[i] < min) | (obs[i] > max), cut, 1.);
    }
    w = cut;
}

} // namespace kernels
} // namespace madspace

#pragma once

#include "definitions.hpp"

namespace madspace {
namespace kernels {

template <typename T>
KERNELSPEC void kernel_sample_discrete(
    FIn<T, 0> r, IIn<T, 0> option_count, IOut<T, 0> output, FOut<T, 0> det
) {
    IVal<T> opt_count_i(option_count);
    FVal<T> opt_count_f(opt_count_i);
    IVal<T> option(r * opt_count_f);
    output = option;
    det = opt_count_f;
}

template <typename T>
KERNELSPEC void kernel_sample_discrete_inverse(
    IIn<T, 0> index, IIn<T, 0> option_count, FOut<T, 0> r, FOut<T, 0> det
) {
    IVal<T> opt_count_i(option_count), index_i(index);
    FVal<T> opt_count_f(opt_count_i), index_f(index_i);
    r = (index_f + 0.5) / opt_count_f;
    det = 1. / opt_count_f;
}

// An option whose probability is exactly zero cannot be produced, so its
// correct contribution is a zero-weight event -- not inf and not NaN. Exact
// zeros are routine here: the probabilities are parton densities (NNPDF4.0
// returns exactly 0.0 for c/b below threshold and at large x, where NNPDF2.3
// had a ~1e-15 floor) optionally multiplied by an active-flavour mask. Two
// distinct divisions have to be guarded:
//
//   (A) every probability is zero -> prob_norm == 0 -> 0/0 -> NaN in every
//       lane. The whole point is kinematically inaccessible.
//   (B) the *selected* option has probability zero -> 1/0 -> inf. This is
//       reachable even when other options are non-zero: cum_prob accumulates
//       in floating point and can end just below r, which selects a trailing
//       zero-probability option.
//
// The guards sit on the divisions, never on the input probabilities: flooring
// the densities would silently alter the physics everywhere else. They test
// "!= 0" rather than "> 0" on purpose, so that every non-zero divisor -- including
// a negative one -- divides exactly as it did before and keeps the sign of its
// weight. That makes the guards provably inert unless the divisor is literally
// zero. This is not academic: a "> 0" version of these guards changed results
// for NNPDF2.3 badly enough to abort the run, while the "!= 0" version below
// reproduces it to 0.003 sigma.

template <typename T>
KERNELSPEC void kernel_sample_discrete_probs(
    FIn<T, 0> r, FIn<T, 1> probs, IOut<T, 0> output, FOut<T, 0> det
) {
    FVal<T> prob_norm(0.);
    for (std::size_t i = 0; i < probs.size(); ++i) {
        prob_norm = prob_norm + probs[i];
    }
    // (A) fall back to a unit norm so the ratios below are 0/1 = 0 instead of
    // 0/0; prob_out then stays zero and the det guard turns the point into a
    // zero-weight event.
    auto norm_ok = prob_norm != 0.;
    auto norm_safe = where(norm_ok, prob_norm, FVal<T>(1.));
    FVal<T> cum_prob(0.), prob_out(0.);
    IVal<T> option(0);
    for (std::size_t i = 0; i < probs.size(); ++i) {
        auto prob = probs[i] / norm_safe;
        auto mask = r < cum_prob;
        cum_prob = cum_prob + prob;
        option = where(mask, option, IVal<T>(i));
        prob_out = where(mask, prob_out, prob);
    }
    // (B) zero probability -> zero weight. Also pins the option to 0 in case
    // (A), where the loop would otherwise return the last index by default.
    auto prob_ok = prob_out != 0.;
    auto prob_safe = where(prob_ok, prob_out, FVal<T>(1.));
    output = where(norm_ok, option, IVal<T>(0));
    det = where(prob_ok, FVal<T>(1.) / prob_safe, FVal<T>(0.));
}

template <typename T>
KERNELSPEC void kernel_sample_discrete_probs_inverse(
    IIn<T, 0> index, FIn<T, 1> probs, FOut<T, 0> r, FOut<T, 0> det
) {
    FVal<T> prob_norm(0.);
    for (std::size_t i = 0; i < probs.size(); ++i) {
        prob_norm = prob_norm + probs[i];
    }
    // Same guard as case (A) above. Here det is the probability itself rather
    // than its reciprocal, so a zero probability already means zero weight and
    // case (B) needs no guard: with a unit fallback norm every ratio is 0,
    // giving r = 0 and det = 0, i.e. a deterministic zero-weight point instead
    // of NaN.
    auto norm_safe = where(prob_norm != 0., prob_norm, FVal<T>(1.));
    FVal<T> cum_prob(0.), random(0.), prob_out(0.);
    for (std::size_t i = 0; i < probs.size(); ++i) {
        auto prob = probs[i] / norm_safe;
        cum_prob = cum_prob + prob;
        auto mask = index == i;
        random = where(mask, cum_prob + 0.5 * prob, random);
        prob_out = where(mask, prob, prob_out);
    }
    r = random;
    det = prob_out;
}

template <typename T>
KERNELSPEC void backward_kernel_sample_discrete_probs_inverse(
    IIn<T, 0> index,
    FIn<T, 1> probs,
    FIn<T, 0> r_grad,
    FIn<T, 0> det_grad,
    IOut<T, 0> index_grad,
    FOut<T, 1> probs_grad
) {
    FVal<T> prob_norm(0.);
    for (std::size_t i = 0; i < probs.size(); ++i) {
        prob_norm = prob_norm + probs[i];
    }
    // Matches the forward guard: where prob_norm is zero the forward det is
    // identically 0 (a constant), so its gradient is 0 rather than NaN.
    auto norm_ok = prob_norm != 0.;
    auto norm_safe = where(norm_ok, prob_norm, FVal<T>(1.));
    FVal<T> det_grad_out(0.);
    auto prob = probs.gather(index) / norm_safe;
    for (std::size_t i = 0; i < probs.size(); ++i) {
        probs_grad[i] = where(
            norm_ok,
            (where(index == i, FVal<T>(1.), 0.) - prob) / norm_safe * det_grad,
            FVal<T>(0.)
        );
    }
}

} // namespace kernels
} // namespace madspace

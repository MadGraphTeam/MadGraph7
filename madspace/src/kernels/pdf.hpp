#pragma once

#include "definitions.hpp"

namespace madspace {
namespace kernels {

template <typename T>
KERNELSPEC std::size_t
binary_search(FVal<T> x, FIn<T, 1> knots, FVal<T>& x_low, FVal<T>& x_high) {
    int64_t index_low = 0, index_high = knots.size();
    while (index_low <= index_high) {
        std::size_t index = index_low + (index_high - index_low) / 2;
        x_low = knots[index];
        x_high = knots[index + 1];
        if (x_low <= x && x <= x_high) {
            return index;
        }
        if (x_low < x) {
            index_low = index + 1;
        } else {
            index_high = index - 1;
        }
    }
    return 0;
}

template <typename T>
KERNELSPEC void kernel_interpolate_pdf(
    FIn<T, 0> x,
    FIn<T, 0> q,
    IIn<T, 1> pid_indices,
    FIn<T, 1> grid_logx,
    FIn<T, 1> grid_logq2,
    FIn<T, 3> grid_coeffs,
    FOut<T, 1> pdf
) {
    auto logx = log(x), logq2 = log(q * q);
    FVal<T> logx_low, logx_high, logq2_low, logq2_high;
    std::size_t x_index = binary_search<T>(logx, grid_logx, logx_low, logx_high);
    std::size_t q2_index = binary_search<T>(logq2, grid_logq2, logq2_low, logq2_high);

    auto t_logx = (logx - logx_low) / (logx_high - logx_low);
    auto t_logx_2 = t_logx * t_logx;
    auto t_logx_3 = t_logx_2 * t_logx;

    auto t_logq2 = (logq2 - logq2_low) / (logq2_high - logq2_low);
    auto t_logq2_2 = t_logq2 * t_logq2;
    auto t_logq2_3 = t_logq2_2 * t_logq2;

    auto vl_val = 2 * t_logq2_3 - 3 * t_logq2_2 + 1;
    auto vdl_val = t_logq2_3 - 2 * t_logq2_2 + t_logq2;
    auto vh_val = -2 * t_logq2_3 + 3 * t_logq2_2;
    auto vdh_val = t_logq2_3 - t_logq2_2;

    FVal<T> values[16] = {
        vl_val * t_logx_3,
        vl_val * t_logx_2,
        vl_val * t_logx,
        vl_val,
        vh_val * t_logx_3,
        vh_val * t_logx_2,
        vh_val * t_logx,
        vh_val,
        vdl_val * t_logx_3,
        vdl_val * t_logx_2,
        vdl_val * t_logx,
        vdl_val,
        vdh_val * t_logx_3,
        vdh_val * t_logx_2,
        vdh_val * t_logx,
        vdh_val
    };

    for (std::size_t i = 0; i < pid_indices.size(); ++i) {
        std::size_t pid_index = pid_indices[i];
        FVal<T> result(0.);
        for (std::size_t j = 0; j < 16; ++j) {
            std::size_t index = (grid_logx.size() - 1) * q2_index + x_index;
            result = result + values[j] * grid_coeffs.get(j, pid_index, index);
        }
        pdf[i] = result;
    }
}

template <typename T>
KERNELSPEC void kernel_interpolate_alpha_s(
    FIn<T, 0> q, FIn<T, 1> grid_logq2, FIn<T, 2> grid_coeffs, FOut<T, 0> alpha_s
) {
    auto logq2 = log(q * q);
    FVal<T> logq2_low, logq2_high;
    std::size_t q2_index = binary_search<T>(logq2, grid_logq2, logq2_low, logq2_high);

    auto t_logq2 = (logq2 - logq2_low) / (logq2_high - logq2_low);
    auto t_logq2_2 = t_logq2 * t_logq2;
    auto t_logq2_3 = t_logq2_2 * t_logq2;

    auto vl_val = 2 * t_logq2_3 - 3 * t_logq2_2 + 1;
    auto vdl_val = t_logq2_3 - 2 * t_logq2_2 + t_logq2;
    auto vh_val = -2 * t_logq2_3 + 3 * t_logq2_2;
    auto vdh_val = t_logq2_3 - t_logq2_2;
    FVal<T> values[4] = {vl_val, vh_val, vdl_val, vdh_val};

    FVal<T> result(0.);
    for (std::size_t j = 0; j < 4; ++j) {
        result = result + values[j] * grid_coeffs.get(j, q2_index);
    }
    alpha_s = result;
}

// phi of the equivalent-photon density of a proton (phi_f of PhotonFlux.f of
// MadGraph) at y = x^2 / (1 - x) and qq = Q^2 / (0.71 GeV^2), with
// log((1 + qq) / qq) as log1p(1 / qq)
template <typename T>
KERNELSPEC FVal<T> epa_proton_phi(FVal<T> y, FVal<T> qq) {
    constexpr double a = 7.16, b = -3.96, c = 0.028;
    FVal<T> q1 = 1. + qq;
    FVal<T> i1 = 1. / q1, i2 = i1 * i1, i3 = i2 * i1;
    FVal<T> f = (1. + a * y) * (-log1p(1. / qq) + i1 + 0.5 * i2 + i3 / 3.);
    f = f + (1. - b) * y / (4. * qq) * i3;
    return f +
        c * (1. + 0.25 * y) *
        (log1p(-b * i1) + b * i1 + 0.5 * b * b * i2 + b * b * b * i3 / 3.);
}

template <typename T>
KERNELSPEC void
kernel_elastic_photon_density(FIn<T, 0> x, FIn<T, 0> q, FOut<T, 0> pdf) {
    // alpha, the proton mass and the scale of the dipole form factor of
    // epa_proton of PhotonFlux.f
    constexpr double alpha = 0.0072992701, mass = 0.938, q0 = 0.71;
    FVal<T> xv = x, qv = q;
    auto below = xv < 1.;
    FVal<T> omx = where(below, 1. - xv, FVal<T>(1.));
    FVal<T> y = xv * xv / omx;
    FVal<T> q2_min = mass * mass * y, q2_max = qv * qv;
    FVal<T> f = alpha / PI * omx *
        (epa_proton_phi<T>(y, q2_max / q0) - epa_proton_phi<T>(y, q2_min / q0));
    pdf = where(below & (q2_min < q2_max), max(f, FVal<T>(0.)), FVal<T>(0.));
}

} // namespace kernels
} // namespace madspace

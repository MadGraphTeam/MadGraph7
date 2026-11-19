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

// the 16 basis functions of the bicubic interpolation of a grid segment at the
// positions t_logx and t_logq2 within it, in the order of the coefficients
template <typename T>
KERNELSPEC void pdf_basis(FVal<T> t_logx, FVal<T> t_logq2, FVal<T>* values) {
    auto t_logx_2 = t_logx * t_logx;
    auto t_logx_3 = t_logx_2 * t_logx;
    auto t_logq2_2 = t_logq2 * t_logq2;
    auto t_logq2_3 = t_logq2_2 * t_logq2;

    FVal<T> q2_basis[4] = {
        2 * t_logq2_3 - 3 * t_logq2_2 + 1,
        -2 * t_logq2_3 + 3 * t_logq2_2,
        t_logq2_3 - 2 * t_logq2_2 + t_logq2,
        t_logq2_3 - t_logq2_2
    };
    for (std::size_t i = 0; i < 4; ++i) {
        values[4 * i] = q2_basis[i] * t_logx_3;
        values[4 * i + 1] = q2_basis[i] * t_logx_2;
        values[4 * i + 2] = q2_basis[i] * t_logx;
        values[4 * i + 3] = q2_basis[i];
    }
}

template <typename T>
KERNELSPEC FVal<T> pdf_segment(
    FIn<T, 3> grid_coeffs, std::size_t pid_index, std::size_t node, FVal<T>* values
) {
    FVal<T> result(0.);
    for (std::size_t j = 0; j < 16; ++j) {
        result = result + values[j] * grid_coeffs.get(j, pid_index, node);
    }
    return result;
}

// _extrapolateLinear of the continuation extrapolator of LHAPDF: the
// value at u = (log x - log x_low) / (log x_high - log x_low) from those at
// x_low and x_high, linear in log x, of log y where both values are above 1e-3
// (which keeps y positive) and of y otherwise
template <typename T>
KERNELSPEC FVal<T> pdf_extrapolate_linear(FVal<T> u, FVal<T> y_low, FVal<T> y_high) {
    auto positive = (y_low > 1e-3) & (y_high > 1e-3);
    FVal<T> log_low = log(where(positive, y_low, FVal<T>(1.)));
    FVal<T> log_high = log(where(positive, y_high, FVal<T>(1.)));
    return where(
        positive, exp(log_low + u * (log_high - log_low)), y_low + u * (y_high - y_low)
    );
}

// The densities outside of the grid, as the continuation extrapolator of LHAPDF
// (ContinuationExtrapolator, its default): below the first x knot linear in
// log x from the first two x knots (of log xf where both values are above 1e-3,
// else of xf), above the last Q^2 knot linear in log Q^2 from the last two, both
// at once first in Q^2 and then in x, and below the first Q^2 knot
//
//   xf(Q^2) = xf(Q0^2) r^(g r + 1 - r),  r = Q^2 / Q0^2,
//
// with the anomalous dimension g = d log xf / d log Q^2 at Q0^2 (from the value
// at 1.01 Q0^2, at least -2.5, and 1 where |xf(Q0^2)| < 1e-5), which goes over
// into xf(Q0^2) r for Q^2 << Q0^2. Above the last x knot, where LHAPDF stops
// with an error, and for x or Q that is not positive the density is zero. It
// takes the bicubic at up to two points in x (a and b) and two in Q^2 (c and d):
// the point itself, below the grid x_min and the second x knot (t = 0 and 1 of
// the first segment), above it Q^2_max and the knot before it (t = 1 and 0 of
// the last segment), and below it Q^2_min and 1.01 Q^2_min. Each point takes
// only the bicubics and the continuation of its own region, in a branch, since
// the logarithms and exponentials of the continuations cost far more than the
// bicubics.
template <typename T>
KERNELSPEC void pdf_outside_grid(
    FVal<T> x,
    FVal<T> q,
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

    std::size_t x_knots = grid_logx.size() - 2, q2_knots = grid_logq2.size() - 2;
    bool below_x = x_index == 0, above_x = x_index == x_knots;
    bool below_q2 = q2_index == 0, above_q2 = q2_index == q2_knots;
    // zero above the grid in x, and for x or q that is not positive (as in
    // the padding of a batch), where the logarithms above are not finite
    if (above_x || !(x > 0.) || !(q > 0.)) {
        for (std::size_t i = 0; i < pid_indices.size(); ++i) {
            pdf[i] = FVal<T>(0.);
        }
        return;
    }
    FVal<T> logx_min = grid_logx[1], logx_min_1 = grid_logx[2];
    FVal<T> logq2_min = grid_logq2[1];
    FVal<T> logq2_max = grid_logq2[q2_knots], logq2_max_1 = grid_logq2[q2_knots - 1];

    std::size_t x_segment = below_x ? 1 : x_index;
    FVal<T> t_x = (logx - logx_low) / (logx_high - logx_low);
    FVal<T> t_a = where(below_x, FVal<T>(0.), t_x);
    FVal<T> t_b(1.);

    std::size_t q2_segment_c = below_q2 ? 1 : (above_q2 ? q2_knots - 1 : q2_index);
    std::size_t q2_segment_d = q2_segment_c;
    FVal<T> t_q2 = (logq2 - logq2_low) / (logq2_high - logq2_low);
    FVal<T> t_c = where(below_q2, FVal<T>(0.), where(above_q2, FVal<T>(1.), t_q2));
    FVal<T> t_d(0.);
    FVal<T> log_r = logq2 - logq2_min, r(0.);
    if (below_q2) {
        FVal<T> logq2_min_1 = logq2_min + log(1.01), logq2_d_low, logq2_d_high;
        q2_segment_d =
            binary_search<T>(logq2_min_1, grid_logq2, logq2_d_low, logq2_d_high);
        t_d = (logq2_min_1 - logq2_d_low) / (logq2_d_high - logq2_d_low);
        r = exp(log_r);
    }

    // the second x point (b) only below the grid in x, the second Q^2 point (d)
    // only below or above it in Q^2
    bool need_b = below_x, need_d = below_q2 || above_q2;
    FVal<T> basis_ac[16], basis_bc[16], basis_ad[16], basis_bd[16];
    pdf_basis<T>(t_a, t_c, basis_ac);
    if (need_b) {
        pdf_basis<T>(t_b, t_c, basis_bc);
    }
    if (need_d) {
        pdf_basis<T>(t_a, t_d, basis_ad);
    }
    if (need_b && need_d) {
        pdf_basis<T>(t_b, t_d, basis_bd);
    }
    std::size_t row = grid_logx.size() - 1;
    std::size_t node_c = row * q2_segment_c + x_segment;
    std::size_t node_d = row * q2_segment_d + x_segment;

    FVal<T> u_x = (logx - logx_min) / (logx_min_1 - logx_min);
    FVal<T> u_q2 = (logq2 - logq2_max) / (logq2_max_1 - logq2_max);

    for (std::size_t i = 0; i < pid_indices.size(); ++i) {
        std::size_t pid_index = pid_indices[i];
        FVal<T> f_ac = pdf_segment<T>(grid_coeffs, pid_index, node_c, basis_ac);
        FVal<T> f_bc(0.), f_ad(0.), f_bd(0.);
        if (need_b) {
            f_bc = pdf_segment<T>(grid_coeffs, pid_index, node_c, basis_bc);
        }
        if (need_d) {
            f_ad = pdf_segment<T>(grid_coeffs, pid_index, node_d, basis_ad);
        }
        if (need_b && need_d) {
            f_bd = pdf_segment<T>(grid_coeffs, pid_index, node_d, basis_bd);
        }

        // in Q^2 within the grid, and below it at Q^2_min
        FVal<T> f_c = below_x ? pdf_extrapolate_linear<T>(u_x, f_ac, f_bc) : f_ac;
        if (below_q2) {
            // and at 1.01 Q^2_min
            FVal<T> f_d = below_x ? pdf_extrapolate_linear<T>(u_x, f_ad, f_bd) : f_ad;
            auto finite = fabs(f_c) >= 1e-5;
            FVal<T> f_c_safe = where(finite, f_c, FVal<T>(1.));
            FVal<T> anom = where(
                finite, max((f_d - f_c) / f_c_safe / 0.01, FVal<T>(-2.5)), FVal<T>(1.)
            );
            pdf[i] = f_c * exp((anom * r + 1. - r) * log_r);
        } else if (above_q2) {
            FVal<T> f_a = pdf_extrapolate_linear<T>(u_q2, f_ac, f_ad);
            if (below_x) {
                FVal<T> f_b = pdf_extrapolate_linear<T>(u_q2, f_bc, f_bd);
                pdf[i] = pdf_extrapolate_linear<T>(u_x, f_a, f_b);
            } else {
                pdf[i] = f_a;
            }
        } else {
            pdf[i] = f_c;
        }
    }
}

// The bicubic interpolation in log x and log Q^2 of the grid, and outside of it
// the extrapolation of pdf_outside_grid, which runs only for the points outside
// of the grid. The kernel is not vectorized, so both branch per point. The knots of the
// grid are padded by -inf and +inf, so that the segments of the n knots are 1 to n - 1,
// with the points below the grid in segment 0 and those above it in segment n, whose
// coefficients are zero.
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
    // (x and Q that are not positive or not a number fall below the grid)
    if (x_index == 0 || q2_index == 0 || q2_index == grid_logq2.size() - 2) {
        pdf_outside_grid<T>(x, q, pid_indices, grid_logx, grid_logq2, grid_coeffs, pdf);
        return;
    }

    auto t_logx = (logx - logx_low) / (logx_high - logx_low);
    auto t_logq2 = (logq2 - logq2_low) / (logq2_high - logq2_low);
    FVal<T> values[16];
    pdf_basis<T>(t_logx, t_logq2, values);
    std::size_t node = (grid_logx.size() - 1) * q2_index + x_index;
    for (std::size_t i = 0; i < pid_indices.size(); ++i) {
        pdf[i] = pdf_segment<T>(grid_coeffs, pid_indices[i], node, values);
    }
}

// The cubic interpolation in log Q^2 of the grid, and outside of it the
// extrapolation of AlphaS_Ipol of LHAPDF: below the first knot
// alpha_s(Q0^2) (Q^2 / Q0^2)^g with the logarithmic gradient g to the next
// knot of another Q, and above the last knot the last value, zero for Q that is
// not positive. The segment below the grid (0) holds alpha_s(Q0^2) and g as its
// first two coefficients, and the one above it the last value at both ends (see
// AlphaSGrid::initialize_coefficients).
template <typename T>
KERNELSPEC void kernel_interpolate_alpha_s(
    FIn<T, 0> q, FIn<T, 1> grid_logq2, FIn<T, 2> grid_coeffs, FOut<T, 0> alpha_s
) {
    auto logq2 = log(q * q);
    FVal<T> logq2_low, logq2_high;
    std::size_t q2_index = binary_search<T>(logq2, grid_logq2, logq2_low, logq2_high);
    // below the grid in a branch, so that the points within it take no
    // exponential (q that is not positive or not a number falls below the grid)
    if (q2_index == 0) {
        FVal<T> below = grid_coeffs.get(0, 0) *
            exp(grid_coeffs.get(1, 0) * (logq2 - FVal<T>(grid_logq2[1])));
        // (zero for q that is not positive, as in the padding of a batch, where
        // the power diverges)
        alpha_s = where(FVal<T>(q) > 0., below, FVal<T>(0.));
        return;
    }

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

} // namespace kernels
} // namespace madspace

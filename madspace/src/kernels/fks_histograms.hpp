#pragma once

#include "fks_cuts.hpp"

namespace madspace {
namespace kernels {

// The histograms of the FKS integrand at fixed order, with the event and its
// counter-events of one point in the same histograms, as the analyses of
// MadFKS fill them (fill_plots of fks_singular.f, HwU_fill and HwU_add_points
// of HwU.f).

// The momenta of one kinematics with its QCD partons clustered into jets: the
// incoming particles and the other final-state particles in their order, then
// the jets of fks_cluster_jets by decreasing pt, and zero momenta in the slots
// without a jet, with the number of jets. parton holds per configuration 1 for
// a final-state QCD parton of its process (the momenta of the kinematics are
// in the order of that process), options jetalgo, jetradius, ptj and etaj,
// and particle_count the size of jet_momenta.
template <typename T>
KERNELSPEC void kernel_fks_jets(
    FIn<T, 2> p,
    IIn<T, 0> index,
    IIn<T, 2> parton,
    FIn<T, 1> options,
    IIn<T, 0> particle_count,
    FOut<T, 2> jet_momenta,
    IOut<T, 0> jet_count
) {
    std::size_t c = single_index(index);
    auto partons_of = parton[c];
    FVal<T> jetalgo = options[0], jetradius = options[1], ptj = options[2],
            etaj = options[3];
    std::size_t n = p.size(), slots = jet_momenta.size();
    FourMom<T> partons[FKS_JET_PARTONS_MAX];
    std::size_t parton_count = 0, kept = 0;
    for (std::size_t i = 0; i < n; ++i) {
        auto p_i = load_mom<T>(p[i]);
        if (single_index(partons_of[i]) == 0) {
            for (std::size_t k = 0; k < 4; ++k) {
                jet_momenta[kept][k] = p_i[k];
            }
            ++kept;
        } else if (parton_count < FKS_JET_PARTONS_MAX) {
            partons[parton_count] = p_i;
            ++parton_count;
        }
    }
    FourMom<T> jets[FKS_JET_PARTONS_MAX];
    std::size_t count =
        fks_cluster_jets<T>(partons, parton_count, jetalgo, jetradius, ptj, etaj, jets);
    // by decreasing pt, as sorted_by_pt of fastjet
    for (std::size_t i = 1; i < count; ++i) {
        for (std::size_t j = i; j > 0; --j) {
            FVal<T> pt2_j = jets[j][1] * jets[j][1] + jets[j][2] * jets[j][2];
            FVal<T> pt2_before =
                jets[j - 1][1] * jets[j - 1][1] + jets[j - 1][2] * jets[j - 1][2];
            if (!(pt2_j > pt2_before)) {
                break;
            }
            auto swap = jets[j];
            jets[j] = jets[j - 1];
            jets[j - 1] = swap;
        }
    }
    std::size_t room = slots - kept;
    count = count < room ? count : room;
    for (std::size_t s = 0; s < room; ++s) {
        for (std::size_t k = 0; k < 4; ++k) {
            jet_momenta[kept + s][k] = s < count ? jets[s][k] : FVal<T>(0.);
        }
    }
    jet_count = static_cast<int>(count);
}

// The weights of the kinematics of one point for a histogram, as HwU_fill and
// HwU_add_points of HwU.f: the kinematics that fall into one bin add up there
// before the square of the point enters the error. values and weights hold
// the observable and the weight of each kinematics, jets its number of jets,
// and options min, max, bin_count and the jets the observable needs. A
// kinematics with fewer jets or a value that is not finite does not count.
// The bins are bin_count equal ones between min and max, with an underflow
// and an overflow bin. The first kinematics of each bin takes the sum of the
// weights there and the others zero, and center is a value in its bin (the
// middle of an inner bin, two bin widths outside the range otherwise), so
// that the histogram of the centers with the merged weights has the sums of
// the point and their squares.
template <typename T>
KERNELSPEC void kernel_fks_histogram_weights(
    FIn<T, 1> values,
    FIn<T, 1> weights,
    IIn<T, 1> jets,
    FIn<T, 1> options,
    FOut<T, 1> center,
    FOut<T, 1> merged
) {
    FVal<T> lower = options[0], upper = options[1];
    int bin_count = static_cast<int>(FVal<T>(options[2]));
    int needed = static_cast<int>(FVal<T>(options[3]));
    FVal<T> width = (upper - lower) / bin_count;
    std::size_t count = values.size();
    int bins[64];
    for (std::size_t k = 0; k < count && k < 64; ++k) {
        FVal<T> x = values[k];
        if (!(fabs(x) <= 1e300) || single_index(jets[k]) < needed) {
            // no bin: the kinematics does not count
            bins[k] = -1;
            center[k] = lower - 2. * width;
            merged[k] = 0.;
            continue;
        }
        FVal<T> r = (x - lower) / width;
        int bin =
            r < 0. ? 0 : (r >= bin_count ? bin_count + 1 : static_cast<int>(r) + 1);
        bins[k] = bin;
        center[k] = bin == 0 ? lower - 2. * width
            : bin == bin_count + 1
            ? upper + 2. * width
            : lower + (bin - 0.5) * width;
        merged[k] = 0.;
        // the sum of the kinematics of this bin goes to the first of them
        std::size_t first = k;
        for (std::size_t l = 0; l < k; ++l) {
            if (bins[l] == bin) {
                first = l;
                break;
            }
        }
        merged[first] = FVal<T>(merged[first]) + FVal<T>(weights[k]);
    }
}

} // namespace kernels
} // namespace madspace

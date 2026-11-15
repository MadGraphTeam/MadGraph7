#pragma once

#include "cuts.hpp"

namespace madspace {
namespace kernels {

// The histograms of an NLO integrand at fixed order, with the kinematics of
// one point (an event and its counter-events) in the same histograms, as the
// analyses of MadFKS fill them (fill_plots of fks_singular.f, HwU_fill and
// HwU_add_points of HwU.f).

// The momenta of one kinematics with its QCD partons clustered into jets: the
// incoming particles and the other final-state particles in their order, then
// the jets of nlo_cluster_jets by decreasing pt, and zero momenta in the slots
// without a jet, with the number of jets. parton holds per flavor layout 1 for
// a final-state QCD parton (the momenta of the kinematics are in the order of
// the layout of the event), options jetalgo, jetradius, ptj and etaj, and
// particle_count the size of jet_momenta.
template <typename T>
KERNELSPEC void kernel_nlo_jets(
    FIn<T, 2> p,
    IIn<T, 0> index,
    ISIn<T, 2> parton,
    FSIn<T, 1> options,
    IIn<T, 0> particle_count,
    FOut<T, 2> jet_momenta,
    IOut<T, 0> jet_count
) {
    double jetalgo = single_value(options[0]), jetradius = single_value(options[1]);
    double ptj = single_value(options[2]), etaj = single_value(options[3]);
    std::size_t n = p.size(), slots = jet_momenta.size();
    for_each_value(IVal<T>(index), [&](std::size_t c, BVal<T> lanes) {
        auto partons_of = parton[c];
        FourMom<T> partons[NLO_JET_PARTONS_MAX];
        BVal<T> member[NLO_JET_PARTONS_MAX];
        std::size_t parton_count = 0, kept = 0;
        for (std::size_t i = 0; i < n; ++i) {
            auto p_i = load_mom<T>(p[i]);
            if (single_index(partons_of[i]) == 0) {
                for (std::size_t k = 0; k < 4; ++k) {
                    masked_store(jet_momenta[kept][k], lanes, p_i[k]);
                }
                ++kept;
            } else if (parton_count < NLO_JET_PARTONS_MAX) {
                partons[parton_count] = p_i;
                member[parton_count] = true;
                ++parton_count;
            }
        }
        FourMom<T> jets[NLO_JET_PARTONS_MAX];
        IVal<T> count = nlo_cluster_jets<T>(
            partons, member, parton_count, jetalgo, jetradius, ptj, etaj, jets
        );
        // by decreasing pt, as sorted_by_pt of fastjet: the insertion of each
        // jet moves it down while its pt is larger
        for (std::size_t i = 1; i < parton_count; ++i) {
            BVal<T> moving = IVal<T>(static_cast<int>(i)) < count;
            for (std::size_t j = i; j > 0; --j) {
                FVal<T> pt2_j = jets[j][1] * jets[j][1] + jets[j][2] * jets[j][2];
                FVal<T> pt2_before =
                    jets[j - 1][1] * jets[j - 1][1] + jets[j - 1][2] * jets[j - 1][2];
                moving = moving & (pt2_j > pt2_before);
                FourMom<T> jet_j = jets[j];
                jets[j] = nlo_where<T>(moving, jets[j - 1], jet_j);
                jets[j - 1] = nlo_where<T>(moving, jet_j, jets[j - 1]);
            }
        }
        IVal<T> room = static_cast<int>(slots - kept);
        IVal<T> shown = where(count < room, count, room);
        for (std::size_t s = 0; s < slots - kept; ++s) {
            BVal<T> filled = IVal<T>(static_cast<int>(s)) < shown;
            for (std::size_t k = 0; k < 4; ++k) {
                FVal<T> value = s < NLO_JET_PARTONS_MAX
                    ? where(filled, jets[s][k], FVal<T>(0.))
                    : FVal<T>(0.);
                masked_store(jet_momenta[kept + s][k], lanes, value);
            }
        }
        masked_store(jet_count, lanes, shown);
    });
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
KERNELSPEC void kernel_nlo_histogram_weights(
    FIn<T, 1> values,
    FIn<T, 1> weights,
    IIn<T, 1> jets,
    FSIn<T, 1> options,
    FOut<T, 1> center,
    FOut<T, 1> merged
) {
    double lower = single_value(options[0]), upper = single_value(options[1]);
    int bin_count = static_cast<int>(single_value(options[2]));
    int needed = static_cast<int>(single_value(options[3]));
    double width = (upper - lower) / bin_count;
    std::size_t count = values.size();
    IVal<T> bins[64];
    for (std::size_t k = 0; k < count && k < 64; ++k) {
        FVal<T> x = values[k];
        // without a bin the kinematics does not count
        BVal<T> counted = (fabs(x) <= 1e300) & !(IVal<T>(jets[k]) < IVal<T>(needed));
        FVal<T> r = (x - lower) / width;
        BVal<T> inner = (r >= 0.) & (r < bin_count);
        IVal<T> bin = where(
            r < 0.,
            IVal<T>(0),
            where(
                inner,
                IVal<T>(where(inner, r, FVal<T>(0.))) + IVal<T>(1),
                IVal<T>(bin_count + 1)
            )
        );
        FVal<T> outside = lower - 2. * width;
        FVal<T> middle = lower + (FVal<T>(bin) - 0.5) * width;
        FVal<T> mid_or_edge = where(
            bin == IVal<T>(0),
            outside,
            where(bin == IVal<T>(bin_count + 1), FVal<T>(upper + 2. * width), middle)
        );
        bins[k] = where(counted, bin, IVal<T>(-1));
        center[k] = where(counted, mid_or_edge, outside);
        merged[k] = 0.;
        // the sum of the kinematics of this bin goes to the first of them
        IVal<T> first = static_cast<int>(k);
        for (std::size_t l = k; l > 0; --l) {
            first = where(bins[l - 1] == bin, IVal<T>(static_cast<int>(l - 1)), first);
        }
        FVal<T> weight = weights[k];
        for (std::size_t l = 0; l <= k; ++l) {
            BVal<T> here = counted & (first == IVal<T>(static_cast<int>(l)));
            merged[l] = where(here, FVal<T>(merged[l]) + weight, FVal<T>(merged[l]));
        }
    }
}

} // namespace kernels
} // namespace madspace
